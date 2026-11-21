import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

# Load .env before importing app modules that read env vars at import time
load_dotenv()

from app.core.cache import (
    TwoTierCache,
    build_cached_openai_response,
    build_cached_sse_chunks,
)
from app.core.guardrails import redact_messages
from app.core.providers import (
    RETRYABLE_STATUS_CODES,
    ProviderRegistry,
    ProviderTarget,
)
from app.core.router import ComplexityRouter
from app.core.security import chat_rate_limit
from app.core.telemetry import TelemetryLogger
from app.shop.rag import ShopRAGStore
from app.shop.routes import shop_router

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://gateway:gateway_password@localhost:5432/telemetry"
)
CACHE_DISTANCE_THRESHOLD = float(os.getenv("CACHE_DISTANCE_THRESHOLD", "0.25"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
ROUTER_CONFIDENCE_MARGIN = float(os.getenv("ROUTER_CONFIDENCE_MARGIN", "0.04"))


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage]
    temperature: float | None = None
    stream: bool = False
    model_config = ConfigDict(extra="allow")


@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
    timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
    app.state.http_client = httpx.AsyncClient(limits=limits, timeout=timeout)

    app.state.providers = ProviderRegistry()
    app.state.cache = TwoTierCache(
        redis_url=REDIS_URL,
        openai_api_key=OPENAI_API_KEY,
        distance_threshold=CACHE_DISTANCE_THRESHOLD,
        ttl_seconds=CACHE_TTL_SECONDS,
    )
    app.state.router = ComplexityRouter(
        openai_api_key=OPENAI_API_KEY,
        confidence_margin=ROUTER_CONFIDENCE_MARGIN,
    )
    await app.state.router.initialize(app.state.http_client)

    app.state.telemetry = TelemetryLogger(database_url=DATABASE_URL)
    await app.state.telemetry.connect()

    app.state.shop_rag = ShopRAGStore(
        redis_url=REDIS_URL,
        openai_api_key=OPENAI_API_KEY,
    )
    app.state.shop_rag.ensure_index()

    yield
    await app.state.telemetry.close()
    await app.state.cache.close()
    await app.state.http_client.aclose()


app = FastAPI(title="Cost-Aware AI Gateway", lifespan=lifespan)
app.include_router(shop_router)


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse("app/static/index.html")


async def stream_upstream(
    client: httpx.AsyncClient,
    target: ProviderTarget,
    payload: dict[str, Any],
    start_time: float,
    cache: TwoTierCache,
    telemetry: TelemetryLogger,
    cacheable_prompt: str | None,
    tenant_id: str,
    cache_model_key: str,
    cache_status: str,
    router_reason: str,
    pii_found: list[str],
) -> AsyncGenerator[bytes, None]:
    ttft_recorded = False
    headers = {"Authorization": f"Bearer {target.api_key}"}
    collected_chunks: list[str] = []

    try:
        async with client.stream(
            "POST", target.chat_completions_url, json=payload, headers=headers
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                print(
                    f"[UPSTREAM ERROR] ({target.provider}/{target.model}) "
                    f"{response.status_code}: {error_body.decode()}"
                )
                raise HTTPException(
                    status_code=response.status_code, detail=error_body.decode()
                )

            async for line in response.aiter_lines():
                if not line:
                    continue
                if not ttft_recorded:
                    ttft_ms = (time.perf_counter() - start_time) * 1000
                    print(f"[STREAM] {target.provider}/{target.model} | TTFT: {ttft_ms:.1f}ms")
                    ttft_recorded = True

                if cacheable_prompt and line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk_json = json.loads(line[6:])
                        delta = chunk_json["choices"][0].get("delta", {}).get("content")
                        if delta:
                            collected_chunks.append(delta)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

                yield f"{line}\n\n".encode("utf-8")

        if cacheable_prompt and collected_chunks:
            full_text = "".join(collected_chunks)
            await cache.set(cacheable_prompt, full_text, tenant_id, cache_model_key)

    except asyncio.CancelledError:
        print("[STREAM] Client disconnected early — cancelled upstream request.")
        raise
    finally:
        total_ms = (time.perf_counter() - start_time) * 1000
        print(f"[STREAM] Total latency: {total_ms:.1f}ms")
        # Estimate streamed token counts (~4 chars per token) for telemetry
        est_out_tokens = max(1, len("".join(collected_chunks)) // 4)
        await telemetry.log_request(
            tenant_id=tenant_id,
            cache_status=cache_status,
            routed_tier=target.tier,
            provider=target.provider,
            model=target.model,
            router_reason=router_reason,
            failover_used=False,
            pii_redacted=pii_found,
            prompt_tokens=40,
            completion_tokens=est_out_tokens,
            latency_ms=total_ms,
        )


@app.get("/health")
async def health(request: Request):
    """Deep health check verifying Redis 8 and PostgreSQL connectivity."""
    providers: ProviderRegistry = request.app.state.providers
    redis_ok = False
    postgres_ok = False

    try:
        redis_client = request.app.state.shop_rag.index.client
        if redis_client is not None:
            redis_ok = bool(await asyncio.to_thread(redis_client.ping))
    except Exception:
        redis_ok = False

    try:
        summary = await request.app.state.telemetry.get_summary()
        postgres_ok = "error" not in summary
    except Exception:
        postgres_ok = False

    healthy = redis_ok and postgres_ok
    payload = {
        "status": "ok" if healthy else "degraded",
        "dependencies": {"redis": redis_ok, "postgres": postgres_ok},
        "fast_tier": f"{providers.fast_target.provider}/{providers.fast_target.model}",
        "strong_tier": f"{providers.strong_target.provider}/{providers.strong_target.model}",
        "fallback_tier": f"{providers.default_fallback.provider}/{providers.default_fallback.model}",
        "cache_threshold": CACHE_DISTANCE_THRESHOLD,
    }
    return JSONResponse(content=payload, status_code=200 if healthy else 503)


@app.get("/api/telemetry")
async def get_telemetry(request: Request):
    """Live PostgreSQL metrics endpoint for benchmarks and the Gateway X-Ray UI."""
    telemetry: TelemetryLogger = request.app.state.telemetry
    return await telemetry.get_summary()


@app.post("/v1/chat/completions", dependencies=[Depends(chat_rate_limit)])
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    client: httpx.AsyncClient = request.app.state.http_client
    cache: TwoTierCache = request.app.state.cache
    router: ComplexityRouter = request.app.state.router
    providers: ProviderRegistry = request.app.state.providers
    telemetry: TelemetryLogger = request.app.state.telemetry

    payload = body.model_dump(exclude_none=True)
    start_time = time.perf_counter()

    tenant_id = request.headers.get("X-Tenant-ID", "default")
    cache_control = request.headers.get("Cache-Control")
    simulate_failure = request.headers.get("X-Simulate-Failure", "").lower() == "true"
    requested_model = body.model

    # --- 1. PII REDACTION GUARDRAIL (Runs before Cache & LLM) ---
    payload["messages"], pii_found = redact_messages(payload["messages"])
    if pii_found:
        print(f"[GUARDRAIL] Redacted PII types: {pii_found}")
    pii_header = ",".join(pii_found) if pii_found else "none"

    cacheable_prompt = cache.extract_cacheable_prompt(
        messages=payload["messages"],
        temperature=body.temperature,
        cache_control=cache_control,
    )

    cache_status = "BYPASS"

    # --- 2. TWO-TIER CACHE LOOKUP ---
    if cacheable_prompt:
        lookup = await cache.get(
            cacheable_prompt, tenant_id=tenant_id, model=requested_model
        )
        cache_status = lookup.status

        if lookup.status in ("EXACT_HIT", "SEMANTIC_HIT") and lookup.response_text:
            total_ms = (time.perf_counter() - start_time) * 1000
            dist_str = f"{lookup.distance:.4f}" if lookup.distance is not None else "n/a"
            print(
                f"[CACHE] {lookup.status} | Tenant: {tenant_id} | "
                f"Distance: {dist_str} | Latency: {total_ms:.1f}ms"
            )

            background_tasks.add_task(
                telemetry.log_request,
                tenant_id=tenant_id,
                cache_status=lookup.status,
                routed_tier="cache",
                provider="redis",
                model="cached",
                router_reason="cache_hit",
                failover_used=False,
                pii_redacted=pii_found,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=total_ms,
            )

            response_headers = {
                "X-Cache-Status": lookup.status,
                "X-Cache-Distance": dist_str,
                "X-Routed-Tier": "cache",
                "X-Routed-Provider": "redis",
                "X-Routed-Model": "cached",
                "X-Router-Reason": "cache_hit",
                "X-Failover-Used": "false",
                "X-PII-Redacted": pii_header,
            }

            if body.stream:
                async def replay_stream():
                    for chunk in build_cached_sse_chunks(requested_model, lookup.response_text):
                        yield chunk

                return StreamingResponse(
                    replay_stream(),
                    media_type="text/event-stream",
                    headers={
                        **response_headers,
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )

            return JSONResponse(
                content=build_cached_openai_response(requested_model, lookup.response_text),
                headers=response_headers,
            )

    # --- 3. COMPLEXITY ROUTING ---
    if requested_model.lower() == "auto":
        decision = await router.route(client, payload["messages"])
        primary_target = (
            providers.fast_target
            if decision.tier == "fast"
            else providers.strong_target
        )
        router_reason = decision.reason
        print(
            f"[ROUTER] {decision.tier.upper()} -> {primary_target.provider}/{primary_target.model} | "
            f"{decision.reason} | {decision.routing_ms:.1f}ms"
        )
    else:
        primary_target = providers.resolve_explicit_model(requested_model)
        router_reason = "explicit_override"
        print(f"[ROUTER] OVERRIDE -> {primary_target.provider}/{primary_target.model}")

    # --- 4. UPSTREAM EXECUTION WITH AUTOMATIC FAILOVER ---
    if body.stream:
        payload["model"] = primary_target.model
        return StreamingResponse(
            stream_upstream(
                client,
                primary_target,
                payload,
                start_time,
                cache,
                telemetry,
                cacheable_prompt,
                tenant_id,
                requested_model,
                cache_status,
                router_reason,
                pii_found,
            ),
            media_type="text/event-stream",
            headers={
                "X-Cache-Status": cache_status,
                "X-Routed-Tier": primary_target.tier,
                "X-Routed-Provider": primary_target.provider,
                "X-Routed-Model": primary_target.model,
                "X-Router-Reason": router_reason,
                "X-Failover-Used": "false",
                "X-PII-Redacted": pii_header,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    fallback_chain = providers.get_fallback_chain(primary_target)
    last_error_status = 502
    last_error_detail = "All upstream providers failed."

    for attempt_idx, target in enumerate(fallback_chain):
        failover_used = attempt_idx > 0
        attempt_payload = {**payload, "model": target.model}
        headers = {"Authorization": f"Bearer {target.api_key}"}

        try:
            # Optional deterministic failover test via header
            if simulate_failure and attempt_idx == 0:
                print(
                    f"[FAILOVER] Simulated 503 on {target.provider}/{target.model} "
                    f"-> switching to backup..."
                )
                last_error_status = 503
                last_error_detail = "Simulated primary 503 outage"
                continue

            response = await client.post(
                target.chat_completions_url, json=attempt_payload, headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                total_ms = (time.perf_counter() - start_time) * 1000
                usage = data.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))

                print(
                    f"[SYNC] {cache_status} | {target.provider}/{target.model} | "
                    f"Failover: {failover_used} | Latency: {total_ms:.1f}ms"
                )

                if cacheable_prompt:
                    try:
                        answer_text = data["choices"][0]["message"]["content"]
                        await cache.set(
                            cacheable_prompt,
                            answer_text,
                            tenant_id=tenant_id,
                            model=requested_model,
                        )
                    except (KeyError, IndexError):
                        pass

                # Non-blocking telemetry write (0ms added to response latency)
                background_tasks.add_task(
                    telemetry.log_request,
                    tenant_id=tenant_id,
                    cache_status=cache_status,
                    routed_tier=target.tier,
                    provider=target.provider,
                    model=target.model,
                    router_reason=router_reason,
                    failover_used=failover_used,
                    pii_redacted=pii_found,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=total_ms,
                )

                return JSONResponse(
                    content=data,
                    headers={
                        "X-Cache-Status": cache_status,
                        "X-Routed-Tier": target.tier,
                        "X-Routed-Provider": target.provider,
                        "X-Routed-Model": target.model,
                        "X-Router-Reason": router_reason,
                        "X-Failover-Used": str(failover_used).lower(),
                        "X-PII-Redacted": pii_header,
                    },
                )

            last_error_status = response.status_code
            last_error_detail = response.text

            if response.status_code in RETRYABLE_STATUS_CODES and attempt_idx + 1 < len(fallback_chain):
                print(
                    f"[FAILOVER] {target.provider}/{target.model} returned {response.status_code} "
                    f"-> failing over to {fallback_chain[attempt_idx + 1].provider}/"
                    f"{fallback_chain[attempt_idx + 1].model}"
                )
                continue

            print(
                f"[UPSTREAM ERROR] ({target.provider}/{target.model}) "
                f"{response.status_code}: {response.text}"
            )
            raise HTTPException(status_code=response.status_code, detail=response.text)

        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_error_status = 504
            last_error_detail = str(exc)
            if attempt_idx + 1 < len(fallback_chain):
                print(
                    f"[FAILOVER] {target.provider}/{target.model} timed out ({exc}) "
                    f"-> failing over to {fallback_chain[attempt_idx + 1].provider}/"
                    f"{fallback_chain[attempt_idx + 1].model}"
                )
                continue
            raise HTTPException(status_code=504, detail=f"Upstream timeout: {exc}")

    raise HTTPException(status_code=last_error_status, detail=last_error_detail)