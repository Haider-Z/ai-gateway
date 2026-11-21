import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import redis.asyncio as aioredis
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.query.filter import Tag
from redisvl.utils.vectorize import OpenAITextVectorizer


@dataclass
class CacheLookupResult:
    status: str  # "EXACT_HIT" | "SEMANTIC_HIT" | "MISS" | "BYPASS"
    response_text: str | None = None
    distance: float | None = None
    lookup_ms: float = 0.0


class TwoTierCache:
    """
    Tier 1: Exact SHA-256 lookup in Redis (~2ms, zero embedding cost).
    Tier 2: Vector similarity lookup in Redis 8 via RedisVL SemanticCache.
    """

    def __init__(
        self,
        redis_url: str,
        openai_api_key: str,
        distance_threshold: float = 0.15,
        ttl_seconds: int = 3600,
    ):
        self.ttl_seconds = ttl_seconds
        self.exact_redis = aioredis.from_url(redis_url, decode_responses=True)

        # Lightweight cloud embeddings (no local PyTorch/model weights needed)
        vectorizer = OpenAITextVectorizer(
            model="text-embedding-3-small",
            api_config={"api_key": openai_api_key},
        )

        # RedisVL manages the vector index in Redis 8 + tenant tag filtering
        self.semantic_cache = SemanticCache(
            name="ai_gateway_cache",
            redis_url=redis_url,
            distance_threshold=distance_threshold,
            ttl=ttl_seconds,
            vectorizer=vectorizer,
            filterable_fields=[
                {"name": "tenant_id", "type": "tag"},
                {"name": "model", "type": "tag"},
            ],
        )

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        """Collapse extra whitespace and lowercase for Tier 1 exact matching."""
        return " ".join(text.strip().lower().split())

    @staticmethod
    def _exact_key(tenant_id: str, model: str, normalized_prompt: str) -> str:
        digest = hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()
        return f"exact_cache:{tenant_id}:{model}:{digest}"

    @staticmethod
    def extract_cacheable_prompt(
        messages: list[dict[str, Any]],
        temperature: float | None,
        cache_control: str | None,
    ) -> str | None:
        """
        Decide whether a request is safe to cache.
        Returns the prompt string if cacheable, or None to bypass the cache.
        """
        # 1. Respect explicit client opt-out
        if cache_control and "no-cache" in cache_control.lower():
            return None

        # 2. Skip high-temperature creative prompts where variety is desired
        if temperature is not None and temperature > 0.7:
            return None

        # 3. Only cache single-turn prompts (avoid mis-serving multi-turn chat context)
        user_messages = [m for m in messages if m.get("role") == "user"]
        if len(user_messages) != 1:
            return None

        return user_messages[0].get("content", "").strip() or None

    async def get(
        self, prompt: str, tenant_id: str, model: str
    ) -> CacheLookupResult:
        start = time.perf_counter()
        normalized = self._normalize_prompt(prompt)

        # --- TIER 1: Exact Match ---
        exact_key = self._exact_key(tenant_id, model, normalized)
        exact_hit = await self.exact_redis.get(exact_key)
        if exact_hit is not None:
            elapsed = (time.perf_counter() - start) * 1000
            return CacheLookupResult(
                status="EXACT_HIT",
                response_text=exact_hit,
                distance=0.0,
                lookup_ms=elapsed,
            )

        # --- TIER 2: Semantic Vector Match ---
        # Filter ensures Tenant A never sees Tenant B's cached answers
        tenant_filter = (Tag("tenant_id") == tenant_id) & (Tag("model") == model)
        results = await self.semantic_cache.acheck(
            prompt=prompt,
            filter_expression=tenant_filter,
            num_results=1,
        )

        elapsed = (time.perf_counter() - start) * 1000
        if results:
            hit = results[0]
            dist = hit.get("vector_distance", hit.get("cosine_distance"))
            return CacheLookupResult(
                status="SEMANTIC_HIT",
                response_text=hit["response"],
                distance=float(dist) if dist is not None else None,
                lookup_ms=elapsed,
            )

        return CacheLookupResult(status="MISS", lookup_ms=elapsed)

    async def set(
        self, prompt: str, response_text: str, tenant_id: str, model: str
    ) -> None:
        """Write successful LLM responses to both Tier 1 and Tier 2."""
        if not response_text or not response_text.strip():
            return

        normalized = self._normalize_prompt(prompt)
        exact_key = self._exact_key(tenant_id, model, normalized)

        # Store in Tier 1 (Exact) with TTL
        await self.exact_redis.set(exact_key, response_text, ex=self.ttl_seconds)

        # Store in Tier 2 (Semantic) with tenant & model tags
        await self.semantic_cache.astore(
            prompt=prompt,
            response=response_text,
            filters={"tenant_id": tenant_id, "model": model},
            ttl=self.ttl_seconds,
        )

    async def close(self) -> None:
        await self.exact_redis.aclose()


def build_cached_openai_response(model: str, content: str) -> dict[str, Any]:
    """Wrap cached text into a valid OpenAI ChatCompletion JSON object."""
    return {
        "id": f"chatcmpl-cache-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def build_cached_sse_chunks(model: str, content: str) -> list[bytes]:
    """Replay cached text as OpenAI-compatible SSE chunks for stream=True clients."""
    chunk_payload = {
        "id": f"chatcmpl-cache-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }
        ],
    }
    stop_payload = {
        "id": chunk_payload["id"],
        "object": "chat.completion.chunk",
        "created": chunk_payload["created"],
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return [
        f"data: {json.dumps(chunk_payload)}\n\n".encode("utf-8"),
        f"data: {json.dumps(stop_payload)}\n\n".encode("utf-8"),
        b"data: [DONE]\n\n",
    ]