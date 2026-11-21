import os
import time
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.core.cache import TwoTierCache
from app.core.guardrails import redact_text
from app.core.security import chat_rate_limit, require_admin_key
from app.core.telemetry import TelemetryLogger
from app.shop.rag import ShopRAGStore

MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB limit
INTERNAL_GATEWAY_URL = os.getenv("INTERNAL_GATEWAY_URL", "http://127.0.0.1:8000")

shop_router = APIRouter(prefix="/api/shop", tags=["RAG Knowledge & Chat"])

SYSTEM_PROMPT_TEMPLATE = """You are a precise, helpful AI assistant answering questions strictly from the uploaded knowledge base documents below.

Rules:
- Ground every claim in the retrieved document excerpts provided below. Never invent facts, numbers, dates, or policies not present in the context.
- Be direct, natural, and concise—avoid robotic boilerplate like "Based on the provided documents".
- For quantitative, pricing, or multi-constraint questions, show clear step-by-step reasoning.
- If the uploaded documents do not contain the answer, clearly state what information is missing from the current knowledge base.

--- RETRIEVED KNOWLEDGE BASE CONTEXT ---
{context}
"""


class ShopChatRequest(BaseModel):
    message: str
    simulate_failure: bool = False
    bypass_cache: bool = False


class FAQItem(BaseModel):
    question: str
    answer: str


class FAQWarmRequest(BaseModel):
    faqs: list[FAQItem]


@shop_router.post("/upload", dependencies=[Depends(require_admin_key)])
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload a PDF document into the Redis 8 RAG vector store."""
    safe_filename = os.path.basename(file.filename or "")
    if not safe_filename or not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds 10 MB size limit.")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="File is not a valid PDF.")

    rag: ShopRAGStore = request.app.state.shop_rag
    client: httpx.AsyncClient = request.app.state.http_client
    cache: TwoTierCache = request.app.state.cache

    try:
        result = await rag.ingest_pdf(
            filename=safe_filename,
            pdf_bytes=pdf_bytes,
            client=client,
            cache=cache,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {"status": "indexed", **result}


@shop_router.get("/documents")
async def list_documents(request: Request) -> dict[str, Any]:
    """Lists all indexed PDFs and current cache version."""
    rag: ShopRAGStore = request.app.state.shop_rag
    return {
        "cache_version": rag.kb_version,
        "warmed_faq_count": len(rag.warmed_faqs),
        "documents": list(rag.uploaded_docs.values()),
    }


@shop_router.delete(
    "/documents/{filename:path}",
    dependencies=[Depends(require_admin_key)],
)
async def delete_document(filename: str, request: Request) -> dict[str, Any]:
    """Removes a PDF from Redis 8 vector store and bumps cache version in O(1)."""
    rag: ShopRAGStore = request.app.state.shop_rag
    cache: TwoTierCache = request.app.state.cache
    client: httpx.AsyncClient = request.app.state.http_client

    safe_filename = os.path.basename(filename)
    result = await rag.delete_pdf(filename=safe_filename, cache=cache, client=client)
    return {"status": "deleted", **result}


@shop_router.post("/faqs/warm", dependencies=[Depends(require_admin_key)])
async def warm_faqs(body: FAQWarmRequest, request: Request) -> dict[str, Any]:
    """Pre-loads FAQs + auto-generated paraphrases into Redis Semantic Cache."""
    rag: ShopRAGStore = request.app.state.shop_rag
    cache: TwoTierCache = request.app.state.cache
    client: httpx.AsyncClient = request.app.state.http_client

    count = await rag.warm_faqs(
        [f.model_dump() for f in body.faqs], cache, client=client
    )
    return {"status": "warmed", "faq_count": count, "cache_tenant": rag.cache_tenant_id}


@shop_router.post("/chat", dependencies=[Depends(chat_rate_limit)])
async def shop_chat(
    body: ShopChatRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    client: httpx.AsyncClient = request.app.state.http_client
    cache: TwoTierCache = request.app.state.cache
    rag: ShopRAGStore = request.app.state.shop_rag
    telemetry: TelemetryLogger = request.app.state.telemetry

    tenant_id = rag.cache_tenant_id

    # --- 1. PII REDACTION AT INGRESS (Before Cache & RAG) ---
    clean_message, pii_found = redact_text(body.message)

    # --- 2. CHECK VERSIONED REDIS SEMANTIC CACHE FIRST ---
    if not body.bypass_cache:
        lookup = await cache.get(clean_message, tenant_id=tenant_id, model="auto")
        if lookup.status in ("EXACT_HIT", "SEMANTIC_HIT") and lookup.response_text:
            total_ms = (time.perf_counter() - start_time) * 1000
            background_tasks.add_task(
                telemetry.log_request,
                tenant_id=tenant_id,
                cache_status=lookup.status,
                routed_tier="cache",
                provider="redis",
                model="cached",
                router_reason="warmed_faq_or_cache_hit",
                failover_used=False,
                pii_redacted=pii_found,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=total_ms,
            )
            return {
                "answer": lookup.response_text,
                "xray": {
                    "cache_status": lookup.status,
                    "cache_distance": round(lookup.distance, 4) if lookup.distance is not None else 0.0,
                    "rag_used": False,
                    "rag_ms": 0.0,
                    "rag_chunks": [],
                    "routed_tier": "cache",
                    "provider": "redis",
                    "model": "cached",
                    "router_reason": "warmed_faq_or_cache_hit",
                    "failover_used": False,
                    "pii_redacted": pii_found,
                    "latency_ms": round(total_ms, 1),
                },
            }

    # --- 3. CACHE MISS -> RETRIEVE GROUNDED PDF CHUNKS FROM REDIS 8 ---
    rag_start = time.perf_counter()
    chunks = await rag.retrieve(client, clean_message, top_k=4)
    rag_ms = (time.perf_counter() - rag_start) * 1000

    if chunks:
        context_block = "\n\n".join(
            f"[{c.source} | Page {c.page}]\n{c.content}" for c in chunks
        )
    else:
        context_block = "No PDF documents are currently indexed in the knowledge base."

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_block)

    # --- 4. ROUTE THROUGH REUSABLE AI GATEWAY (/v1/chat/completions) ---
    gateway_url = f"{INTERNAL_GATEWAY_URL.rstrip('/')}/v1/chat/completions"
    gateway_headers = {
        "X-Tenant-ID": tenant_id,
        "Cache-Control": "no-cache" if body.bypass_cache else "default",
    }
    if body.simulate_failure:
        gateway_headers["X-Simulate-Failure"] = "true"

    gateway_payload = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": body.message},
        ],
    }

    gw_resp = await client.post(gateway_url, json=gateway_payload, headers=gateway_headers)
    if gw_resp.status_code != 200:
        raise HTTPException(status_code=gw_resp.status_code, detail=gw_resp.text)

    data = gw_resp.json()
    answer = data["choices"][0]["message"]["content"]
    h = gw_resp.headers
    total_ms = (time.perf_counter() - start_time) * 1000

    return {
        "answer": answer,
        "xray": {
            "cache_status": h.get("x-cache-status", "MISS"),
            "cache_distance": None,
            "rag_used": True,
            "rag_ms": round(rag_ms, 1),
            "rag_chunks": [
                {
                    "source": c.source,
                    "page": c.page,
                    "distance": c.score,
                    "preview": c.content[:140] + "...",
                }
                for c in chunks
            ],
            "routed_tier": h.get("x-routed-tier", "unknown"),
            "provider": h.get("x-routed-provider", "unknown"),
            "model": h.get("x-routed-model", "unknown"),
            "router_reason": h.get("x-router-reason", "unknown"),
            "failover_used": h.get("x-failover-used", "false") == "true",
            "pii_redacted": pii_found,
            "latency_ms": round(total_ms, 1),
        },
    }