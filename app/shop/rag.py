import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from pypdf import PdfReader
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

from app.core.cache import TwoTierCache

SHOP_RAG_INDEX_NAME = "scoop_byte_rag"
UPLOAD_DIR = Path("uploads")


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    page: int
    content: str
    score: float  # Cosine distance (lower = closer match)


class ShopRAGStore:
    """
    Dynamic PDF RAG store backed by Redis 8 vector search.
    Supports live PDF uploads, deletion, sliding-window chunking, and O(1) cache invalidation.
    """

    def __init__(self, redis_url: str, openai_api_key: str) -> None:
        self.redis_url = redis_url
        self.openai_api_key = openai_api_key
        self.kb_version: int = 1
        self.uploaded_docs: dict[str, dict[str, Any]] = {}
        self.warmed_faqs: list[dict[str, str]] = []

        UPLOAD_DIR.mkdir(exist_ok=True)

        schema = {
            "index": {
                "name": SHOP_RAG_INDEX_NAME,
                "prefix": "shop_chunk",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "chunk_id", "type": "tag"},
                {"name": "source", "type": "tag"},
                {"name": "page", "type": "numeric"},
                {"name": "content", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 1536,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32",
                    },
                },
            ],
        }
        self.index = SearchIndex.from_dict(schema, redis_url=self.redis_url)

    @property
    def cache_tenant_id(self) -> str:
        """Versioned tenant namespace so uploading/deleting a PDF invalidates stale cache entries in O(1)."""
        return f"kb_workspace_v{self.kb_version}"

    def ensure_index(self) -> None:
        """Create the Redis 8 vector index if missing and sync existing PDF metadata from Redis."""
        if not self.index.exists():
            self.index.create(overwrite=False)
            print(f"[RAG] Created Redis 8 vector index '{SHOP_RAG_INDEX_NAME}'.")
        else:
            print(f"[RAG] Connected to existing Redis 8 index '{SHOP_RAG_INDEX_NAME}'.")
            self._sync_docs_from_redis()

    def _sync_docs_from_redis(self) -> None:
        """Rebuilds uploaded_docs metadata from existing Redis hashes so restarts preserve the file list."""
        redis_client = self.index.client
        if redis_client is None:
            return

        doc_pages: dict[str, set[int]] = {}
        doc_chunks: dict[str, int] = {}

        for key in redis_client.scan_iter(match="shop_chunk:*"):
            raw_source = redis_client.hget(key, "source")
            raw_page = redis_client.hget(key, "page")
            if not raw_source:
                continue
            source = raw_source.decode("utf-8", errors="ignore") if isinstance(raw_source, bytes) else str(raw_source)
            page = int(raw_page.decode("utf-8", errors="ignore") if isinstance(raw_page, bytes) else (raw_page or 1))
            doc_pages.setdefault(source, set()).add(page)
            doc_chunks[source] = doc_chunks.get(source, 0) + 1

        self.uploaded_docs = {
            src: {
                "filename": src,
                "pages": max(pages) if pages else 1,
                "chunks": doc_chunks[src],
            }
            for src, pages in doc_pages.items()
        }

    @staticmethod
    def chunk_text(
        text: str, chunk_size: int = 700, overlap: int = 120
    ) -> list[str]:
        """Splits page text into overlapping chunks without breaking mid-word."""
        cleaned = " ".join(text.split())
        if not cleaned:
            return []

        chunks: list[str] = []
        start = 0
        length = len(cleaned)

        while start < length:
            end = min(start + chunk_size, length)
            if end < length:
                split_at = cleaned.rfind(". ", start, end)
                if split_at == -1 or split_at <= start + chunk_size // 2:
                    split_at = cleaned.rfind(" ", start, end)
                if split_at > start:
                    end = split_at + 1

            piece = cleaned[start:end].strip()
            if piece:
                chunks.append(piece)

            if end >= length:
                break
            start = max(end - overlap, start + 1)

        return chunks

    async def _embed(self, client: httpx.AsyncClient, texts: list[str]) -> list[bytes]:
        """Batch-embed text chunks with OpenAI text-embedding-3-small."""
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.openai_api_key}"},
            json={"model": "text-embedding-3-small", "input": texts},
        )
        resp.raise_for_status()
        items = resp.json()["data"]
        return [
            np.array(item["embedding"], dtype=np.float32).tobytes()
            for item in items
        ]

    async def ingest_pdf(
        self,
        *,
        filename: str,
        pdf_bytes: bytes,
        client: httpx.AsyncClient,
        cache: TwoTierCache,
    ) -> dict[str, Any]:
        """Parses an uploaded PDF, indexes vectors into Redis 8, and bumps kb_version."""
        # Remove old chunks for this filename first if re-uploading an updated version
        await self._delete_chunks_for_source(filename)

        reader = PdfReader(io.BytesIO(pdf_bytes))
        raw_chunks: list[dict[str, Any]] = []

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for idx, text_chunk in enumerate(self.chunk_text(page_text)):
                raw_id = f"{filename}:p{page_num}:c{idx}:{text_chunk[:40]}"
                chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
                raw_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source": filename,
                        "page": page_num,
                        "content": text_chunk,
                    }
                )

        if not raw_chunks:
            raise ValueError(
                f"No readable text found in '{filename}'. Ensure it is a text-based PDF, not a scanned image."
            )

        (UPLOAD_DIR / filename).write_bytes(pdf_bytes)
        vectors = await self._embed(client, [c["content"] for c in raw_chunks])

        records = [
            {**chunk, "embedding": vec}
            for chunk, vec in zip(raw_chunks, vectors)
        ]
        self.index.load(records, id_field="chunk_id")

        self.uploaded_docs[filename] = {
            "filename": filename,
            "pages": len(reader.pages),
            "chunks": len(records),
        }

        self.kb_version += 1
        if self.warmed_faqs:
            await self.warm_faqs(self.warmed_faqs, cache, client=client)

        print(
            f"[RAG] Ingested '{filename}' ({len(reader.pages)} pages, "
            f"{len(records)} chunks) | Cache version -> v{self.kb_version}"
        )
        return {
            "filename": filename,
            "pages": len(reader.pages),
            "chunks_indexed": len(records),
            "cache_version": self.kb_version,
        }

    async def _delete_chunks_for_source(self, filename: str) -> int:
        """Deletes all Redis vector hashes belonging to a specific PDF filename."""
        deleted = 0
        redis_client = self.index.client
        if redis_client is not None:
            for key in redis_client.scan_iter(match="shop_chunk:*"):
                raw_source = redis_client.hget(key, "source")
                source = (
                    raw_source.decode("utf-8", errors="ignore")
                    if isinstance(raw_source, bytes)
                    else str(raw_source or "")
                )
                if source == filename:
                    redis_client.delete(key)
                    deleted += 1
        return deleted

    async def delete_pdf(
        self,
        *,
        filename: str,
        cache: TwoTierCache,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        """Removes a PDF from Redis 8, deletes it from disk, and bumps kb_version in O(1)."""
        deleted_chunks = await self._delete_chunks_for_source(filename)
        self.uploaded_docs.pop(filename, None)

        safe_path = UPLOAD_DIR / Path(filename).name
        if safe_path.exists():
            safe_path.unlink(missing_ok=True)

        self.kb_version += 1
        if self.warmed_faqs:
            await self.warm_faqs(self.warmed_faqs, cache, client=client)

        print(
            f"[RAG] Deleted '{filename}' ({deleted_chunks} chunks removed) | "
            f"Cache version -> v{self.kb_version}"
        )
        return {
            "filename": filename,
            "deleted_chunks": deleted_chunks,
            "cache_version": self.kb_version,
        }

    async def _generate_paraphrases(
        self, client: httpx.AsyncClient, faqs: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Uses gpt-6-luna once at warm time to generate 3 diverse phrasings per FAQ."""
        questions = [f["question"] for f in faqs]
        prompt = (
            "For each FAQ question below, generate 3 natural paraphrases that a user or recruiter "
            "might ask instead. Vary vocabulary (e.g., role/opportunity, bio/who is) and include both "
            "entity names and pronouns where natural.\n"
            "Return ONLY a valid JSON object mapping each original question string to a list of 3 strings.\n\n"
            f"Questions:\n{json.dumps(questions, indent=2)}"
        )
        try:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openai_api_key}"},
                json={
                    "model": "gpt-6-luna",
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            mapping = json.loads(resp.json()["choices"][0]["message"]["content"])
        except Exception as exc:
            print(f"[RAG] Paraphrase expansion skipped ({exc}); warming base FAQs only.")
            mapping = {}

        expanded: list[dict[str, str]] = []
        for faq in faqs:
            q = faq["question"]
            a = faq["answer"]
            expanded.append({"question": q, "answer": a})
            for alt in mapping.get(q, [])[:3]:
                if isinstance(alt, str) and alt.strip():
                    expanded.append({"question": alt.strip(), "answer": a})
        return expanded

    async def warm_faqs(
        self,
        faqs: list[dict[str, str]],
        cache: TwoTierCache,
        client: httpx.AsyncClient | None = None,
    ) -> int:
        """Pre-loads base FAQs + auto-generated paraphrases into the versioned Redis cache."""
        self.warmed_faqs = faqs
        entries = (
            await self._generate_paraphrases(client, faqs)
            if client is not None
            else faqs
        )
        for item in entries:
            await cache.set(
                prompt=item["question"],
                response_text=item["answer"],
                tenant_id=self.cache_tenant_id,
                model="auto",
            )
        print(
            f"[RAG] Warmed Redis cache ({self.cache_tenant_id}) with "
            f"{len(faqs)} base FAQs ({len(entries)} total entries)."
        )
        return len(entries)

    async def retrieve(
        self, client: httpx.AsyncClient, query: str, top_k: int = 4
    ) -> list[RetrievedChunk]:
        """Vector search over all uploaded PDFs in Redis 8."""
        query_vec = (await self._embed(client, [query]))[0]
        vq = VectorQuery(
            vector=query_vec,
            vector_field_name="embedding",
            return_fields=["chunk_id", "source", "page", "content"],
            num_results=top_k,
        )
        results = self.index.query(vq)
        return [
            RetrievedChunk(
                chunk_id=row.get("chunk_id", ""),
                source=row.get("source", "unknown.pdf"),
                page=int(row.get("page", 0)),
                content=row.get("content", ""),
                score=round(float(row.get("vector_distance", 1.0)), 4),
            )
            for row in results
        ]