import time
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

# Reference utterances embedded ONCE at server startup
SIMPLE_EXAMPLES = [
    "Fix the grammar and spelling in this sentence.",
    "What is the capital of France?",
    "Summarize this short paragraph in two bullet points.",
    "Translate 'good morning' into Spanish.",
    "Convert this list of names into comma-separated JSON.",
    "Define HTTP status code 404 in one sentence.",
    "Classify this customer review as positive, neutral, or negative.",
]

COMPLEX_EXAMPLES = [
    "Design a fault-tolerant distributed lock using Redis and explain split-brain failure modes.",
    "Debug this asynchronous Python race condition and refactor it for thread safety.",
    "Compare B-Trees vs LSM-Trees for write-heavy time-series workloads with trade-offs.",
    "Write a production-ready SQL query with window functions and explain the execution plan.",
    "Prove why this dynamic programming recurrence has O(n log n) time complexity.",
    "Architect a multi-region microservices system with active-active failover.",
    "Analyze the security vulnerabilities in this OAuth2 PKCE flow and propose mitigations.",
]

COMPLEX_KEYWORDS = (
    "architect",
    "distributed",
    "race condition",
    "deadlock",
    "refactor",
    "debug",
    "trade-off",
    "tradeoff",
    "time complexity",
    "space complexity",
    "proof",
    "prove",
    "security",
    "vulnerability",
    "microservices",
    "system design",
    "calculate",
    "budget",
    "itemized",
    "sundae bar",
)


@dataclass
class RouteDecision:
    tier: str              # "fast" or "strong"
    reason: str            # Human-readable explanation for headers & logs
    simple_score: float
    complex_score: float
    routing_ms: float


class ComplexityRouter:
    """
    Hybrid 2-stage router:
      1) Zero-cost heuristic check for obvious complex prompts (<1ms).
      2) Semantic cosine similarity against pre-embedded reference centroids.
    Defaults safely to 'strong' whenever confidence is below the margin.
    """

    def __init__(self, openai_api_key: str, confidence_margin: float = 0.04) -> None:
        self.openai_api_key = openai_api_key
        self.confidence_margin = confidence_margin
        self.simple_matrix: np.ndarray | None = None
        self.complex_matrix: np.ndarray | None = None

    async def _embed_texts(
        self, client: httpx.AsyncClient, texts: list[str]
    ) -> np.ndarray:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.openai_api_key}"},
            json={"model": "text-embedding-3-small", "input": texts},
        )
        resp.raise_for_status()
        vectors = [item["embedding"] for item in resp.json()["data"]]
        arr = np.array(vectors, dtype=np.float32)
        # L2-normalize so dot product == cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-12, None)

    async def initialize(self, client: httpx.AsyncClient) -> None:
        """Embed reference utterances once during FastAPI lifespan startup."""
        all_texts = SIMPLE_EXAMPLES + COMPLEX_EXAMPLES
        embeddings = await self._embed_texts(client, all_texts)
        split = len(SIMPLE_EXAMPLES)
        self.simple_matrix = embeddings[:split]
        self.complex_matrix = embeddings[split:]
        print(f"[ROUTER] Initialized with {split} simple & {len(COMPLEX_EXAMPLES)} complex anchors.")

    @staticmethod
    def _extract_latest_user_text(messages: list[dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg.get("content", "")).strip()
        return ""

    async def route(
        self, client: httpx.AsyncClient, messages: list[dict[str, Any]]
    ) -> RouteDecision:
        start = time.perf_counter()
        text = self._extract_latest_user_text(messages)
        lowered = text.lower()

        # --- STAGE 1: Fast Heuristic Pre-Filter (<1ms, $0) ---
        if "```" in text:
            return RouteDecision(
                tier="strong",
                reason="heuristic:code_block",
                simple_score=0.0,
                complex_score=1.0,
                routing_ms=(time.perf_counter() - start) * 1000,
            )

        if len(text) > 800 or len(messages) > 6:
            return RouteDecision(
                tier="strong",
                reason="heuristic:long_context",
                simple_score=0.0,
                complex_score=1.0,
                routing_ms=(time.perf_counter() - start) * 1000,
            )

        for kw in COMPLEX_KEYWORDS:
            if kw in lowered:
                return RouteDecision(
                    tier="strong",
                    reason=f"heuristic:keyword({kw})",
                    simple_score=0.0,
                    complex_score=1.0,
                    routing_ms=(time.perf_counter() - start) * 1000,
                )

        # --- STAGE 2: Semantic Similarity Classifier ---
        if self.simple_matrix is None or self.complex_matrix is None:
            return RouteDecision(
                tier="strong",
                reason="fallback:router_uninitialized",
                simple_score=0.0,
                complex_score=0.0,
                routing_ms=(time.perf_counter() - start) * 1000,
            )

        prompt_vec = (await self._embed_texts(client, [text]))[0]

        # Compare max similarity to the closest anchor in each class
        simple_score = float(np.max(self.simple_matrix @ prompt_vec))
        complex_score = float(np.max(self.complex_matrix @ prompt_vec))
        margin = simple_score - complex_score
        elapsed = (time.perf_counter() - start) * 1000

        # Only route to FAST if simple beats complex by at least confidence_margin
        if margin >= self.confidence_margin:
            return RouteDecision(
                tier="fast",
                reason=f"semantic:simple(margin=+{margin:.3f})",
                simple_score=simple_score,
                complex_score=complex_score,
                routing_ms=elapsed,
            )

        return RouteDecision(
            tier="strong",
            reason=f"semantic:complex_or_uncertain(margin={margin:+.3f})",
            simple_score=simple_score,
            complex_score=complex_score,
            routing_ms=elapsed,
        )