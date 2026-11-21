from typing import Any

import asyncpg

# Approximate USD cost per 1,000,000 tokens: (input_per_1m, output_per_1m)
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-6.2-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    "gemini-3.8-flash": (0.15, 0.60),
    "gemini-3.5-flash-lite": (0.075, 0.30),
    "cached": (0.0, 0.0),
}
BASELINE_MODEL = "gpt-6.2-sol"


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_rate, out_rate = MODEL_PRICING_PER_1M.get(model, (0.50, 1.50))
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000


class TelemetryLogger:
    """Non-blocking PostgreSQL telemetry logger using an asyncpg connection pool."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=10)
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS request_logs (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    tenant_id TEXT NOT NULL,
                    cache_status TEXT NOT NULL,
                    routed_tier TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    router_reason TEXT NOT NULL,
                    failover_used BOOLEAN DEFAULT FALSE,
                    pii_redacted TEXT DEFAULT '',
                    prompt_tokens INT DEFAULT 0,
                    completion_tokens INT DEFAULT 0,
                    latency_ms DOUBLE PRECISION NOT NULL,
                    actual_cost_usd DOUBLE PRECISION DEFAULT 0.0,
                    baseline_cost_usd DOUBLE PRECISION DEFAULT 0.0,
                    saved_cost_usd DOUBLE PRECISION DEFAULT 0.0
                );
                """
            )
        print("[TELEMETRY] Connected to PostgreSQL and ensured request_logs table exists.")

    async def log_request(
        self,
        *,
        tenant_id: str,
        cache_status: str,
        routed_tier: str,
        provider: str,
        model: str,
        router_reason: str,
        failover_used: bool,
        pii_redacted: list[str],
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
    ) -> None:
        if not self.pool:
            return

        # For cache hits, estimate baseline tokens (~40 in, ~80 out) if token counts are 0
        # so savings reflect the frontier LLM call we avoided.
        baseline_in = prompt_tokens if prompt_tokens > 0 else 40
        baseline_out = completion_tokens if completion_tokens > 0 else 80

        actual_cost = (
            0.0
            if cache_status in ("EXACT_HIT", "SEMANTIC_HIT")
            else estimate_cost_usd(model, prompt_tokens, completion_tokens)
        )
        baseline_cost = estimate_cost_usd(BASELINE_MODEL, baseline_in, baseline_out)
        saved_cost = max(0.0, baseline_cost - actual_cost)

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO request_logs (
                        tenant_id, cache_status, routed_tier, provider, model,
                        router_reason, failover_used, pii_redacted,
                        prompt_tokens, completion_tokens, latency_ms,
                        actual_cost_usd, baseline_cost_usd, saved_cost_usd
                    ) VALUES (
                        $1, $2, $3, $4, $5,
                        $6, $7, $8,
                        $9, $10, $11,
                        $12, $13, $14
                    )
                    """,
                    tenant_id,
                    cache_status,
                    routed_tier,
                    provider,
                    model,
                    router_reason,
                    failover_used,
                    ",".join(pii_redacted),
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                    actual_cost,
                    baseline_cost,
                    saved_cost,
                )
        except Exception as exc:
            # Telemetry failures must NEVER crash user requests
            print(f"[TELEMETRY ERROR] Failed to log request: {exc}")

    async def get_summary(self) -> dict[str, Any]:
        """Aggregates live metrics from PostgreSQL for the Gateway X-Ray dashboard."""
        if not self.pool:
            return {"error": "Telemetry pool not initialized"}

        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) AS total_requests,
                    COUNT(*) FILTER (WHERE cache_status IN ('EXACT_HIT', 'SEMANTIC_HIT')) AS cache_hits,
                    COUNT(*) FILTER (WHERE failover_used = TRUE) AS failovers,
                    COUNT(*) FILTER (WHERE pii_redacted <> '') AS pii_redactions,
                    COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                    COALESCE(SUM(actual_cost_usd), 0) AS total_actual_cost,
                    COALESCE(SUM(baseline_cost_usd), 0) AS total_baseline_cost,
                    COALESCE(SUM(saved_cost_usd), 0) AS total_saved_cost
                FROM request_logs;
                """
            )
            recent = await conn.fetch(
                """
                SELECT
                    id, created_at, tenant_id, cache_status, routed_tier,
                    provider, model, failover_used, pii_redacted,
                    ROUND(latency_ms::numeric, 1) AS latency_ms,
                    ROUND(saved_cost_usd::numeric, 6) AS saved_cost_usd
                FROM request_logs
                ORDER BY id DESC
                LIMIT 10;
                """
            )

        total = stats["total_requests"]
        baseline = float(stats["total_baseline_cost"])
        saved = float(stats["total_saved_cost"])

        return {
            "total_requests": total,
            "cache_hits": stats["cache_hits"],
            "cache_hit_rate_pct": round((stats["cache_hits"] / total * 100), 1) if total else 0.0,
            "failovers": stats["failovers"],
            "pii_redactions": stats["pii_redactions"],
            "avg_latency_ms": round(float(stats["avg_latency_ms"]), 1),
            "total_actual_cost_usd": round(float(stats["total_actual_cost"]), 6),
            "total_baseline_cost_usd": round(baseline, 6),
            "total_saved_cost_usd": round(saved, 6),
            "cost_savings_pct": round((saved / baseline * 100), 1) if baseline > 0 else 0.0,
            "recent_requests": [dict(r) for r in recent],
        }

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()