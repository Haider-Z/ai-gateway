import os
import secrets
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")


async def require_admin_key(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> None:
    """Blocks mutating admin endpoints unless the caller sends a matching X-Admin-Key."""
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_API_KEY is not configured on the server.",
        )
    if not x_admin_key or not secrets.compare_digest(x_admin_key, ADMIN_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Key.")


class SlidingWindowRateLimiter:
    """Lightweight in-memory per-IP rate limiter for public chat endpoints."""

    def __init__(self, max_requests: int = 20, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def __call__(self, request: Request) -> None:
        forwarded = request.headers.get("x-forwarded-for")
        client_ip = (
            forwarded.split(",")[0].strip()
            if forwarded
            else (request.client.host if request.client else "unknown")
        )
        if client_ip in ("127.0.0.1", "::1"):
            return

        now = time.monotonic()
        bucket = self._hits[client_ip]
        while bucket and now - bucket[0] > self.window_seconds:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({self.max_requests} req/{self.window_seconds}s).",
            )
        bucket.append(now)


chat_rate_limit = SlidingWindowRateLimiter(max_requests=20, window_seconds=60)