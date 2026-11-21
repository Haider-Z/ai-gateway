import asyncio

import pytest
from fastapi import HTTPException

from app.core import security
from app.core.guardrails import redact_text
from app.shop.rag import ShopRAGStore


def test_pii_redaction_scrubs_email_and_phone():
    raw = "Reach me at zaman@example.com or 555-867-5309 for details."
    cleaned, found = redact_text(raw)

    assert "zaman@example.com" not in cleaned
    assert "555-867-5309" not in cleaned
    assert len(found) >= 2


def test_pii_redaction_leaves_clean_prompt_untouched():
    raw = "What is Zaman's CGPA and degree?"
    cleaned, found = redact_text(raw)

    assert cleaned == raw
    assert found == []


def test_admin_key_rejects_missing_or_wrong_key(monkeypatch):
    monkeypatch.setattr(security, "ADMIN_API_KEY", "super-secret-key")

    with pytest.raises(HTTPException) as missing_exc:
        asyncio.run(security.require_admin_key(None))
    assert missing_exc.value.status_code == 401

    with pytest.raises(HTTPException) as wrong_exc:
        asyncio.run(security.require_admin_key("wrong-key"))
    assert wrong_exc.value.status_code == 401


def test_admin_key_accepts_valid_key(monkeypatch):
    monkeypatch.setattr(security, "ADMIN_API_KEY", "super-secret-key")
    # Should complete without raising HTTPException
    asyncio.run(security.require_admin_key("super-secret-key"))


def test_pdf_chunker_splits_long_text_with_overlap():
    text = ("FastAPI and Redis 8 power this cost-aware AI gateway. " * 30).strip()
    chunks = ShopRAGStore.chunk_text(text, chunk_size=300, overlap=60)

    assert len(chunks) > 2
    assert all(len(c) <= 320 for c in chunks)


def test_rate_limiter_allows_loopback_and_throttles_external():
    from types import SimpleNamespace

    limiter = security.SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    loopback_req = SimpleNamespace(headers={}, client=SimpleNamespace(host="127.0.0.1"))
    external_req = SimpleNamespace(
        headers={"x-forwarded-for": "203.0.113.10"},
        client=SimpleNamespace(host="172.18.0.5"),
    )

    for _ in range(5):
        asyncio.run(limiter(loopback_req))

    asyncio.run(limiter(external_req))
    asyncio.run(limiter(external_req))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(limiter(external_req))
    assert exc.value.status_code == 429


def test_cache_extract_cacheable_prompt_rules():
    from app.core.cache import TwoTierCache

    single = [{"role": "user", "content": "  What is Redis?  "}]
    assert TwoTierCache.extract_cacheable_prompt(single, temperature=0.2, cache_control=None) == "What is Redis?"
    assert TwoTierCache.extract_cacheable_prompt(single, temperature=0.9, cache_control=None) is None
    assert TwoTierCache.extract_cacheable_prompt(single, temperature=0.2, cache_control="no-cache") is None

    multi = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Follow-up"},
    ]
    assert TwoTierCache.extract_cacheable_prompt(multi, temperature=0.2, cache_control=None) is None
