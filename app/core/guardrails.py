import re
from typing import Any

# Compiled once at import time for <1ms scanning
PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        "API_KEY",
        re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{20,}|AQ\.[A-Za-z0-9_-]{20,})\b"),
        "[REDACTED_API_KEY]",
    ),
    (
        "CARD",
        re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "[REDACTED_CARD]",
    ),
    (
        "PHONE",
        re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"),
        "[REDACTED_PHONE]",
    ),
]


def redact_text(text: str) -> tuple[str, list[str]]:
    """Scrubs sensitive PII from a string and returns (clean_text, detected_types)."""
    detected: list[str] = []
    cleaned = text
    for label, pattern, replacement in PII_PATTERNS:
        if pattern.search(cleaned):
            cleaned = pattern.sub(replacement, cleaned)
            detected.append(label)
    return cleaned, detected


def redact_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Returns a sanitized copy of chat messages and a deduplicated list of PII types found."""
    sanitized: list[dict[str, Any]] = []
    all_detected: set[str] = set()

    for msg in messages:
        new_msg = dict(msg)
        content = new_msg.get("content")
        if isinstance(content, str):
            clean_content, found = redact_text(content)
            new_msg["content"] = clean_content
            all_detected.update(found)
        sanitized.append(new_msg)

    return sanitized, sorted(all_detected)