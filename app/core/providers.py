import os
from dataclasses import dataclass

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ProviderTarget:
    provider: str       # "openai" or "gemini"
    model: str          # e.g. "gemini-3.5-flash-lite" or "gpt-6.2-sol"
    base_url: str
    api_key: str
    tier: str           # "fast", "strong", "override", or "fallback"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


class ProviderRegistry:
    """Maps provider names and routing tiers to concrete endpoints and failover chains."""

    PROVIDER_URLS = {
        "openai": "https://api.openai.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    }

    def __init__(self) -> None:
        self.api_keys = {
            "openai": os.environ["OPENAI_API_KEY"],
            "gemini": os.environ["GEMINI_API_KEY"],
        }
        self.fast_target = self._build_target(
            provider=os.getenv("FAST_PROVIDER", "gemini").lower(),
            model=os.getenv("FAST_MODEL", "gemini-3.5-flash-lite"),
            tier="fast",
        )
        self.strong_target = self._build_target(
            provider=os.getenv("STRONG_PROVIDER", "openai").lower(),
            model=os.getenv("STRONG_MODEL", "gpt-6.2-sol"),
            tier="strong",
        )
        self.default_fallback = self._build_target(
            provider=os.getenv("FALLBACK_PROVIDER", "openai").lower(),
            model=os.getenv("FALLBACK_MODEL", "gpt-6-luna"),
            tier="fallback",
        )

    def _build_target(self, provider: str, model: str, tier: str) -> ProviderTarget:
        if provider not in self.PROVIDER_URLS:
            raise ValueError(f"Unsupported provider '{provider}'. Use 'openai' or 'gemini'.")
        return ProviderTarget(
            provider=provider,
            model=model,
            base_url=self.PROVIDER_URLS[provider],
            api_key=self.api_keys[provider],
            tier=tier,
        )

    def resolve_explicit_model(self, model_name: str) -> ProviderTarget:
        provider = "gemini" if model_name.lower().startswith("gemini") else "openai"
        return self._build_target(provider=provider, model=model_name, tier="override")

    def get_fallback_chain(self, primary: ProviderTarget) -> list[ProviderTarget]:
        """Returns [primary, cross-provider backup] so a provider outage never breaks the user."""
        if primary.provider != self.default_fallback.provider:
            return [primary, self.default_fallback]

        # If primary is already on the default fallback provider, fail over to the other provider
        other_provider = "gemini" if primary.provider == "openai" else "openai"
        other_model = "gemini-3.5-flash-lite" if other_provider == "gemini" else "gpt-6-luna"
        backup = self._build_target(provider=other_provider, model=other_model, tier="fallback")
        return [primary, backup]