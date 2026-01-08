from .llm_clients import (
    DEFAULT_MODELS,
    DEFAULT_PROVIDER_PRIORITIES,
    RECOMMENDED_MODELS,
    BaseLLMClient,
    LiteLLMClient,
    LLMConfig,
    LLMManager,
    OllamaLocalClient,
    OpenRouterClient,
    get_client,
    get_provider_model,
    list_available_providers,
)


def get_provider_priorities_from_litellm() -> dict[str, int]:
    try:
        from lolm import get_provider_priorities_from_litellm as _get
    except ImportError:
        return {}

    try:
        raw = _get() or {}
        return {str(k): int(v) for k, v in raw.items()}
    except Exception:
        return {}

__all__ = [
    'DEFAULT_MODELS',
    'DEFAULT_PROVIDER_PRIORITIES',
    'RECOMMENDED_MODELS',
    'BaseLLMClient',
    'LiteLLMClient',
    'LLMConfig',
    'LLMManager',
    'OpenRouterClient',
    'OllamaLocalClient',
    'get_client',
    'get_provider_model',
    'get_provider_priorities_from_litellm',
    'list_available_providers',
]

