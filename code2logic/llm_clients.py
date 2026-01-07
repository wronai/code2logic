"""
LLM Client implementations for various providers.

DEPRECATED: This module is maintained for backward compatibility.
New code should use the lolm package directly:

    from lolm import get_client, OpenRouterClient, OllamaClient

This module re-exports from lolm with additional backward-compatible
functions that were specific to code2logic.
"""

# Re-export everything from lolm for backward compatibility
from lolm import (
    BaseLLMClient,
    OpenRouterClient,
    OllamaClient as OllamaLocalClient,
    LiteLLMClient,
    LLMManager,
    get_client,
    list_available_providers,
    RECOMMENDED_MODELS,
    DEFAULT_MODELS,
    DEFAULT_PROVIDER_PRIORITIES,
    load_config as _load_lolm_config,
    get_provider_model,
    get_provider_priorities_from_litellm as _get_provider_priorities_from_litellm_yaml,
)

import os
import json
from typing import Optional, List, Dict, Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# Legacy code2logic config path (for backward compatibility)
def _get_user_llm_config_path() -> str:
    """Get path to legacy code2logic LLM config."""
    return os.path.join(os.path.expanduser('~'), '.code2logic', 'llm_config.json')


def _load_user_llm_config() -> Dict[str, Any]:
    """Load legacy code2logic LLM config."""
    path = _get_user_llm_config_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _get_priority_mode() -> str:
    """Get priority mode from legacy config."""
    cfg = _load_user_llm_config()
    return (cfg.get('priority_mode') or 'provider-first').strip()


def get_priority_mode() -> str:
    """Get priority mode (legacy wrapper)."""
    return _get_priority_mode()


def _get_provider_priority_overrides() -> Dict[str, int]:
    """Get provider priority overrides from legacy config."""
    cfg = _load_user_llm_config()
    raw = cfg.get('provider_priorities') or {}
    out: Dict[str, int] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out


def _get_model_priority_rules() -> Dict[str, Dict[str, int]]:
    """Get model priority rules from legacy config."""
    cfg = _load_user_llm_config()
    mp = cfg.get('model_priorities') or {}

    exact_raw = mp.get('exact') or {}
    prefix_raw = mp.get('prefix') or {}

    exact: Dict[str, int] = {}
    prefix: Dict[str, int] = {}
    for k, v in exact_raw.items():
        try:
            exact[str(k)] = int(v)
        except Exception:
            continue
    for k, v in prefix_raw.items():
        try:
            prefix[str(k)] = int(v)
        except Exception:
            continue

    return {'exact': exact, 'prefix': prefix}


def _get_model_priority(model_string: str) -> Optional[int]:
    """Get model priority from legacy config."""
    if not model_string:
        return None

    rules = _get_model_priority_rules()

    if model_string in rules['exact']:
        return int(rules['exact'][model_string])

    best: Optional[int] = None
    for prefix, pr in rules['prefix'].items():
        if model_string.startswith(prefix):
            if best is None:
                best = int(pr)
            else:
                best = min(best, int(pr))
    return best


def _get_provider_model_string(provider: str) -> str:
    """Get model string for provider from environment."""
    return get_provider_model(provider)


def _get_priority_order() -> List[str]:
    """Get providers ordered by priority."""
    return [p for p, _ in _get_effective_provider_order()]


def _get_effective_provider_order() -> List[tuple]:
    """Get effective provider order with priorities."""
    mode = _get_priority_mode()
    provider_priorities = dict(DEFAULT_PROVIDER_PRIORITIES)

    yaml_priorities = _get_provider_priorities_from_litellm_yaml()
    yaml_providers = set(yaml_priorities.keys())
    for provider, pr in yaml_priorities.items():
        provider_priorities[provider] = min(int(provider_priorities.get(provider, 100)), int(pr))

    override_priorities = _get_provider_priority_overrides()
    override_providers = set(override_priorities.keys())
    for provider, pr in override_priorities.items():
        provider_priorities[provider] = int(pr)

    effective: Dict[str, int] = {}
    has_model_rule: Dict[str, bool] = {}
    for provider, base_pr in provider_priorities.items():
        model_str = _get_provider_model_string(provider)
        model_pr = _get_model_priority(model_str)
        has_model_rule[provider] = model_pr is not None

        if mode == 'model-first':
            effective[provider] = int(model_pr) if model_pr is not None else int(base_pr)
        elif mode == 'mixed':
            if model_pr is None:
                effective[provider] = int(base_pr)
            else:
                effective[provider] = min(int(base_pr), int(model_pr))
        else:
            effective[provider] = int(base_pr)

    def _provider_source_rank(p: str) -> int:
        if p in override_providers:
            return 0
        if p in yaml_providers:
            return 1
        return 2

    if mode == 'provider-first':
        return sorted(
            effective.items(),
            key=lambda kv: (int(kv[1]), _provider_source_rank(kv[0]), kv[0]),
        )

    return sorted(
        effective.items(),
        key=lambda kv: (int(kv[1]), 0 if has_model_rule.get(kv[0], False) else 1, _provider_source_rank(kv[0]), kv[0]),
    )


def get_effective_provider_priorities() -> Dict[str, int]:
    """Get effective provider priorities (legacy wrapper)."""
    return {p: int(pr) for p, pr in _get_effective_provider_order()}


def _candidate_litellm_yaml_paths() -> List[str]:
    """Get candidate paths for litellm_config.yaml."""
    return [
        os.path.join(os.getcwd(), 'litellm_config.yaml'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'litellm_config.yaml'),
        os.path.join(os.path.expanduser('~'), '.code2logic', 'litellm_config.yaml'),
    ]


# Export all for backward compatibility
__all__ = [
    # Classes (from lolm)
    'BaseLLMClient',
    'OpenRouterClient',
    'OllamaLocalClient',
    'LiteLLMClient',
    'LLMManager',
    # Functions
    'get_client',
    'list_available_providers',
    'get_priority_mode',
    'get_effective_provider_priorities',
    # Constants
    'RECOMMENDED_MODELS',
    'DEFAULT_MODELS',
    'DEFAULT_PROVIDER_PRIORITIES',
]
