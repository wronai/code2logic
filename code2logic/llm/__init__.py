"""LLM client integrations.

Re-exports from lolm package for unified LLM management.
Backward compatible with existing code2logic imports.
"""

# Import from lolm package (shared LLM infrastructure)
from lolm import (
    BaseLLMClient,
    OpenRouterClient,
    OllamaClient as OllamaLocalClient,  # Alias for backward compatibility
    LiteLLMClient,
    get_client,
    LLMManager,
    LLMConfig,
    RECOMMENDED_MODELS,
    DEFAULT_MODELS,
)

from ..intent import EnhancedIntentGenerator

__all__ = [
    # Core clients (from lolm)
    'BaseLLMClient',
    'OpenRouterClient', 
    'OllamaLocalClient',
    'LiteLLMClient',
    'get_client',
    # Manager
    'LLMManager',
    'LLMConfig',
    # Constants
    'RECOMMENDED_MODELS',
    'DEFAULT_MODELS',
    # Intent
    'EnhancedIntentGenerator',
]
