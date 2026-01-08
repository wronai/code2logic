"""
LOLM - Lightweight Orchestrated LLM Manager

A reusable LLM configuration and management package for Python projects.
Supports multiple providers with automatic fallback, priority routing,
and unified configuration via .env and litellm_config.yaml.

Supported Providers:
- OpenRouter (cloud, multiple models)
- Ollama (local inference)
- OpenAI, Anthropic, Groq, Together (cloud)
- LiteLLM (universal interface)

Usage:
    from lolm import get_client, LLMManager
    
    # Simple usage
    client = get_client()
    response = client.generate("Explain this code")
    
    # With manager
    manager = LLMManager()
    await manager.initialize()
    response = await manager.generate(GenerateOptions(
        system="You are a code generator",
        user="Create a function to add two numbers"
    ))

CLI:
    lolm status           # Show provider status
    lolm set-provider     # Set default provider
    lolm key set          # Manage API keys
    lolm test             # Test LLM generation
"""

from .config import (
    LLMConfig,
    load_config,
    save_config,
    get_config_path,
    get_provider_model,
    get_provider_priorities_from_litellm,
    DEFAULT_MODELS,
    DEFAULT_PROVIDER_PRIORITIES,
    RECOMMENDED_MODELS,
)
from .provider import (
    BaseLLMClient,
    LLMProvider,
    LLMProviderStatus,
    LLMResponse,
    LLMModelInfo,
    GenerateOptions,
)
from .clients import (
    OpenRouterClient,
    OllamaClient,
    LiteLLMClient,
)
from .manager import (
    LLMManager,
    ProviderInfo,
    get_client,
    list_available_providers,
)

__version__ = '0.1.2'
__all__ = [
    # Config
    'LLMConfig',
    'load_config',
    'save_config',
    'get_config_path',
    'get_provider_model',
    'get_provider_priorities_from_litellm',
    'DEFAULT_MODELS',
    'DEFAULT_PROVIDER_PRIORITIES',
    'RECOMMENDED_MODELS',
    # Provider base
    'BaseLLMClient',
    'LLMProvider',
    'LLMProviderStatus',
    'LLMResponse',
    'LLMModelInfo',
    'GenerateOptions',
    # Clients
    'OpenRouterClient',
    'OllamaClient',
    'LiteLLMClient',
    # Manager
    'LLMManager',
    'ProviderInfo',
    'get_client',
    'list_available_providers',
]
