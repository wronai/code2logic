"""
LLM Manager with Multi-Provider Support.

Manages multiple LLM providers with automatic fallback and priority routing.
"""

import os
from typing import Dict, List, Optional

from .config import (
    DEFAULT_MODELS,
    DEFAULT_PROVIDER_PRIORITIES,
    LLMConfig,
    get_provider_model,
    get_provider_priorities_from_litellm,
    load_config,
    load_env_file,
)
from .provider import BaseLLMClient, LLMModelInfo, LLMProviderStatus
from .clients import (
    OpenRouterClient,
    OllamaClient,
    LiteLLMClient,
    GroqClient,
    TogetherClient,
)


class ProviderInfo:
    """Information about a configured provider."""
    
    def __init__(
        self, 
        name: str, 
        status: LLMProviderStatus, 
        client: Optional[BaseLLMClient] = None,
        model: str = "",
        priority: int = 100,
    ):
        self.name = name
        self.status = status
        self.client = client
        self.model = model
        self.priority = priority
        self.error: Optional[str] = None


class LLMManager:
    """
    LLM Manager with multi-provider support.
    
    Manages multiple LLM providers and provides fallback logic.
    
    Example:
        manager = LLMManager()
        manager.initialize()
        
        if manager.is_available:
            response = manager.generate("Explain this code")
            print(response)
    """
    
    def __init__(self, verbose: bool = False):
        self._providers: Dict[str, ProviderInfo] = {}
        self._primary_provider: Optional[str] = None
        self._initialized = False
        self._verbose = verbose
        self._config = load_config()
    
    @property
    def is_available(self) -> bool:
        """Check if any provider is available."""
        return any(
            info.status == LLMProviderStatus.AVAILABLE 
            for info in self._providers.values()
        )
    
    def is_ready(self) -> bool:
        """Check if manager is initialized and has available provider."""
        return self._initialized and self.is_available
    
    @property
    def primary_provider(self) -> Optional[BaseLLMClient]:
        """Get the primary (first available) provider."""
        if self._primary_provider and self._primary_provider in self._providers:
            info = self._providers[self._primary_provider]
            if info.client:
                return info.client
        return None
    
    @property
    def providers(self) -> Dict[str, ProviderInfo]:
        """Get all provider info."""
        return self._providers
    
    def initialize(self) -> None:
        """Initialize all configured providers."""
        if self._initialized:
            return
        
        load_env_file()
        
        # Initialize all providers
        self._init_openrouter()
        self._init_ollama()
        self._init_groq()
        self._init_together()
        self._init_litellm()
        
        # Determine primary provider based on config and availability
        preferred = self._config.default_provider or os.environ.get('LLM_PROVIDER', 'auto')
        
        if preferred not in ('', 'auto'):
            info = self._providers.get(preferred)
            if info and info.status == LLMProviderStatus.AVAILABLE:
                self._primary_provider = preferred
        
        if not self._primary_provider:
            # Use priority order
            for name in self._get_priority_order():
                info = self._providers.get(name)
                if info and info.status == LLMProviderStatus.AVAILABLE:
                    self._primary_provider = name
                    break
        
        self._initialized = True
    
    def _init_openrouter(self) -> None:
        """Initialize OpenRouter provider."""
        api_key = os.environ.get('OPENROUTER_API_KEY')
        model = get_provider_model('openrouter')
        priority = self._get_effective_priority('openrouter')
        
        info = ProviderInfo('openrouter', LLMProviderStatus.NOT_CONFIGURED, model=model, priority=priority)
        
        if not api_key:
            info.error = "Set OPENROUTER_API_KEY"
        else:
            try:
                client = OpenRouterClient(api_key=api_key, model=model)
                if client.is_available():
                    info.status = LLMProviderStatus.AVAILABLE
                    info.client = client
                else:
                    info.status = LLMProviderStatus.UNAVAILABLE
                    info.error = "API key invalid or expired"
            except Exception as e:
                info.status = LLMProviderStatus.ERROR
                info.error = str(e)
        
        self._providers['openrouter'] = info
    
    def _init_ollama(self) -> None:
        """Initialize Ollama provider."""
        model = get_provider_model('ollama')
        priority = self._get_effective_priority('ollama')
        
        info = ProviderInfo('ollama', LLMProviderStatus.UNAVAILABLE, model=model, priority=priority)
        
        try:
            client = OllamaClient(model=model)
            if client.is_available():
                info.status = LLMProviderStatus.AVAILABLE
                info.client = client
            else:
                info.error = "Ollama not running"
        except Exception as e:
            info.status = LLMProviderStatus.ERROR
            info.error = str(e)
        
        self._providers['ollama'] = info
    
    def _init_groq(self) -> None:
        """Initialize Groq provider."""
        api_key = os.environ.get('GROQ_API_KEY')
        model = get_provider_model('groq')
        priority = self._get_effective_priority('groq')
        
        info = ProviderInfo('groq', LLMProviderStatus.NOT_CONFIGURED, model=model, priority=priority)
        
        if not api_key:
            info.error = "Set GROQ_API_KEY"
        else:
            try:
                client = GroqClient(api_key=api_key, model=model)
                if client.is_available():
                    info.status = LLMProviderStatus.AVAILABLE
                    info.client = client
            except Exception as e:
                info.status = LLMProviderStatus.ERROR
                info.error = str(e)
        
        self._providers['groq'] = info
    
    def _init_together(self) -> None:
        """Initialize Together provider."""
        api_key = os.environ.get('TOGETHER_API_KEY')
        model = get_provider_model('together')
        priority = self._get_effective_priority('together')
        
        info = ProviderInfo('together', LLMProviderStatus.NOT_CONFIGURED, model=model, priority=priority)
        
        if not api_key:
            info.error = "Set TOGETHER_API_KEY"
        else:
            try:
                client = TogetherClient(api_key=api_key, model=model)
                if client.is_available():
                    info.status = LLMProviderStatus.AVAILABLE
                    info.client = client
            except Exception as e:
                info.status = LLMProviderStatus.ERROR
                info.error = str(e)
        
        self._providers['together'] = info
    
    def _init_litellm(self) -> None:
        """Initialize LiteLLM provider."""
        model = get_provider_model('litellm')
        priority = self._get_effective_priority('litellm')
        
        info = ProviderInfo('litellm', LLMProviderStatus.UNAVAILABLE, model=model, priority=priority)
        
        try:
            client = LiteLLMClient(model=model)
            if client.is_available():
                info.status = LLMProviderStatus.AVAILABLE
                info.client = client
            else:
                info.error = "LiteLLM not installed"
        except Exception as e:
            info.status = LLMProviderStatus.ERROR
            info.error = str(e)
        
        self._providers['litellm'] = info
    
    def _get_effective_priority(self, provider: str) -> int:
        """Get effective priority for a provider."""
        # Check user config first
        if provider in self._config.provider_priorities:
            return self._config.provider_priorities[provider]
        
        # Check litellm config
        litellm_priorities = get_provider_priorities_from_litellm()
        if provider in litellm_priorities:
            return litellm_priorities[provider]
        
        # Fall back to defaults
        return DEFAULT_PROVIDER_PRIORITIES.get(provider, 100)
    
    def _get_priority_order(self) -> List[str]:
        """Get providers ordered by priority."""
        providers_with_priority = [
            (name, info.priority) 
            for name, info in self._providers.items()
        ]
        return [name for name, _ in sorted(providers_with_priority, key=lambda x: x[1])]
    
    def get_client(self, provider: str = None) -> Optional[BaseLLMClient]:
        """Get a specific or primary client."""
        if not self._initialized:
            self.initialize()
        
        if provider:
            info = self._providers.get(provider)
            if info and info.client:
                return info.client
            return None
        
        return self.primary_provider
    
    def generate(
        self, 
        prompt: str, 
        system: str = None,
        max_tokens: int = 4000,
        provider: str = None
    ) -> str:
        """
        Generate completion using available provider.
        
        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Maximum tokens
            provider: Specific provider to use (optional)
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If no provider is available
        """
        if not self._initialized:
            self.initialize()
        
        client = self.get_client(provider)
        if client:
            return client.generate(prompt, system=system, max_tokens=max_tokens)
        
        raise RuntimeError("No LLM provider available. Run: lolm status")
    
    def generate_with_fallback(
        self, 
        prompt: str,
        system: str = None,
        max_tokens: int = 4000,
        providers: Optional[List[str]] = None
    ) -> str:
        """
        Generate with fallback to other providers on failure.
        
        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Maximum tokens
            providers: List of providers to try (in order)
            
        Returns:
            Generated text from first successful provider
        """
        if not self._initialized:
            self.initialize()
        
        provider_list = providers or self._get_priority_order()
        last_error: Optional[Exception] = None
        
        for name in provider_list:
            info = self._providers.get(name)
            if not info or not info.client:
                continue
            
            try:
                return info.client.generate(prompt, system=system, max_tokens=max_tokens)
            except Exception as e:
                last_error = e
                continue
        
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    def get_status(self) -> Dict[str, Dict]:
        """Get status of all providers."""
        if not self._initialized:
            self.initialize()
        
        return {
            name: {
                "status": info.status.value,
                "model": info.model,
                "priority": info.priority,
                "error": info.error,
            }
            for name, info in self._providers.items()
        }


def get_client(provider: str = None, model: str = None) -> BaseLLMClient:
    """
    Get appropriate LLM client based on provider.
    
    Args:
        provider: 'openrouter', 'ollama', 'litellm', 'groq', 'together', or 'auto'
        model: Model to use
    
    Returns:
        Configured LLM client
    """
    load_env_file()
    
    config = load_config()
    provider = provider or config.default_provider or os.environ.get('LLM_PROVIDER', 'auto')
    
    if provider in ('auto', 'AUTO', '', None):
        manager = LLMManager()
        manager.initialize()
        client = manager.primary_provider
        if client:
            return client
        raise RuntimeError("No LLM provider available. Configure a provider and try again.")
    
    # Direct provider selection
    if provider == 'openrouter':
        return OpenRouterClient(model=model)
    elif provider == 'ollama':
        return OllamaClient(model=model)
    elif provider == 'litellm':
        return LiteLLMClient(model=model)
    elif provider == 'groq':
        return GroqClient(model=model)
    elif provider == 'together':
        return TogetherClient(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def list_available_providers() -> List[str]:
    """List all available (configured and reachable) providers."""
    manager = LLMManager()
    manager.initialize()
    return [
        name for name, info in manager.providers.items()
        if info.status == LLMProviderStatus.AVAILABLE
    ]
