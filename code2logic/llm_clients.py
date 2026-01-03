"""
LLM Client implementations for various providers.

Supports:
- OpenRouter (cloud, multiple models)
- Ollama (local)
- LiteLLM (universal interface)

Usage:
    from code2logic.llm_clients import OpenRouterClient, OllamaClient, LiteLLMClient
    
    client = OpenRouterClient()
    response = client.generate("Explain this code")
"""

import os

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


# Recommended models by provider
RECOMMENDED_MODELS = {
    'openrouter': [
        ("qwen/qwen-2.5-coder-32b-instruct", "Best for code, 32B"),
        ("deepseek/deepseek-coder-33b-instruct", "DeepSeek Coder 33B"),
        ("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B (free)"),
        ("nvidia/nemotron-3-nano-30b-a3b:free", "Nemotron 30B (free)"),
    ],
    'ollama': [
        ("qwen2.5-coder:14b", "Best local code model"),
        ("qwen2.5-coder:7b", "Fast local code model"),
        ("deepseek-coder:6.7b", "DeepSeek Coder"),
        ("codellama:7b-instruct", "CodeLlama 7B"),
    ],
}


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, system: str = None, max_tokens: int = 4000) -> str:
        """Generate completion."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if client is available."""
        pass
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 4000) -> str:
        """Chat completion (default implementation)."""
        # Convert messages to single prompt
        prompt_parts = []
        system = None
        for msg in messages:
            if msg['role'] == 'system':
                system = msg['content']
            else:
                prompt_parts.append(f"{msg['role']}: {msg['content']}")
        return self.generate('\n'.join(prompt_parts), system=system, max_tokens=max_tokens)


class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client for cloud LLM access."""
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Model to use (default from OPENROUTER_MODEL or qwen-2.5-coder-32b)
        """
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        self.model = model or os.environ.get('OPENROUTER_MODEL', 'qwen/qwen-2.5-coder-32b-instruct')
    
    def generate(self, prompt: str, system: str = None, max_tokens: int = 4000) -> str:
        """Generate completion using OpenRouter."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required: pip install httpx")
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured. Set OPENROUTER_API_KEY.")
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/code2logic",
            "X-Title": "Code2Logic",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        
        try:
            response = httpx.post(self.API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if hasattr(e, 'response') else str(e)
            raise RuntimeError(f"OpenRouter API error: {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Request failed: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenRouter is configured."""
        return bool(self.api_key)
    
    @staticmethod
    def list_recommended_models() -> List[tuple]:
        """List recommended models for code tasks."""
        return RECOMMENDED_MODELS['openrouter']


class OllamaLocalClient(BaseLLMClient):
    """Ollama client for local LLM inference."""
    
    def __init__(self, model: str = None, host: str = None):
        """Initialize Ollama client.
        
        Args:
            model: Model to use (default from OLLAMA_MODEL or qwen2.5-coder:7b)
            host: Ollama host URL (default from OLLAMA_HOST or localhost:11434)
        """
        self.model = model or os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b')
        self.host = host or os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    
    def generate(self, prompt: str, system: str = None, max_tokens: int = 4000) -> str:
        """Generate completion using Ollama."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required: pip install httpx")
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tokens}
        }
        if system:
            payload["system"] = system
        
        try:
            response = httpx.post(f"{self.host}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        if not HTTPX_AVAILABLE:
            return False
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=3)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """List available Ollama models."""
        if not HTTPX_AVAILABLE:
            return []
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m['name'] for m in data.get('models', [])]
        except Exception:
            pass
        return []
    
    @staticmethod
    def list_recommended_models() -> List[tuple]:
        """List recommended models for code tasks."""
        return RECOMMENDED_MODELS['ollama']


class LiteLLMClient(BaseLLMClient):
    """LiteLLM client for universal LLM access."""
    
    def __init__(self, model: str = None):
        """Initialize LiteLLM client.
        
        Args:
            model: Model identifier (e.g., 'ollama/qwen2.5-coder:7b', 'gpt-4')
        """
        self.model = model or os.environ.get('LITELLM_MODEL', 'ollama/qwen2.5-coder:7b')
    
    def generate(self, prompt: str, system: str = None, max_tokens: int = 4000) -> str:
        """Generate completion using LiteLLM."""
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm required: pip install litellm")
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LiteLLM error: {e}")
    
    def is_available(self) -> bool:
        """Check if LiteLLM is available."""
        return LITELLM_AVAILABLE


def get_client(provider: str = None, model: str = None) -> BaseLLMClient:
    """Get appropriate LLM client based on provider.
    
    Args:
        provider: 'openrouter', 'ollama', 'litellm', or None (auto-detect)
        model: Model to use
    
    Returns:
        Configured LLM client
    """
    provider = provider or os.environ.get('CODE2LOGIC_DEFAULT_PROVIDER', 'openrouter')
    
    if provider == 'openrouter':
        return OpenRouterClient(model=model)
    elif provider == 'ollama':
        return OllamaLocalClient(model=model)
    elif provider == 'litellm':
        return LiteLLMClient(model=model)
    else:
        # Auto-detect: try Ollama first, then OpenRouter
        ollama = OllamaLocalClient(model=model)
        if ollama.is_available():
            return ollama
        openrouter = OpenRouterClient(model=model)
        if openrouter.is_available():
            return openrouter
        raise RuntimeError("No LLM provider available. Install Ollama or set OPENROUTER_API_KEY.")
