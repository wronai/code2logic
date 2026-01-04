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
import json

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

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


DEFAULT_MODELS = {
    'openrouter': 'qwen/qwen-2.5-coder-32b-instruct',
    'openai': 'gpt-4-turbo',
    'anthropic': 'claude-3-sonnet-20240229',
    'groq': 'llama-3.1-70b-versatile',
    'together': 'Qwen/Qwen2.5-Coder-32B-Instruct',
    'ollama': 'qwen2.5-coder:14b',
    'litellm': 'ollama/qwen2.5-coder:14b',
}


def _get_user_llm_config_path() -> str:
    return os.path.join(os.path.expanduser('~'), '.code2logic', 'llm_config.json')


def _load_user_llm_config() -> Dict[str, Any]:
    path = _get_user_llm_config_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _get_priority_mode() -> str:
    cfg = _load_user_llm_config()
    return (cfg.get('priority_mode') or 'provider-first').strip()


def get_priority_mode() -> str:
    return _get_priority_mode()


def _get_provider_priority_overrides() -> Dict[str, int]:
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
    env_var_map = {
        'openrouter': 'OPENROUTER_MODEL',
        'openai': 'OPENAI_MODEL',
        'anthropic': 'ANTHROPIC_MODEL',
        'groq': 'GROQ_MODEL',
        'together': 'TOGETHER_MODEL',
        'ollama': 'OLLAMA_MODEL',
        'litellm': 'LITELLM_MODEL',
    }
    env_var = env_var_map.get(provider)
    if env_var:
        v = os.environ.get(env_var)
        if v:
            return v

    if provider == 'litellm':
        return os.environ.get('LITELLM_MODEL', DEFAULT_MODELS['litellm'])

    return DEFAULT_MODELS.get(provider, '')


DEFAULT_PROVIDER_PRIORITIES = {
    'ollama': 10,
    'openrouter': 20,
    'groq': 30,
    'together': 40,
    'openai': 50,
    'anthropic': 60,
    'litellm': 70,
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
            model: Model to use (default from OLLAMA_MODEL or qwen2.5-coder:14b)
            host: Ollama host URL (default from OLLAMA_HOST or localhost:11434)
        """
        self.model = model or os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:14b')
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
        self.model = model or os.environ.get('LITELLM_MODEL', 'ollama/qwen2.5-coder:14b')
    
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

    if provider in ("auto", "AUTO"):
        for p in _get_priority_order():
            client = _try_client(p, model=model)
            if client is not None:
                return client
        raise RuntimeError("No LLM provider available. Configure a provider and try again.")
    
    if provider == 'openrouter':
        return OpenRouterClient(model=model)
    elif provider == 'ollama':
        return OllamaLocalClient(model=model)
    elif provider == 'litellm':
        return LiteLLMClient(model=model)
    else:
        # Auto-detect (backward compatible): try providers in priority order
        for p in _get_priority_order():
            client = _try_client(p, model=model)
            if client is not None:
                return client
        raise RuntimeError("No LLM provider available. Install Ollama or set OPENROUTER_API_KEY.")


def _try_client(provider: str, model: str = None) -> Optional[BaseLLMClient]:
    try:
        if provider == 'openrouter':
            client = OpenRouterClient(model=model)
        elif provider == 'ollama':
            client = OllamaLocalClient(model=model)
        elif provider == 'litellm':
            client = LiteLLMClient(model=model)
        else:
            return None
        if client.is_available():
            return client
    except Exception:
        return None
    return None


def _get_priority_order() -> List[str]:
    return [p for p, _ in _get_effective_provider_order()]


def _get_effective_provider_order() -> List[tuple[str, int]]:
    mode = _get_priority_mode()
    provider_priorities = dict(DEFAULT_PROVIDER_PRIORITIES)

    for provider, pr in _get_provider_priorities_from_litellm_yaml().items():
        provider_priorities[provider] = min(int(provider_priorities.get(provider, 100)), int(pr))

    for provider, pr in _get_provider_priority_overrides().items():
        provider_priorities[provider] = int(pr)

    effective: Dict[str, int] = {}
    for provider, base_pr in provider_priorities.items():
        model_str = _get_provider_model_string(provider)
        model_pr = _get_model_priority(model_str)

        if mode == 'model-first':
            # Prefer model rules; if missing, fall back to provider priority.
            effective[provider] = int(model_pr) if model_pr is not None else int(base_pr)
        elif mode == 'mixed':
            # Take the best (lowest number) from either provider priority or model rule.
            if model_pr is None:
                effective[provider] = int(base_pr)
            else:
                effective[provider] = min(int(base_pr), int(model_pr))
        else:
            # provider-first
            effective[provider] = int(base_pr)

    return sorted(effective.items(), key=lambda kv: int(kv[1]))


def get_effective_provider_priorities() -> Dict[str, int]:
    return {p: int(pr) for p, pr in _get_effective_provider_order()}


def _get_provider_priorities_from_litellm_yaml() -> Dict[str, int]:
    if not YAML_AVAILABLE:
        return {}

    for path in _candidate_litellm_yaml_paths():
        if not path:
            continue
        try:
            if not os.path.exists(path):
                continue
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            model_list = data.get('model_list') or []
            result: Dict[str, int] = {}
            for entry in model_list:
                litellm_model = ((entry.get('litellm_params') or {}).get('model') or '')
                if not litellm_model:
                    continue
                provider = litellm_model.split('/', 1)[0] if '/' in litellm_model else 'openai'
                pr = int(entry.get('priority', 100))
                result[provider] = min(result.get(provider, 100), pr)
            return result
        except Exception:
            continue
    return {}


def _candidate_litellm_yaml_paths() -> List[str]:
    # Similar to config.Config env search strategy
    return [
        os.path.join(os.getcwd(), 'litellm_config.yaml'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'litellm_config.yaml'),
        os.path.join(os.path.expanduser('~'), '.code2logic', 'litellm_config.yaml'),
    ]
