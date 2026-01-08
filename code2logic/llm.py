"""
LLM Integration for Code2Logic

Provides integration with local Ollama and LiteLLM for:
- Code generation from CSV analysis
- Refactoring suggestions
- Duplicate detection with semantic analysis
- Code translation between languages

Usage:
    from code2logic.llm import CodeAnalyzer
    
    analyzer = CodeAnalyzer(model="qwen2.5-coder:7b")
    suggestions = analyzer.suggest_refactoring(project_info)
"""

import json
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Optional imports
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


from .llm_clients import (
    OpenRouterClient,
    OllamaLocalClient,
    LiteLLMClient,
    get_client,
)


@dataclass
class LLMConfig:
    """Configuration for LLM backend."""
    provider: str = "ollama"  # "ollama" or "litellm"
    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 2000


class OllamaClient:
    """Direct Ollama API client."""
    
    def __init__(self, config: LLMConfig):
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required: pip install httpx")
        self.config = config
        self.client = httpx.Client(timeout=config.timeout)
    
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate completion from Ollama."""
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }
        
        if system:
            payload["system"] = system
        
        response = self.client.post(
            f"{self.config.base_url}/api/generate",
            json=payload
        )
        response.raise_for_status()
        return response.json().get("response", "")
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat completion from Ollama."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }
        
        response = self.client.post(
            f"{self.config.base_url}/api/chat",
            json=payload
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = self.client.get(f"{self.config.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = self.client.get(f"{self.config.base_url}/api/tags")
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


class LiteLLMClient:
    """LiteLLM client for unified API access."""
    
    def __init__(self, config: LLMConfig):
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm required: pip install litellm")
        self.config = config
    
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate completion via LiteLLM."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages)
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat completion via LiteLLM."""
        model = f"ollama/{self.config.model}"
        if self.config.provider == "litellm":
            model = self.config.model
        
        response = completion(
            model=model,
            messages=messages,
            api_base=self.config.base_url,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content
    
    def is_available(self) -> bool:
        """Check if LiteLLM backend is available."""
        try:
            self.chat([{"role": "user", "content": "test"}])
            return True
        except Exception:
            return False


class CodeAnalyzer:
    """
    LLM-powered code analysis for Code2Logic.
    
    Example:
        >>> from code2logic import analyze_project
        >>> from code2logic.llm import CodeAnalyzer
        >>> 
        >>> project = analyze_project("/path/to/project")
        >>> analyzer = CodeAnalyzer()
        >>> 
        >>> # Get refactoring suggestions
        >>> suggestions = analyzer.suggest_refactoring(project)
        >>> 
        >>> # Generate code in another language
        >>> code = analyzer.generate_code(project, target_lang="typescript")
    """
    
    SYSTEM_PROMPT = """You are an expert software architect and code analyst.
You analyze code structure and provide actionable suggestions for:
- Refactoring and code improvement
- Duplicate detection and consolidation
- Code generation and translation
- Architecture optimization

Be specific, practical, and provide code examples when helpful."""
    
    def __init__(
        self,
        model: str = None,
        provider: str = None,
        base_url: str = None,
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize CodeAnalyzer.
        
        Args:
            model: Model name (e.g., "qwen2.5-coder:7b")
            provider: "ollama" or "litellm"
            base_url: API base URL
        """
        selected_provider = provider or os.environ.get('CODE2LOGIC_DEFAULT_PROVIDER', 'ollama')

        # Keep legacy defaults for local usage
        if model is None:
            model = os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b')
        if base_url is None:
            base_url = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

        # Prefer unified clients + allow optional overrides
        if selected_provider in ("auto", "AUTO"):
            # In auto mode, model/base_url/api_key may not apply uniformly; let get_client decide.
            self.client = get_client('auto')
        elif selected_provider == "openrouter":
            self.client = OpenRouterClient(api_key=api_key, model=model)
        elif selected_provider == "ollama":
            self.client = OllamaLocalClient(model=model, host=base_url)
        elif selected_provider == "litellm":
            self.client = LiteLLMClient(model=model)
        else:
            # Fallback to environment-based selection (may raise if unsupported)
            self.client = get_client(selected_provider, model=model)

        self.config = LLMConfig(
            provider=selected_provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            **kwargs
        )
    
    def is_available(self) -> bool:
        """Check if LLM backend is available."""
        return bool(getattr(self.client, "is_available", lambda: False)())
    
    def suggest_refactoring(self, project) -> List[Dict[str, Any]]:
        """
        Analyze project and suggest refactoring improvements.
        
        Args:
            project: ProjectInfo from code2logic analysis
            
        Returns:
            List of refactoring suggestions with details
        """
        from .generators import CSVGenerator
        
        # Generate compact representation
        csv_gen = CSVGenerator()
        csv_data = csv_gen.generate(project, detail='full')
        
        # Truncate if too long
        if len(csv_data) > 8000:
            lines = csv_data.split('\n')
            csv_data = '\n'.join(lines[:100]) + f"\n... ({len(lines)-100} more lines)"
        
        prompt = f"""Analyze this codebase and suggest refactoring improvements:

```csv
{csv_data}
```

For each suggestion, provide:
1. Issue type (complexity, duplication, naming, structure)
2. Specific location (path, function name)
3. Problem description
4. Recommended fix with code example if applicable
5. Priority (high/medium/low)

Format as JSON array."""
        
        response = self.client.generate(prompt, system=self.SYSTEM_PROMPT)
        
        # Try to parse JSON from response
        try:
            # Find JSON in response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        
        # Return raw response if JSON parsing fails
        return [{"raw_response": response}]
    
    def find_semantic_duplicates(self, project) -> List[Dict[str, Any]]:
        """
        Find semantically similar functions using LLM.
        
        Args:
            project: ProjectInfo from code2logic analysis
            
        Returns:
            List of duplicate groups with similarity analysis
        """
        # Collect all functions with intents
        functions = []
        for m in project.modules:
            for f in m.functions:
                functions.append({
                    'path': m.path,
                    'name': f.name,
                    'signature': self._build_signature(f),
                    'intent': f.intent or '',
                })
            for c in m.classes:
                for method in c.methods:
                    functions.append({
                        'path': m.path,
                        'name': f"{c.name}.{method.name}",
                        'signature': self._build_signature(method),
                        'intent': method.intent or '',
                    })
        
        if len(functions) > 50:
            functions = functions[:50]
        
        prompt = f"""Analyze these functions and find semantic duplicates:

{json.dumps(functions, indent=2)}

Group functions that:
1. Do the same thing (even with different names)
2. Have similar logic patterns
3. Could be consolidated into shared utilities

For each group, explain:
- Why they are duplicates
- How to consolidate them
- Suggested shared function name

Format as JSON array of groups."""
        
        response = self.client.generate(prompt, system=self.SYSTEM_PROMPT)
        
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        
        return [{"raw_response": response}]
    
    def generate_code(
        self,
        project,
        target_lang: str,
        module_filter: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate code in target language from project analysis.
        
        Args:
            project: ProjectInfo from code2logic analysis
            target_lang: Target language (typescript, python, go, rust, etc.)
            module_filter: Optional filter for specific module paths
            
        Returns:
            Dict mapping original path to generated code
        """
        results = {}
        
        modules = project.modules
        if module_filter:
            modules = [m for m in modules if module_filter in m.path]
        
        for module in modules[:5]:  # Limit to 5 modules
            # Build specification
            spec_lines = [f"Module: {module.path}"]
            spec_lines.append(f"Language: {module.language}")
            spec_lines.append(f"Lines: {module.lines_code}")
            
            if module.imports:
                spec_lines.append(f"Imports: {', '.join(module.imports[:10])}")
            
            if module.classes:
                spec_lines.append("\nClasses:")
                for c in module.classes[:5]:
                    spec_lines.append(f"  class {c.name}({', '.join(c.bases)})")
                    for m in c.methods[:10]:
                        spec_lines.append(f"    - {m.name}{self._build_signature(m)}: {m.intent}")
            
            if module.functions:
                spec_lines.append("\nFunctions:")
                for f in module.functions[:10]:
                    spec_lines.append(f"  - {f.name}{self._build_signature(f)}: {f.intent}")
            
            spec = '\n'.join(spec_lines)
            
            prompt = f"""Generate {target_lang} code from this specification:

{spec}

Requirements:
1. Idiomatic {target_lang} code
2. Full type annotations
3. Docstrings/comments
4. Error handling
5. Maintain the same public API

Output only the code."""
            
            response = self.client.generate(prompt, system=self.SYSTEM_PROMPT)
            results[module.path] = response
        
        return results
    
    def translate_function(
        self,
        name: str,
        signature: str,
        intent: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Translate a single function to another language.
        
        Args:
            name: Function name
            signature: Function signature
            intent: What the function does
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Generated code in target language
        """
        prompt = f"""Translate this {source_lang} function to {target_lang}:

Function: {name}
Signature: {signature}
Purpose: {intent}

Generate idiomatic {target_lang} code with:
1. Proper type annotations
2. Error handling
3. Documentation

Output only the code."""
        
        return self.client.generate(prompt, system=self.SYSTEM_PROMPT)
    
    def _build_signature(self, f) -> str:
        """Build compact signature."""
        params = ','.join(f.params[:4])
        if len(f.params) > 4:
            params += '...'
        ret = f"->{f.return_type}" if f.return_type else ""
        return f"({params}){ret}"


def get_available_backends() -> Dict[str, bool]:
    """Get availability status of LLM backends."""
    status = {
        'httpx': HTTPX_AVAILABLE,
        'litellm': LITELLM_AVAILABLE,
        'ollama': False,
    }
    
    if HTTPX_AVAILABLE:
        try:
            client = OllamaClient(LLMConfig())
            status['ollama'] = client.is_available()
        except Exception:
            pass
    
    return status
