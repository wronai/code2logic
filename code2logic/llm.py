"""
LLM integration for code2logic using Ollama and LiteLLM.

This module provides integration with Large Language Models for
intelligent code analysis, refactoring suggestions, and documentation generation.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

try:
    import litellm
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from .models import Project, Module, Function, Class

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    provider: str = "ollama"  # "ollama" or "litellm"
    model: str = "codellama"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000


class LLMInterface:
    """Interface for Large Language Model integration."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM interface.
        
        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        self._validate_dependencies()
        
        if self.config.provider == "litellm" and LITELLM_AVAILABLE:
            litellm.set_verbose = False
            if self.config.api_key:
                litellm.api_key = self.config.api_key
    
    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are available."""
        if self.config.provider == "ollama" and not OLLAMA_AVAILABLE:
            raise ImportError("ollama package is required for Ollama provider")
        elif self.config.provider == "litellm" and not LITELLM_AVAILABLE:
            raise ImportError("litellm package is required for LiteLLM provider")
    
    def analyze_code(
        self, 
        code: str, 
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze code using LLM.
        
        Args:
            code: Code to analyze
            context: Additional context
            
        Returns:
            Analysis results
        """
        prompt = self._build_analysis_prompt(code, context)
        response = self._call_llm(prompt)
        return self._parse_analysis_response(response)
    
    def suggest_refactoring(
        self, 
        target: Union[Module, Class, Function], 
        project: Project
    ) -> List[str]:
        """
        Suggest refactoring options using LLM.
        
        Args:
            target: Target to refactor
            project: Project context
            
        Returns:
            List of refactoring suggestions
        """
        prompt = self._build_refactoring_prompt(target, project)
        response = self._call_llm(prompt)
        return self._parse_suggestions(response)
    
    def generate_documentation(
        self, 
        target: Union[Module, Class, Function], 
        style: str = "google"
    ) -> str:
        """
        Generate documentation using LLM.
        
        Args:
            target: Target to document
            style: Documentation style
            
        Returns:
            Generated documentation
        """
        prompt = self._build_documentation_prompt(target, style)
        response = self._call_llm(prompt)
        return response.strip()
    
    def explain_code(
        self, 
        code: str, 
        detail_level: str = "medium"
    ) -> str:
        """
        Explain code using LLM.
        
        Args:
            code: Code to explain
            detail_level: Level of detail ("low", "medium", "high")
            
        Returns:
            Code explanation
        """
        prompt = self._build_explanation_prompt(code, detail_level)
        response = self._call_llm(prompt)
        return response.strip()
    
    def generate_tests(
        self, 
        function: Function, 
        test_framework: str = "pytest"
    ) -> str:
        """
        Generate unit tests using LLM.
        
        Args:
            function: Function to test
            test_framework: Test framework to use
            
        Returns:
            Generated test code
        """
        prompt = self._build_test_generation_prompt(function, test_framework)
        response = self._call_llm(prompt)
        return response.strip()
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        try:
            if self.config.provider == "ollama":
                return self._call_ollama(prompt)
            elif self.config.provider == "litellm":
                return self._call_litellm(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error: Failed to get LLM response - {e}"
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        try:
            response = ollama.generate(
                model=self.config.model,
                prompt=prompt,
                options={
                    'temperature': self.config.temperature,
                    'num_predict': self.config.max_tokens
                }
            )
            return response['response']
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise
    
    def _call_litellm(self, prompt: str) -> str:
        """Call LiteLLM."""
        try:
            response = completion(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LiteLLM call failed: {e}")
            raise
    
    def _build_analysis_prompt(self, code: str, context: str) -> str:
        """Build prompt for code analysis."""
        return f"""Analyze the following code and provide insights:

Context: {context}

Code:
```python
{code}
```

Please provide:
1. Code quality assessment
2. Potential issues or bugs
3. Performance considerations
4. Security concerns
5. Best practices violations
6. Suggestions for improvement

Format your response as JSON with the following structure:
{{
    "quality_score": <0-100>,
    "issues": [<list of issues>],
    "suggestions": [<list of suggestions>],
    "complexity": <low/medium/high>,
    "maintainability": <low/medium/high>
}}"""
    
    def _build_refactoring_prompt(
        self, 
        target: Union[Module, Class, Function], 
        project: Project
    ) -> str:
        """Build prompt for refactoring suggestions."""
        target_info = self._get_target_info(target)
        project_context = self._get_project_context(project)
        
        return f"""Provide refactoring suggestions for the following {type(target).__name__.lower()}:

Project Context:
{project_context}

Target Information:
{target_info}

Please suggest specific refactoring actions that would improve:
1. Code readability
2. Maintainability
3. Performance
4. Testability
5. Design patterns

Format your response as a numbered list of actionable suggestions."""
    
    def _build_documentation_prompt(
        self, 
        target: Union[Module, Class, Function], 
        style: str
    ) -> str:
        """Build prompt for documentation generation."""
        target_info = self._get_target_info(target)
        
        return f"""Generate {style} style documentation for the following {type(target).__name__.lower()}:

{target_info}

Please provide comprehensive documentation that includes:
1. Purpose and functionality
2. Parameters and return values (if applicable)
3. Usage examples
4. Important notes or warnings
5. Related components

Format the documentation appropriately for {style} style."""
    
    def _build_explanation_prompt(self, code: str, detail_level: str) -> str:
        """Build prompt for code explanation."""
        detail_instructions = {
            "low": "Provide a brief, high-level explanation",
            "medium": "Provide a detailed explanation with key concepts",
            "high": "Provide an in-depth explanation covering all aspects"
        }
        
        return f"""Explain the following code. {detail_instructions.get(detail_level, detail_instructions["medium"])}:

```python
{code}
```

Focus on:
1. What the code does
2. How it works
3. Key algorithms or patterns used
4. Any important considerations"""
    
    def _build_test_generation_prompt(
        self, 
        function: Function, 
        test_framework: str
    ) -> str:
        """Build prompt for test generation."""
        func_info = f"""
Function: {function.name}
Parameters: {function.parameters}
Code:
```python
{function.code}
```"""
        
        return f"""Generate comprehensive unit tests for the following function using {test_framework}:

{func_info}

Please include:
1. Test cases for normal operation
2. Edge cases and boundary conditions
3. Error handling scenarios
4. Mock usage if external dependencies exist
5. Clear test descriptions

Generate complete, runnable test code."""
    
    def _get_target_info(self, target: Union[Module, Class, Function]) -> str:
        """Get formatted information about the target."""
        if isinstance(target, Module):
            return f"""Module: {target.name}
Path: {target.path}
Lines of Code: {target.lines_of_code}
Imports: {', '.join(target.imports[:5])}
Functions: {len(target.functions)}
Classes: {len(target.classes)}"""
        
        elif isinstance(target, Class):
            methods_info = "\n".join([
                f"  - {method.name}({', '.join(method.parameters)})"
                for method in target.methods[:5]
            ])
            
            return f"""Class: {target.name}
Base Classes: {', '.join(target.base_classes)}
Methods: {len(target.methods)}
Lines of Code: {target.lines_of_code}

Methods:
{methods_info}"""
        
        elif isinstance(target, Function):
            return f"""Function: {target.name}
Parameters: {', '.join(function.parameters)}
Lines of Code: {function.lines_of_code}
Complexity: {function.complexity}
Has Docstring: {function.docstring is not None}

Code:
```python
{function.code}
```"""
        
        return ""
    
    def _get_project_context(self, project: Project) -> str:
        """Get project context information."""
        return f"""Project: {project.name}
Modules: {len(project.modules)}
Total Functions: {sum(len(m.functions) for m in project.modules)}
Total Classes: {sum(len(m.classes) for m in project.modules)}
Dependencies: {len(project.dependencies)}"""
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis response."""
        try:
            # Try to parse as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback to simple structure
            return {
                "quality_score": 75,
                "issues": ["Could not parse detailed analysis"],
                "suggestions": ["Review code manually"],
                "complexity": "medium",
                "maintainability": "medium",
                "raw_response": response
            }
    
    def _parse_suggestions(self, response: str) -> List[str]:
        """Parse suggestions from LLM response."""
        suggestions = []
        
        # Try to extract numbered list
        import re
        numbered_items = re.findall(r'^\d+\.\s*(.+)$', response, re.MULTILINE)
        if numbered_items:
            suggestions = numbered_items
        else:
            # Split by newlines and clean up
            lines = response.split('\n')
            suggestions = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        
        return suggestions[:10]  # Limit to 10 suggestions
    
    def list_available_models(self) -> List[str]:
        """List available models for the current provider."""
        if self.config.provider == "ollama":
            try:
                models = ollama.list()
                return [model['name'] for model in models.get('models', [])]
            except Exception as e:
                logger.error(f"Failed to list Ollama models: {e}")
                return []
        
        elif self.config.provider == "litellm":
            # Common LiteLLM models
            return [
                "gpt-3.5-turbo",
                "gpt-4",
                "claude-3-sonnet",
                "gemini-pro",
                "codellama/CodeLlama-34b-Instruct-hf"
            ]
        
        return []
    
    def health_check(self) -> Dict[str, Any]:
        """Check if LLM service is healthy."""
        status = {
            "provider": self.config.provider,
            "model": self.config.model,
            "available": False,
            "error": None
        }
        
        try:
            if self.config.provider == "ollama":
                models = ollama.list()
                status["available"] = True
                status["models"] = [model['name'] for model in models.get('models', [])]
            
            elif self.config.provider == "litellm":
                # Simple test call
                response = completion(
                    model=self.config.model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=10
                )
                status["available"] = True
        
        except Exception as e:
            status["error"] = str(e)
        
        return status
