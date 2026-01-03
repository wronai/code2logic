#!/usr/bin/env python3
"""
Example: LiteLLM + MCP Refactoring Workflow.

Demonstrates using LiteLLM with code2logic for refactoring suggestions
through various LLM providers (Ollama, OpenAI, Anthropic, etc.).

Requirements:
    pip install litellm httpx
    
For Ollama:
    ollama serve
    ollama pull qwen2.5-coder:7b

Usage:
    python mcp_litellm_refactor.py /path/to/project
    python mcp_litellm_refactor.py /path/to/project --model gpt-4
    python mcp_litellm_refactor.py /path/to/project --model ollama/qwen2.5-coder:7b
    python mcp_litellm_refactor.py /path/to/project --provider anthropic
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    print("Warning: litellm not installed. Install with: pip install litellm")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from code2logic import analyze_project, CSVGenerator, GherkinGenerator


# Default models by provider
PROVIDER_MODELS = {
    'ollama': 'ollama/qwen2.5-coder:7b',
    'openai': 'gpt-4',
    'anthropic': 'claude-3-sonnet-20240229',
    'groq': 'groq/llama3-70b-8192',
    'together': 'together_ai/togethercomputer/CodeLlama-34b-Instruct',
}


def check_ollama_available() -> bool:
    """Check if Ollama is running."""
    if not HTTPX_AVAILABLE:
        return False
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def get_ollama_models() -> List[str]:
    """Get list of available Ollama models."""
    if not HTTPX_AVAILABLE:
        return []
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m['name'] for m in data.get('models', [])]
    except Exception:
        pass
    return []


def generate_with_litellm(
    prompt: str,
    model: str = "ollama/qwen2.5-coder:7b",
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> str:
    """Generate text using LiteLLM (supports multiple providers)."""
    if not LITELLM_AVAILABLE:
        return "Error: litellm not installed"
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def analyze_for_refactoring(project) -> Dict[str, Any]:
    """Analyze project and identify refactoring opportunities."""
    issues = {
        'long_functions': [],
        'long_files': [],
        'large_classes': [],
        'duplicates': [],
        'complex_functions': [],
    }
    
    from collections import defaultdict
    import hashlib
    
    sig_groups = defaultdict(list)
    
    for module in project.modules:
        # Long files
        if module.lines_code > 300:
            issues['long_files'].append({
                'path': module.path,
                'lines': module.lines_code,
                'functions': len(module.functions),
                'classes': len(module.classes),
            })
        
        # Functions
        for func in module.functions:
            if func.lines > 40:
                issues['long_functions'].append({
                    'path': module.path,
                    'name': func.name,
                    'lines': func.lines,
                    'complexity': func.complexity,
                })
            
            if func.complexity > 10:
                issues['complex_functions'].append({
                    'path': module.path,
                    'name': func.name,
                    'complexity': func.complexity,
                })
            
            # Track signatures for duplicates
            sig = f"({','.join(func.params)})->{func.return_type or ''}"
            h = hashlib.md5(f"{func.name}:{sig}".encode()).hexdigest()[:8]
            sig_groups[h].append(f"{module.path}::{func.name}")
        
        # Classes
        for cls in module.classes:
            if len(cls.methods) > 15:
                issues['large_classes'].append({
                    'path': module.path,
                    'name': cls.name,
                    'methods': len(cls.methods),
                })
    
    # Duplicates
    for h, funcs in sig_groups.items():
        if len(funcs) > 1:
            issues['duplicates'].append({
                'hash': h,
                'count': len(funcs),
                'functions': funcs[:5],
            })
    
    return issues


def format_issues_for_llm(issues: Dict[str, Any], project) -> str:
    """Format issues as context for LLM."""
    lines = [
        f"# Project Analysis: {project.name}",
        f"Total files: {project.total_files}",
        f"Total lines: {project.total_lines}",
        f"Languages: {', '.join(project.languages.keys())}",
        "",
        "## Issues Found",
        "",
    ]
    
    if issues['long_files']:
        lines.append(f"### Long Files ({len(issues['long_files'])})")
        for f in issues['long_files'][:5]:
            lines.append(f"- {f['path']}: {f['lines']} lines")
        lines.append("")
    
    if issues['long_functions']:
        lines.append(f"### Long Functions ({len(issues['long_functions'])})")
        for f in issues['long_functions'][:5]:
            lines.append(f"- {f['path']}::{f['name']}: {f['lines']} lines")
        lines.append("")
    
    if issues['complex_functions']:
        lines.append(f"### Complex Functions ({len(issues['complex_functions'])})")
        for f in issues['complex_functions'][:5]:
            lines.append(f"- {f['path']}::{f['name']}: complexity {f['complexity']}")
        lines.append("")
    
    if issues['duplicates']:
        lines.append(f"### Potential Duplicates ({len(issues['duplicates'])})")
        for d in issues['duplicates'][:5]:
            lines.append(f"- {', '.join(d['functions'][:3])}")
        lines.append("")
    
    if issues['large_classes']:
        lines.append(f"### Large Classes ({len(issues['large_classes'])})")
        for c in issues['large_classes'][:5]:
            lines.append(f"- {c['path']}::{c['name']}: {c['methods']} methods")
        lines.append("")
    
    return '\n'.join(lines)


def get_refactoring_suggestions(
    issues: Dict[str, Any],
    project,
    model: str,
) -> str:
    """Get LLM-powered refactoring suggestions."""
    context = format_issues_for_llm(issues, project)
    
    system = """You are an expert software architect specializing in code refactoring.
Analyze the provided code issues and give specific, actionable refactoring suggestions.
For each suggestion:
1. Explain the problem briefly
2. Recommend a specific refactoring pattern
3. Provide estimated effort (low/medium/high)
4. List any risks or considerations

Focus on high-impact, low-effort changes first."""

    prompt = f"""Analyze these code issues and provide refactoring suggestions:

{context}

Provide 5-7 specific, prioritized refactoring recommendations."""

    return generate_with_litellm(prompt, model=model, system=system)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python mcp_litellm_refactor.py /path/to/project [options]")
        print("")
        print("Options:")
        print("  --model MODEL      LLM model to use (default: ollama/qwen2.5-coder:7b)")
        print("  --provider PROV    Use default model for provider (ollama/openai/anthropic)")
        print("  --list-models      List available Ollama models")
        print("  --no-llm           Skip LLM suggestions")
        sys.exit(1)
    
    # Parse arguments
    project_path = sys.argv[1]
    model = PROVIDER_MODELS['ollama']  # default
    use_llm = '--no-llm' not in sys.argv
    
    if '--list-models' in sys.argv:
        print("Available Ollama models:")
        for m in get_ollama_models():
            print(f"  - ollama/{m}")
        sys.exit(0)
    
    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        model = sys.argv[idx + 1]
    
    if '--provider' in sys.argv:
        idx = sys.argv.index('--provider')
        provider = sys.argv[idx + 1]
        model = PROVIDER_MODELS.get(provider, model)
    
    print("="*70)
    print("LITELLM REFACTORING ANALYSIS")
    print("="*70)
    
    # Check providers
    print("\nProvider Status:")
    print(f"  LiteLLM: {'✓' if LITELLM_AVAILABLE else '✗'}")
    print(f"  Ollama:  {'✓' if check_ollama_available() else '✗'}")
    print(f"  Model:   {model}")
    
    # Analyze project
    print(f"\nAnalyzing: {project_path}")
    project = analyze_project(project_path)
    print(f"Found {project.total_files} files, {project.total_lines} lines")
    
    # Find issues
    print("\nIdentifying refactoring opportunities...")
    issues = analyze_for_refactoring(project)
    
    # Summary
    print("\n" + "-"*70)
    print("ISSUES SUMMARY")
    print("-"*70)
    print(f"  Long files (>300 lines):     {len(issues['long_files'])}")
    print(f"  Long functions (>40 lines):  {len(issues['long_functions'])}")
    print(f"  Complex functions (>10):     {len(issues['complex_functions'])}")
    print(f"  Potential duplicates:        {len(issues['duplicates'])}")
    print(f"  Large classes (>15 methods): {len(issues['large_classes'])}")
    
    # LLM suggestions
    if use_llm and LITELLM_AVAILABLE:
        if model.startswith('ollama/') and not check_ollama_available():
            print("\n⚠️  Ollama not running. Start with: ollama serve")
        else:
            print("\n" + "-"*70)
            print("LLM REFACTORING SUGGESTIONS")
            print("-"*70)
            
            suggestions = get_refactoring_suggestions(issues, project, model)
            print(suggestions)
    elif use_llm:
        print("\n⚠️  LiteLLM not installed. Install with: pip install litellm")
    
    # Save results
    output = {
        'project': {
            'name': project.name,
            'files': project.total_files,
            'lines': project.total_lines,
        },
        'issues': {k: len(v) for k, v in issues.items()},
        'details': issues,
        'model': model,
    }
    
    output_file = "litellm_refactoring.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
