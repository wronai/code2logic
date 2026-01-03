#!/usr/bin/env python3
"""
OpenRouter Code Reproduction Example.

Demonstrates complete workflow:
1. Analyze code → Generate Gherkin specification
2. Send Gherkin to LLM → Generate code back
3. Compare original vs generated code

Uses OpenRouter with optimal model (Qwen 2.5 Coder 32B) for code tasks.

Requirements:
    pip install httpx python-dotenv

Configuration:
    export OPENROUTER_API_KEY="sk-or-v1-your-key"
    # or create .env file with OPENROUTER_API_KEY=...

Usage:
    python openrouter_code_reproduction.py
    python openrouter_code_reproduction.py --source code2logic/models.py
    python openrouter_code_reproduction.py --model qwen/qwen-2.5-coder-32b-instruct
"""

import sys
import os
import re
import json
import difflib
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from code2logic import analyze_project
from code2logic.gherkin import GherkinGenerator
from code2logic.config import Config


# OpenRouter configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Recommended models for code tasks (sorted by quality/cost)
RECOMMENDED_MODELS = [
    ("qwen/qwen-2.5-coder-32b-instruct", "Best for code, 32B, fast"),
    ("deepseek/deepseek-coder-33b-instruct", "DeepSeek Coder 33B"),
    ("codellama/codellama-34b-instruct", "CodeLlama 34B"),
    ("mistralai/codestral-latest", "Codestral by Mistral"),
    ("anthropic/claude-3-haiku", "Fast, good for code"),
]


class OpenRouterClient:
    """OpenRouter API client for code generation."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        self.model = model or os.environ.get('OPENROUTER_MODEL', 'qwen/qwen-2.5-coder-32b-instruct')
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter.\n"
                "Get your key at: https://openrouter.ai/keys"
            )
    
    def generate(self, prompt: str, system: str = None, max_tokens: int = 4000) -> str:
        """Generate completion using OpenRouter."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required: pip install httpx")
        
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
            "temperature": 0.2,  # Low temperature for code generation
        }
        
        try:
            response = httpx.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )
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


def read_source_file(file_path: str) -> str:
    """Read source file content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def analyze_to_gherkin(source_path: str, detail: str = 'full') -> str:
    """Analyze source code and generate Gherkin specification."""
    path = Path(source_path)
    
    if path.is_file():
        # For single file, generate custom Gherkin from file content
        return generate_file_gherkin(path)
    else:
        # For directory, use standard analysis
        project = analyze_project(source_path)
        generator = GherkinGenerator()
        return generator.generate(project, detail=detail)


def generate_file_gherkin(file_path: Path) -> str:
    """Generate detailed Gherkin specification for a single file with types."""
    content = file_path.read_text()
    
    # Extract structure from file
    classes = []
    functions = []
    imports = []
    module_doc = ""
    
    lines = content.split('\n')
    in_class = None
    in_docstring = False
    in_class_docstring = False
    docstring_lines = []
    class_docstring = ""
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        raw_line = line
        
        # Module docstring (first few lines)
        if i < 5 and stripped.startswith('"""') and not module_doc:
            if stripped.count('"""') >= 2:
                module_doc = stripped.strip('"""').strip()
            else:
                in_docstring = True
                docstring_lines = [stripped.lstrip('"""')]
                continue
        
        if in_docstring and not in_class:
            if '"""' in stripped:
                docstring_lines.append(stripped.rstrip('"""'))
                module_doc = ' '.join(docstring_lines)[:200]
                in_docstring = False
            else:
                docstring_lines.append(stripped)
            continue
        
        # Class docstring
        if in_class_docstring:
            if '"""' in stripped:
                docstring_lines.append(stripped.rstrip('"""'))
                class_docstring = ' '.join(docstring_lines)[:150]
                classes[-1]['docstring'] = class_docstring
                in_class_docstring = False
            else:
                docstring_lines.append(stripped)
            continue
        
        # Imports
        if stripped.startswith('from ') or stripped.startswith('import '):
            imports.append(stripped)
        
        # Classes
        if stripped.startswith('class '):
            class_name = stripped.split('(')[0].split(':')[0].replace('class ', '')
            in_class = class_name
            classes.append({'name': class_name, 'attributes': [], 'methods': [], 'docstring': ''})
        
        # Class docstring start
        if in_class and stripped.startswith('"""') and not classes[-1]['docstring']:
            if stripped.count('"""') >= 2:
                classes[-1]['docstring'] = stripped.strip('"""').strip()[:100]
            else:
                in_class_docstring = True
                docstring_lines = [stripped.lstrip('"""')]
            continue
        
        # Class attributes with FULL type info (only for dataclasses or class-level attributes)
        if in_class and ':' in stripped and not stripped.startswith('def ') and not stripped.startswith('#'):
            # Skip docstring-like lines and common patterns
            if stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if any(stripped.startswith(x) for x in ['Attributes', '-', 'Args', 'Returns', 'Raises', 'Note', 'Example']):
                continue
            # Skip lines that look like docstring content (contain common doc patterns)
            if any(x in stripped.lower() for x in ['path to', 'the ', 'a ', 'an ', 'this ', 'that ', 'if ', 'when ']):
                continue
            # Skip lines starting with lowercase that aren't valid attribute patterns
            if stripped[0].islower() and not re.match(r'^[a-z_][a-z0-9_]*\s*:', stripped):
                continue
            
            # Parse full attribute definition: name: Type = default
            attr_full = stripped
            if attr_full and not attr_full.startswith('return'):
                attr_name = attr_full.split(':')[0].strip()
                # Must be valid Python identifier and not a keyword
                if attr_name and attr_name.isidentifier() and attr_name not in ['try', 'if', 'for', 'while', 'class', 'def', 'return']:
                    # Check if not already added
                    existing = [a['name'] for a in classes[-1]['attributes'] if isinstance(a, dict)]
                    if attr_name not in existing:
                        classes[-1]['attributes'].append({
                            'name': attr_name,
                            'full': attr_full
                        })
        
        # Functions/methods with signatures
        if stripped.startswith('def '):
            # Extract full signature
            func_line = stripped
            if func_line.endswith(':'):
                func_line = func_line[:-1]
            func_name = func_line.split('(')[0].replace('def ', '')
            
            # Get params and return type
            try:
                params_part = func_line.split('(', 1)[1].rsplit(')', 1)[0]
                return_part = func_line.split('->')[-1].strip() if '->' in func_line else 'None'
            except:
                params_part = ''
                return_part = 'None'
            
            func_info = {
                'name': func_name,
                'params': params_part,
                'returns': return_part,
                'full': func_line
            }
            
            if in_class:
                classes[-1]['methods'].append(func_info)
            else:
                functions.append(func_info)
    
    # Generate detailed Gherkin
    gherkin_lines = [
        f"# Gherkin specification for {file_path.name}",
        f"# {module_doc}" if module_doc else "# Python module",
        "",
        f"@{file_path.stem}",
        f"Feature: {file_path.stem} module",
        f"  {module_doc[:100]}" if module_doc else f"  Source file: {file_path.name}",
        "",
    ]
    
    if imports:
        gherkin_lines.append("  Background:")
        gherkin_lines.append("    Given the following imports are required:")
        for imp in imports[:10]:
            gherkin_lines.append(f"      | {imp} |")
        gherkin_lines.append("")
    
    is_dataclass = '@dataclass' in content
    
    for cls in classes:
        gherkin_lines.append(f"  @dataclass" if is_dataclass else f"  @class")
        gherkin_lines.append(f"  Scenario: Define {cls['name']} {'dataclass' if is_dataclass else 'class'}")
        
        if cls['docstring']:
            gherkin_lines.append(f"    # {cls['docstring'][:80]}")
        
        gherkin_lines.append(f"    Given a {'dataclass' if is_dataclass else 'class'} named \"{cls['name']}\"")
        
        if cls['attributes']:
            gherkin_lines.append("    Then it should have the following typed attributes:")
            gherkin_lines.append("      | name | type | default |")
            for attr in cls['attributes']:
                if isinstance(attr, dict):
                    full = attr['full']
                    name = attr['name']
                    # Parse type and default
                    type_part = ''
                    default_part = ''
                    if ':' in full:
                        after_colon = full.split(':', 1)[1].strip()
                        if '=' in after_colon:
                            type_part = after_colon.split('=')[0].strip()
                            default_part = after_colon.split('=', 1)[1].strip()
                        else:
                            type_part = after_colon
                    gherkin_lines.append(f"      | {name} | {type_part} | {default_part} |")
                else:
                    gherkin_lines.append(f"      | {attr} | | |")
        
        if cls['methods']:
            gherkin_lines.append("    And it should have the following methods:")
            gherkin_lines.append("      | name | params | returns |")
            for method in cls['methods']:
                if isinstance(method, dict):
                    gherkin_lines.append(f"      | {method['name']} | {method['params'][:50]} | {method['returns']} |")
                else:
                    gherkin_lines.append(f"      | {method} | | |")
        
        gherkin_lines.append("")
    
    for func in functions:
        if isinstance(func, dict):
            gherkin_lines.append(f"  Scenario: Define {func['name']} function")
            gherkin_lines.append(f"    Given a function named \"{func['name']}\"")
            gherkin_lines.append(f"    With parameters: {func['params']}")
            gherkin_lines.append(f"    And returns: {func['returns']}")
        else:
            gherkin_lines.append(f"  Scenario: Define {func} function")
            gherkin_lines.append(f"    Given a function named \"{func}\"")
        gherkin_lines.append("")
    
    return '\n'.join(gherkin_lines)


def generate_code_from_gherkin(client: OpenRouterClient, gherkin: str, language: str = 'python') -> str:
    """Use LLM to generate code from Gherkin specification."""
    
    system_prompt = f"""You are an expert {language} developer. Your task is to generate clean, 
production-ready {language} code based on the given Gherkin/BDD specification.

Rules:
1. Generate ONLY code, no explanations
2. Include all necessary imports
3. Add docstrings to all functions and classes
4. Follow PEP 8 style guidelines
5. Include type hints
6. The code must be complete and runnable

Output format: Return ONLY the code wrapped in ```{language} ... ``` blocks."""

    prompt = f"""Generate {language} code that implements this Gherkin specification:

{gherkin}

Generate complete, working {language} code with all classes, functions, and methods described.
Include proper docstrings and type hints."""

    response = client.generate(prompt, system=system_prompt, max_tokens=8000)
    
    # Extract code from response
    code = extract_code_block(response, language)
    return code


def extract_code_block(text: str, language: str = 'python') -> str:
    """Extract code block from LLM response."""
    # Try to find code block
    markers = [f'```{language}', '```py', '```']
    
    for marker in markers:
        if marker in text:
            start = text.find(marker) + len(marker)
            end = text.find('```', start)
            if end > start:
                return text[start:end].strip()
    
    # No code block found, return as-is (might already be code)
    return text.strip()


def compare_code(original: str, generated: str) -> Dict[str, Any]:
    """Compare original and generated code."""
    
    # Normalize whitespace for comparison
    def normalize(code: str) -> List[str]:
        lines = code.strip().split('\n')
        return [line.rstrip() for line in lines if line.strip()]
    
    orig_lines = normalize(original)
    gen_lines = normalize(generated)
    
    # Calculate similarity using difflib
    matcher = difflib.SequenceMatcher(None, orig_lines, gen_lines)
    similarity = matcher.ratio() * 100
    
    # Get diff
    diff = list(difflib.unified_diff(
        orig_lines, gen_lines,
        fromfile='original',
        tofile='generated',
        lineterm=''
    ))
    
    # Count structural elements
    def count_elements(code: str) -> Dict[str, int]:
        return {
            'classes': code.count('class '),
            'functions': code.count('def '),
            'imports': code.count('import '),
            'docstrings': code.count('"""'),
            'lines': len(code.strip().split('\n')),
            'chars': len(code),
        }
    
    orig_elements = count_elements(original)
    gen_elements = count_elements(generated)
    
    # Calculate structural match
    struct_matches = sum(
        1 for k in orig_elements 
        if orig_elements[k] == gen_elements.get(k, 0)
    )
    struct_score = (struct_matches / len(orig_elements)) * 100
    
    return {
        'similarity_percent': round(similarity, 2),
        'structural_score': round(struct_score, 2),
        'original_elements': orig_elements,
        'generated_elements': gen_elements,
        'diff_lines': len([d for d in diff if d.startswith('+') or d.startswith('-')]),
        'diff': '\n'.join(diff[:50]),  # First 50 lines of diff
    }


def save_results(output_dir: Path, results: Dict[str, Any]):
    """Save comparison results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original code
    (output_dir / 'original.py').write_text(results['original_code'])
    
    # Save Gherkin specification
    (output_dir / 'specification.feature').write_text(results['gherkin'])
    
    # Save generated code
    (output_dir / 'generated.py').write_text(results['generated_code'])
    
    # Save comparison report
    report = generate_report(results)
    (output_dir / 'COMPARISON_REPORT.md').write_text(report)
    
    # Save JSON metrics
    metrics = {
        'timestamp': results['timestamp'],
        'source_file': results['source_file'],
        'model': results['model'],
        'comparison': results['comparison'],
    }
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))


def generate_report(results: Dict[str, Any]) -> str:
    """Generate markdown comparison report."""
    comp = results['comparison']
    
    report = f"""# Code Reproduction Comparison Report

> Generated: {results['timestamp']}
> Source: `{results['source_file']}`
> Model: `{results['model']}`

## Summary

| Metric | Value |
|--------|-------|
| **Similarity** | {comp['similarity_percent']}% |
| **Structural Score** | {comp['structural_score']}% |
| **Diff Lines** | {comp['diff_lines']} |

## Structural Comparison

| Element | Original | Generated | Match |
|---------|----------|-----------|-------|
"""
    
    for key in comp['original_elements']:
        orig = comp['original_elements'][key]
        gen = comp['generated_elements'].get(key, 0)
        match = "✓" if orig == gen else "✗"
        report += f"| {key} | {orig} | {gen} | {match} |\n"
    
    report += f"""
## Quality Assessment

"""
    
    # Quality rating based on scores
    avg_score = (comp['similarity_percent'] + comp['structural_score']) / 2
    if avg_score >= 80:
        rating = "⭐⭐⭐⭐⭐ Excellent"
    elif avg_score >= 60:
        rating = "⭐⭐⭐⭐ Good"
    elif avg_score >= 40:
        rating = "⭐⭐⭐ Fair"
    elif avg_score >= 20:
        rating = "⭐⭐ Poor"
    else:
        rating = "⭐ Needs Improvement"
    
    report += f"**Overall Rating:** {rating} ({avg_score:.1f}%)\n\n"
    
    if avg_score >= 70:
        report += "The generated code closely matches the original structure and logic.\n"
    elif avg_score >= 40:
        report += "The generated code captures the main concepts but differs in implementation details.\n"
    else:
        report += "The generated code diverges significantly from the original. Consider using a more detailed specification.\n"
    
    report += f"""
## Diff (first 50 lines)

```diff
{comp['diff']}
```

## Files Generated

- `original.py` - Source code
- `specification.feature` - Gherkin specification
- `generated.py` - LLM-generated code
- `metrics.json` - Comparison metrics
"""
    
    return report


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Code reproduction using OpenRouter LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python openrouter_code_reproduction.py
    python openrouter_code_reproduction.py --source code2logic/models.py
    python openrouter_code_reproduction.py --model deepseek/deepseek-coder-33b-instruct
    
Environment:
    OPENROUTER_API_KEY  - Your OpenRouter API key
    OPENROUTER_MODEL    - Model to use (default: qwen/qwen-2.5-coder-32b-instruct)
"""
    )
    
    parser.add_argument(
        '--source', '-s',
        default='code2logic/models.py',
        help='Source file or directory to analyze'
    )
    parser.add_argument(
        '--model', '-m',
        default=None,
        help='OpenRouter model to use'
    )
    parser.add_argument(
        '--output', '-o',
        default='examples/reproduction/openrouter_test',
        help='Output directory for results'
    )
    parser.add_argument(
        '--detail', '-d',
        choices=['minimal', 'standard', 'full'],
        default='full',
        help='Gherkin detail level'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List recommended models and exit'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate Gherkin but skip LLM call'
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        print("Recommended models for code tasks:\n")
        for model, desc in RECOMMENDED_MODELS:
            print(f"  {model:<45} - {desc}")
        print("\nSet with: export OPENROUTER_MODEL=<model>")
        sys.exit(0)
    
    print("="*60)
    print("CODE2LOGIC - OPENROUTER CODE REPRODUCTION")
    print("="*60)
    
    # Check configuration
    config = Config()
    api_key = config.get_api_key('openrouter')
    
    if not api_key and not args.dry_run:
        print("\n⚠️  OpenRouter API key not configured!")
        print("\nTo configure:")
        print("  1. Get API key from https://openrouter.ai/keys")
        print("  2. Set environment variable:")
        print('     export OPENROUTER_API_KEY="sk-or-v1-your-key"')
        print("  3. Or create .env file with OPENROUTER_API_KEY=...")
        print("\nRunning in dry-run mode (Gherkin only)...")
        args.dry_run = True
    
    model = args.model or config.get_model('openrouter')
    
    print(f"\nSource: {args.source}")
    print(f"Model: {model}")
    print(f"Detail: {args.detail}")
    print(f"Output: {args.output}")
    
    # Step 1: Read original code
    print("\n" + "-"*40)
    print("Step 1: Reading source code...")
    
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source not found: {args.source}")
        sys.exit(1)
    
    if source_path.is_file():
        original_code = read_source_file(str(source_path))
        print(f"  Read {len(original_code):,} chars from {source_path.name}")
    else:
        # For directories, use a sample file
        py_files = list(source_path.glob('*.py'))
        if not py_files:
            print(f"Error: No Python files in {args.source}")
            sys.exit(1)
        sample_file = py_files[0]
        original_code = read_source_file(str(sample_file))
        print(f"  Using sample: {sample_file.name} ({len(original_code):,} chars)")
    
    # Step 2: Generate Gherkin
    print("\n" + "-"*40)
    print("Step 2: Generating Gherkin specification...")
    
    gherkin = analyze_to_gherkin(args.source, detail=args.detail)
    print(f"  Generated {len(gherkin):,} chars of Gherkin")
    print(f"  Features: {gherkin.count('Feature:')}")
    print(f"  Scenarios: {gherkin.count('Scenario')}")
    
    if args.dry_run:
        print("\n" + "-"*40)
        print("DRY RUN - Skipping LLM generation")
        print("-"*40)
        print("\nGherkin specification preview:")
        print(gherkin[:2000])
        if len(gherkin) > 2000:
            print(f"\n... ({len(gherkin) - 2000} more chars)")
        sys.exit(0)
    
    # Step 3: Generate code from Gherkin
    print("\n" + "-"*40)
    print("Step 3: Generating code with LLM...")
    
    client = OpenRouterClient(api_key=api_key, model=model)
    
    try:
        generated_code = generate_code_from_gherkin(client, gherkin)
        print(f"  Generated {len(generated_code):,} chars of code")
    except Exception as e:
        print(f"  Error: {e}")
        sys.exit(1)
    
    # Step 4: Compare codes
    print("\n" + "-"*40)
    print("Step 4: Comparing original vs generated...")
    
    comparison = compare_code(original_code, generated_code)
    print(f"  Similarity: {comparison['similarity_percent']}%")
    print(f"  Structural score: {comparison['structural_score']}%")
    print(f"  Diff lines: {comparison['diff_lines']}")
    
    # Step 5: Save results
    print("\n" + "-"*40)
    print("Step 5: Saving results...")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'source_file': args.source,
        'model': model,
        'original_code': original_code,
        'gherkin': gherkin,
        'generated_code': generated_code,
        'comparison': comparison,
    }
    
    output_dir = Path(args.output)
    save_results(output_dir, results)
    print(f"  Results saved to: {output_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_score = (comparison['similarity_percent'] + comparison['structural_score']) / 2
    
    print(f"\n  Similarity:       {comparison['similarity_percent']}%")
    print(f"  Structural Match: {comparison['structural_score']}%")
    print(f"  Average Score:    {avg_score:.1f}%")
    
    if avg_score >= 70:
        print("\n  ✓ Good reproduction quality!")
    elif avg_score >= 40:
        print("\n  ~ Fair reproduction, some differences")
    else:
        print("\n  ✗ Low reproduction quality")
    
    print(f"\n  View results: {output_dir}/COMPARISON_REPORT.md")


if __name__ == '__main__':
    main()
