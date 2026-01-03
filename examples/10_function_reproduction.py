#!/usr/bin/env python3
"""
Function-Level Reproduction Test.

Tests code2logic's ability to reproduce individual functions within files
across different programming languages.

Usage:
    python examples/10_function_reproduction.py
    python examples/10_function_reproduction.py --function calculate_total
    python examples/10_function_reproduction.py --all-functions
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from difflib import unified_diff, SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from code2logic import analyze_project, get_client
from code2logic.parsers import UniversalParser
from code2logic.models import FunctionInfo


@dataclass
class FunctionReproductionResult:
    """Result of function reproduction."""
    file_path: str
    language: str
    function_name: str
    
    original_code: str = ""
    reproduced_code: str = ""
    
    similarity: float = 0.0
    syntax_ok: bool = False
    
    gen_time: float = 0.0
    error: str = ""


def get_function_code(content: str, func: FunctionInfo, language: str) -> str:
    """Extract function code from file content."""
    lines = content.split('\n')
    start = func.start_line - 1  # 0-indexed
    end = func.end_line if func.end_line else start + func.lines
    
    # For Python, include decorators
    if language == 'python' and start > 0:
        # Look back for decorators
        i = start - 1
        while i >= 0 and lines[i].strip().startswith('@'):
            start = i
            i -= 1
    
    return '\n'.join(lines[start:end])


def generate_function_spec(func: FunctionInfo, language: str) -> str:
    """Generate specification for a single function."""
    spec = f"""Function: {func.name}
Language: {language}
Signature: {func.name}({', '.join(func.params)}) -> {func.return_type or 'None'}
Description: {func.intent or func.docstring or 'No description'}
Is Async: {func.is_async}
Decorators: {', '.join(func.decorators) if func.decorators else 'None'}
Calls: {', '.join(func.calls[:5]) if func.calls else 'None'}
Raises: {', '.join(func.raises) if func.raises else 'None'}
Lines: {func.lines}
"""
    return spec


def reproduce_function(func: FunctionInfo, language: str, client) -> Tuple[str, float]:
    """Reproduce a function using LLM."""
    spec = generate_function_spec(func, language)
    
    lang_hints = {
        'python': "Use type hints. Include docstring. Use standard Python conventions.",
        'javascript': "Use modern ES6+ syntax. Add JSDoc comment.",
        'typescript': "Use TypeScript types. Add JSDoc comment.",
        'go': "Use Go conventions. Add documentation comment.",
    }
    
    prompt = f"""Generate ONLY the function code (no explanations) based on this specification:

{spec}

Requirements:
- Generate complete, working {language} function
- {lang_hints.get(language, '')}
- Match the signature exactly
- Implement the described behavior
- Output ONLY the function code, nothing else

```{language}
"""
    
    start = time.time()
    response = client.generate(prompt, max_tokens=2000)
    gen_time = time.time() - start
    
    # Extract code
    code = response
    if '```' in response:
        match = re.search(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
        if match:
            code = match.group(1)
    
    return code.strip(), gen_time


def calculate_similarity(original: str, reproduced: str) -> float:
    """Calculate similarity between original and reproduced code."""
    # Normalize whitespace
    orig_norm = ' '.join(original.split())
    repr_norm = ' '.join(reproduced.split())
    
    return SequenceMatcher(None, orig_norm, repr_norm).ratio() * 100


def test_syntax(code: str, language: str) -> Tuple[bool, str]:
    """Test if code has valid syntax."""
    if language == 'python':
        try:
            compile(code, '<string>', 'exec')
            return True, ""
        except SyntaxError as e:
            return False, str(e)
    
    elif language in ('javascript', 'typescript'):
        # Simple check - look for obvious syntax errors
        try:
            # Check balanced braces
            if code.count('{') != code.count('}'):
                return False, "Unbalanced braces"
            if code.count('(') != code.count(')'):
                return False, "Unbalanced parentheses"
            return True, ""
        except:
            return False, "Unknown error"
    
    elif language == 'go':
        try:
            if code.count('{') != code.count('}'):
                return False, "Unbalanced braces"
            return True, ""
        except:
            return False, "Unknown error"
    
    return True, ""


def replace_function_in_file(
    file_path: Path,
    func: FunctionInfo,
    new_code: str,
    language: str,
) -> Path:
    """Replace a function in file with reproduced version."""
    content = file_path.read_text()
    lines = content.split('\n')
    
    start = func.start_line - 1
    end = func.end_line if func.end_line else start + func.lines
    
    # For Python, include decorators
    if language == 'python' and start > 0:
        i = start - 1
        while i >= 0 and lines[i].strip().startswith('@'):
            start = i
            i -= 1
    
    # Create new content
    new_lines = lines[:start] + [new_code] + lines[end:]
    new_content = '\n'.join(new_lines)
    
    # Save to new file
    output_path = file_path.parent / f"{file_path.stem}_reproduced{file_path.suffix}"
    output_path.write_text(new_content)
    
    return output_path


def test_function_reproduction(
    file_path: str,
    function_name: Optional[str] = None,
    all_functions: bool = False,
    verbose: bool = False,
) -> List[FunctionReproductionResult]:
    """Test function reproduction for a file."""
    
    path = Path(file_path)
    content = path.read_text()
    
    # Detect language
    ext_to_lang = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.go': 'go',
    }
    language = ext_to_lang.get(path.suffix, 'python')
    
    # Parse file
    parser = UniversalParser()
    module = parser.parse(str(path), content, language)
    
    # Get functions to test
    functions = module.functions
    if function_name:
        functions = [f for f in functions if f.name == function_name]
    elif not all_functions:
        functions = functions[:3]  # Limit to first 3
    
    if not functions:
        print(f"No functions found in {file_path}")
        return []
    
    # Initialize LLM
    try:
        client = get_client()
    except Exception as e:
        print(f"LLM not available: {e}")
        return []
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"FUNCTION REPRODUCTION TEST")
    print(f"{'='*70}")
    print(f"File: {file_path}")
    print(f"Language: {language}")
    print(f"Functions: {len(functions)}")
    print(f"{'─'*70}")
    
    for func in functions:
        result = FunctionReproductionResult(
            file_path=str(path),
            language=language,
            function_name=func.name,
        )
        
        try:
            # Get original code
            result.original_code = get_function_code(content, func, language)
            
            # Reproduce
            result.reproduced_code, result.gen_time = reproduce_function(func, language, client)
            
            # Calculate similarity
            result.similarity = calculate_similarity(result.original_code, result.reproduced_code)
            
            # Test syntax
            result.syntax_ok, error = test_syntax(result.reproduced_code, language)
            if error:
                result.error = error
            
            # Print result
            status = "✓" if result.similarity > 50 else "○"
            syntax = "S✓" if result.syntax_ok else "S✗"
            print(f"{status} {func.name:<30} {result.similarity:>5.1f}% {syntax} ({result.gen_time:.1f}s)")
            
            if verbose:
                print(f"\n--- Original ---")
                print(result.original_code[:500])
                print(f"\n--- Reproduced ---")
                print(result.reproduced_code[:500])
                print()
            
        except Exception as e:
            result.error = str(e)
            print(f"✗ {func.name:<30} ERROR: {e}")
        
        results.append(result)
    
    print(f"{'─'*70}")
    
    return results


def run_multi_language_test(verbose: bool = False) -> Dict[str, List[FunctionReproductionResult]]:
    """Run function reproduction test across multiple languages."""
    
    test_files = [
        'tests/samples/sample_functions.py',
        'tests/samples/sample_javascript.js',
        'tests/samples/sample_typescript.ts',
    ]
    
    all_results = {}
    
    for file_path in test_files:
        if Path(file_path).exists():
            results = test_function_reproduction(file_path, all_functions=True, verbose=verbose)
            all_results[file_path] = results
    
    return all_results


def print_comparison(all_results: Dict[str, List[FunctionReproductionResult]]):
    """Print comparison of results across languages."""
    
    print(f"\n{'='*70}")
    print("CROSS-LANGUAGE COMPARISON")
    print(f"{'='*70}")
    
    # Group by language
    by_lang = {}
    for file_path, results in all_results.items():
        if results:
            lang = results[0].language
            if lang not in by_lang:
                by_lang[lang] = []
            by_lang[lang].extend(results)
    
    print(f"\n{'Language':<15} {'Functions':<10} {'Avg Sim%':<10} {'Syntax OK':<10} {'Avg Time':<10}")
    print(f"{'─'*70}")
    
    for lang, results in sorted(by_lang.items()):
        avg_sim = sum(r.similarity for r in results) / len(results) if results else 0
        syntax_ok = sum(1 for r in results if r.syntax_ok)
        avg_time = sum(r.gen_time for r in results) / len(results) if results else 0
        
        print(f"{lang:<15} {len(results):<10} {avg_sim:>8.1f}% {syntax_ok}/{len(results):<8} {avg_time:>8.1f}s")
    
    print(f"{'─'*70}")
    
    # Best/worst functions
    all_funcs = [r for results in all_results.values() for r in results]
    if all_funcs:
        best = max(all_funcs, key=lambda r: r.similarity)
        worst = min(all_funcs, key=lambda r: r.similarity)
        
        print(f"\n🏆 Best reproduction: {best.function_name} ({best.language}) - {best.similarity:.1f}%")
        print(f"📉 Worst reproduction: {worst.function_name} ({worst.language}) - {worst.similarity:.1f}%")


def save_results(all_results: Dict[str, List[FunctionReproductionResult]], output: str):
    """Save results to JSON."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'files': {}
    }
    
    for file_path, results in all_results.items():
        data['files'][file_path] = [
            {
                'function': r.function_name,
                'language': r.language,
                'similarity': r.similarity,
                'syntax_ok': r.syntax_ok,
                'gen_time': r.gen_time,
                'error': r.error,
                'original_lines': len(r.original_code.split('\n')),
                'reproduced_lines': len(r.reproduced_code.split('\n')),
            }
            for r in results
        ]
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n📄 Results saved: {output}")


def demo_function_edit():
    """Demo: Edit a specific function in a file."""
    
    print(f"\n{'='*70}")
    print("DEMO: Function Edit in File")
    print(f"{'='*70}")
    
    # Copy sample file
    source = Path('tests/samples/sample_functions.py')
    if not source.exists():
        print("Sample file not found")
        return
    
    work_dir = Path('examples/output/function_edit')
    work_dir.mkdir(parents=True, exist_ok=True)
    
    target = work_dir / 'sample_functions_copy.py'
    shutil.copy(source, target)
    print(f"Copied: {source} -> {target}")
    
    # Parse and get function
    parser = UniversalParser()
    content = target.read_text()
    module = parser.parse(str(target), content, 'python')
    
    if not module.functions:
        print("No functions found")
        return
    
    func = module.functions[0]  # First function
    print(f"Target function: {func.name}")
    
    # Reproduce
    try:
        client = get_client()
        new_code, gen_time = reproduce_function(func, 'python', client)
        
        # Replace in file
        output = replace_function_in_file(target, func, new_code, 'python')
        print(f"Created: {output}")
        
        # Show diff
        original = get_function_code(content, func, 'python')
        diff = list(unified_diff(
            original.split('\n'),
            new_code.split('\n'),
            fromfile='original',
            tofile='reproduced',
            lineterm=''
        ))
        
        print(f"\n--- Diff ---")
        for line in diff[:30]:
            print(line)
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Function-Level Reproduction Test')
    parser.add_argument('--file', '-f', help='File to test')
    parser.add_argument('--function', '-n', help='Specific function name')
    parser.add_argument('--all-functions', '-a', action='store_true')
    parser.add_argument('--multi-lang', '-m', action='store_true', help='Test multiple languages')
    parser.add_argument('--demo-edit', '-d', action='store_true', help='Demo function editing')
    parser.add_argument('--output', '-o', default='examples/output/function_reproduction.json')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    if args.demo_edit:
        demo_function_edit()
        return
    
    if args.multi_lang:
        all_results = run_multi_language_test(verbose=args.verbose)
        print_comparison(all_results)
        save_results(all_results, args.output)
        return
    
    if args.file:
        results = test_function_reproduction(
            args.file,
            function_name=args.function,
            all_functions=args.all_functions,
            verbose=args.verbose,
        )
        save_results({args.file: results}, args.output)
        return
    
    # Default: test sample_functions.py
    results = test_function_reproduction(
        'tests/samples/sample_functions.py',
        all_functions=True,
        verbose=args.verbose,
    )
    save_results({'tests/samples/sample_functions.py': results}, args.output)


if __name__ == '__main__':
    main()
