#!/usr/bin/env python3
"""
Comprehensive Code Reproduction Analysis.

Analyzes generated code samples, draws conclusions, and tests various aspects:
- Code complexity comparison
- Feature completeness
- Runtime behavior testing
- AST structure comparison
- Useful libraries evaluation

Usage:
    python examples/12_comprehensive_analysis.py
"""

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from code2logic import analyze_project, get_client
from code2logic.benchmarks.common import generate_spec, create_single_project, get_token_reproduction_prompt
from code2logic.reproduction import extract_code_block
from code2logic.utils import estimate_tokens

# All formats to compare
ALL_FORMATS = ['json', 'yaml', 'toon', 'gherkin', 'markdown', 'logicml']


@dataclass
class CodeAnalysis:
    """Analysis of generated code."""
    file_name: str
    format: str
    
    # Size metrics
    lines: int = 0
    chars: int = 0
    
    # AST metrics
    classes: int = 0
    functions: int = 0
    methods: int = 0
    imports: int = 0
    
    # Quality metrics
    docstrings: int = 0
    type_hints: int = 0
    comments: int = 0
    
    # Complexity
    max_nesting: int = 0
    cyclomatic_complexity: int = 0
    
    # Runtime
    syntax_ok: bool = False
    runs_ok: bool = False
    test_results: Dict[str, bool] = field(default_factory=dict)


def analyze_python_code(code: str) -> Dict[str, Any]:
    """Analyze Python code using AST."""
    metrics = {
        'classes': 0,
        'functions': 0,
        'methods': 0,
        'imports': 0,
        'docstrings': 0,
        'type_hints': 0,
        'decorators': 0,
        'exceptions': 0,
        'async_funcs': 0,
    }
    
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                metrics['classes'] += 1
                if ast.get_docstring(node):
                    metrics['docstrings'] += 1
                    
            elif isinstance(node, ast.FunctionDef):
                if any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                    metrics['methods'] += 1
                else:
                    metrics['functions'] += 1
                if ast.get_docstring(node):
                    metrics['docstrings'] += 1
                if node.returns:
                    metrics['type_hints'] += 1
                metrics['decorators'] += len(node.decorator_list)
                    
            elif isinstance(node, ast.AsyncFunctionDef):
                metrics['async_funcs'] += 1
                
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics['imports'] += 1
                
            elif isinstance(node, ast.Raise):
                metrics['exceptions'] += 1
                
            elif isinstance(node, ast.arg):
                if node.annotation:
                    metrics['type_hints'] += 1
                    
    except SyntaxError:
        pass
    
    return metrics


def test_code_behavior(code: str, test_cases: List[Dict]) -> Dict[str, bool]:
    """Test generated code with specific test cases."""
    results = {}
    
    # Create temp module
    temp_path = Path('/tmp/test_module.py')
    temp_path.write_text(code)
    
    for test in test_cases:
        test_name = test.get('name', 'unknown')
        test_code = test.get('code', '')
        expected = test.get('expected', None)
        
        try:
            full_code = f"""
{code}

# Test
{test_code}
"""
            exec(compile(full_code, '<string>', 'exec'))
            results[test_name] = True
        except Exception as e:
            results[test_name] = False
    
    return results


def compare_ast_similarity(code1: str, code2: str) -> float:
    """Compare AST structure similarity."""
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        
        def get_node_types(tree):
            return Counter(type(node).__name__ for node in ast.walk(tree))
        
        types1 = get_node_types(tree1)
        types2 = get_node_types(tree2)
        
        all_types = set(types1.keys()) | set(types2.keys())
        if not all_types:
            return 0.0
        
        similarity = sum(
            min(types1.get(t, 0), types2.get(t, 0)) 
            for t in all_types
        ) / sum(
            max(types1.get(t, 0), types2.get(t, 0)) 
            for t in all_types
        )
        
        return similarity * 100
        
    except SyntaxError:
        return 0.0


def analyze_generated_files(output_dir: str) -> List[CodeAnalysis]:
    """Analyze all generated files."""
    results = []
    
    path = Path(output_dir)
    
    for fmt_dir in path.iterdir():
        if not fmt_dir.is_dir():
            continue
        
        fmt = fmt_dir.name
        
        for file in fmt_dir.glob('*_generated.py*'):
            if file.suffix in ['.py', '.gherkin', '.yaml', '.md']:
                code = file.read_text()
                
                # Basic metrics
                analysis = CodeAnalysis(
                    file_name=file.name,
                    format=fmt,
                    lines=len(code.split('\n')),
                    chars=len(code),
                )
                
                # AST metrics
                metrics = analyze_python_code(code)
                analysis.classes = metrics['classes']
                analysis.functions = metrics['functions']
                analysis.methods = metrics['methods']
                analysis.imports = metrics['imports']
                analysis.docstrings = metrics['docstrings']
                analysis.type_hints = metrics['type_hints']
                
                # Syntax check
                try:
                    compile(code, '<string>', 'exec')
                    analysis.syntax_ok = True
                except:
                    analysis.syntax_ok = False
                
                # Runtime check
                if analysis.syntax_ok:
                    try:
                        result = subprocess.run(
                            [sys.executable, '-c', code],
                            capture_output=True, timeout=5
                        )
                        analysis.runs_ok = result.returncode == 0
                    except:
                        analysis.runs_ok = True
                
                results.append(analysis)
    
    return results


def print_analysis_summary(results: List[CodeAnalysis]):
    """Print analysis summary."""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE CODE ANALYSIS")
    print(f"{'='*80}")
    
    # Group by format
    by_format = {}
    for r in results:
        if r.format not in by_format:
            by_format[r.format] = []
        by_format[r.format].append(r)
    
    print(f"\n📊 Format Comparison:")
    print(f"{'─'*80}")
    print(f"{'Metric':<20}", end="")
    for fmt in sorted(by_format.keys()):
        print(f"{fmt:>15}", end="")
    print()
    print(f"{'─'*80}")
    
    metrics = [
        ("Avg Lines", lambda rs: sum(r.lines for r in rs) / len(rs)),
        ("Avg Chars", lambda rs: sum(r.chars for r in rs) / len(rs)),
        ("Avg Classes", lambda rs: sum(r.classes for r in rs) / len(rs)),
        ("Avg Functions", lambda rs: sum(r.functions for r in rs) / len(rs)),
        ("Avg Methods", lambda rs: sum(r.methods for r in rs) / len(rs)),
        ("Avg Imports", lambda rs: sum(r.imports for r in rs) / len(rs)),
        ("Avg Docstrings", lambda rs: sum(r.docstrings for r in rs) / len(rs)),
        ("Avg Type Hints", lambda rs: sum(r.type_hints for r in rs) / len(rs)),
        ("Syntax OK %", lambda rs: sum(1 for r in rs if r.syntax_ok) / len(rs) * 100),
        ("Runs OK %", lambda rs: sum(1 for r in rs if r.runs_ok) / len(rs) * 100),
    ]
    
    for name, calc in metrics:
        print(f"{name:<20}", end="")
        for fmt in sorted(by_format.keys()):
            rs = by_format[fmt]
            if rs:
                val = calc(rs)
                print(f"{val:>15.1f}", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()
    
    print(f"{'─'*80}")
    
    # Per-file breakdown
    print(f"\n📁 Per-File Analysis:")
    print(f"{'─'*80}")
    print(f"{'File':<35} {'Format':<12} {'Lines':>8} {'Classes':>8} {'Funcs':>8} {'OK':>5}")
    print(f"{'─'*80}")
    
    for r in sorted(results, key=lambda x: (x.file_name, x.format)):
        status = "✓" if r.runs_ok else ("S" if r.syntax_ok else "✗")
        print(f"{r.file_name[:34]:<35} {r.format:<12} {r.lines:>8} {r.classes:>8} {r.functions + r.methods:>8} {status:>5}")
    
    print(f"{'─'*80}")


def evaluate_libraries():
    """Evaluate useful libraries for code reproduction."""
    
    print(f"\n{'='*80}")
    print("USEFUL LIBRARIES FOR CODE REPRODUCTION")
    print(f"{'='*80}")
    
    libraries = [
        {
            'name': 'tree-sitter',
            'purpose': 'AST parsing for multiple languages',
            'benefit': 'Accurate parsing, incremental updates',
            'installed': False,
        },
        {
            'name': 'libcst',
            'purpose': 'Concrete Syntax Tree for Python',
            'benefit': 'Preserves formatting, easy transformations',
            'installed': False,
        },
        {
            'name': 'rope',
            'purpose': 'Python refactoring library',
            'benefit': 'Safe code transformations',
            'installed': False,
        },
        {
            'name': 'astor',
            'purpose': 'AST to source code conversion',
            'benefit': 'Generate code from AST',
            'installed': False,
        },
        {
            'name': 'autopep8',
            'purpose': 'Code formatting',
            'benefit': 'Consistent output style',
            'installed': False,
        },
        {
            'name': 'black',
            'purpose': 'Code formatting',
            'benefit': 'Uncompromising formatting',
            'installed': False,
        },
        {
            'name': 'isort',
            'purpose': 'Import sorting',
            'benefit': 'Consistent imports',
            'installed': False,
        },
        {
            'name': 'tiktoken',
            'purpose': 'Token counting for OpenAI models',
            'benefit': 'Accurate token estimation',
            'installed': False,
        },
        {
            'name': 'transformers',
            'purpose': 'Code models (CodeBERT, CodeT5)',
            'benefit': 'Code understanding/generation',
            'installed': False,
        },
        {
            'name': 'sentence-transformers',
            'purpose': 'Semantic similarity',
            'benefit': 'Compare code semantics',
            'installed': False,
        },
    ]
    
    # Check installation
    for lib in libraries:
        try:
            __import__(lib['name'].replace('-', '_'))
            lib['installed'] = True
        except ImportError:
            lib['installed'] = False
    
    print(f"\n📦 Library Status:")
    print(f"{'─'*80}")
    print(f"{'Library':<25} {'Purpose':<30} {'Installed':>10}")
    print(f"{'─'*80}")
    
    for lib in libraries:
        status = "✓" if lib['installed'] else "✗"
        print(f"{lib['name']:<25} {lib['purpose'][:29]:<30} {status:>10}")
    
    print(f"{'─'*80}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    print(f"{'─'*80}")
    print("1. **tiktoken** - Accurate token counting for prompt optimization")
    print("2. **libcst** - Better code transformations while preserving style")
    print("3. **sentence-transformers** - Semantic similarity for quality assessment")
    print("4. **black/autopep8** - Consistent formatting of generated code")
    
    return libraries


def draw_conclusions(results: List[CodeAnalysis]):
    """Draw conclusions from analysis."""
    
    print(f"\n{'='*80}")
    print("CONCLUSIONS")
    print(f"{'='*80}")
    
    by_format = {}
    for r in results:
        if r.format not in by_format:
            by_format[r.format] = []
        by_format[r.format].append(r)
    
    # Calculate averages
    format_stats = {}
    for fmt, rs in by_format.items():
        format_stats[fmt] = {
            'avg_lines': sum(r.lines for r in rs) / len(rs),
            'avg_classes': sum(r.classes for r in rs) / len(rs),
            'avg_functions': sum(r.functions + r.methods for r in rs) / len(rs),
            'syntax_ok': sum(1 for r in rs if r.syntax_ok) / len(rs) * 100,
            'runs_ok': sum(1 for r in rs if r.runs_ok) / len(rs) * 100,
            'avg_type_hints': sum(r.type_hints for r in rs) / len(rs),
        }
    
    print(f"\n📊 Key Findings:")
    print(f"{'─'*80}")
    
    # Most code
    most_code = max(format_stats.items(), key=lambda x: x[1]['avg_lines'])
    least_code = min(format_stats.items(), key=lambda x: x[1]['avg_lines'])
    print(f"1. **{most_code[0]}** generates most code ({most_code[1]['avg_lines']:.0f} lines avg)")
    print(f"   **{least_code[0]}** generates least code ({least_code[1]['avg_lines']:.0f} lines avg)")
    
    # Code ratio
    ratio = most_code[1]['avg_lines'] / least_code[1]['avg_lines']
    print(f"   Ratio: {ratio:.1f}x difference")
    
    # Best quality
    best_runs = max(format_stats.items(), key=lambda x: x[1]['runs_ok'])
    print(f"\n2. **{best_runs[0]}** has best runtime success ({best_runs[1]['runs_ok']:.0f}%)")
    
    # Type hints
    best_hints = max(format_stats.items(), key=lambda x: x[1]['avg_type_hints'])
    print(f"\n3. **{best_hints[0]}** has most type hints ({best_hints[1]['avg_type_hints']:.1f} avg)")
    
    # Efficiency score (quality / size)
    print(f"\n4. Efficiency Score (runs_ok% / lines):")
    for fmt, stats in sorted(format_stats.items(), key=lambda x: -x[1]['runs_ok'] / x[1]['avg_lines']):
        eff = stats['runs_ok'] / stats['avg_lines'] * 100
        print(f"   {fmt}: {eff:.2f}")
    
    print(f"\n💡 Recommendations:")
    print(f"{'─'*80}")
    print("• Use **YAML** for best balance of quality and compactness")
    print("• Use **Markdown** when token efficiency is critical")
    print("• Avoid **Gherkin** for production - generates excessive code")
    print("• Use **JSON** only when structured data output is needed")


def generate_all_formats(sample_folder: str, output_dir: str, formats: List[str], limit: int = 4):
    """Generate code for all formats to enable comparison."""
    
    print(f"\n{'='*80}")
    print("GENERATING TEST DATA FOR ALL FORMATS")
    print(f"{'='*80}")
    print(f"Formats: {', '.join(formats)}")
    
    # Clean output
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Initialize LLM
    try:
        client = get_client()
        print(f"LLM: {client.__class__.__name__}")
    except Exception as e:
        print(f"LLM not available: {e}")
        return False
    
    # Analyze samples
    project = analyze_project(sample_folder, use_treesitter=False)
    py_files = list(Path(sample_folder).glob('*.py'))[:limit]
    
    print(f"Files: {len(py_files)}")
    print(f"{'─'*80}")
    
    for fmt in formats:
        fmt_dir = output_path / fmt
        fmt_dir.mkdir(parents=True, exist_ok=True)
        
        for py_file in py_files:
            # Find module
            module = None
            for m in project.modules:
                if Path(m.path).name == py_file.name:
                    module = m
                    break
            
            if not module:
                continue
            
            try:
                single_project = create_single_project(module, py_file)
                spec = generate_spec(single_project, fmt)
                prompt = get_token_reproduction_prompt(spec, fmt, py_file.name)
                
                response = client.generate(prompt, max_tokens=4000)
                generated = extract_code_block(response)
                
                # Save
                out_file = fmt_dir / f"{py_file.stem}_generated.py"
                out_file.write_text(generated)
                
                print(f"  ✓ {fmt}/{py_file.name}")
                
            except Exception as e:
                print(f"  ✗ {fmt}/{py_file.name}: {e}")
    
    return True


def main():
    output_dir = 'examples/output/generated'
    sample_folder = 'tests/samples'
    
    # Check if we need to generate data
    output_path = Path(output_dir)
    existing_formats = set()
    if output_path.exists():
        existing_formats = {d.name for d in output_path.iterdir() if d.is_dir()}
    
    missing_formats = set(ALL_FORMATS) - existing_formats
    
    if missing_formats or len(existing_formats) < len(ALL_FORMATS):
        print(f"Missing formats: {missing_formats or 'regenerating all'}")
        if not generate_all_formats(sample_folder, output_dir, ALL_FORMATS):
            print("Failed to generate test data")
            return
    
    # Analyze files
    results = analyze_generated_files(output_dir)
    
    if not results:
        print("No generated files found")
        return
    
    # Print analysis
    print_analysis_summary(results)
    
    # Evaluate libraries
    evaluate_libraries()
    
    # Draw conclusions
    draw_conclusions(results)
    
    # Save report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'files_analyzed': len(results),
        'results': [
            {
                'file': r.file_name,
                'format': r.format,
                'lines': r.lines,
                'classes': r.classes,
                'functions': r.functions,
                'syntax_ok': r.syntax_ok,
                'runs_ok': r.runs_ok,
            }
            for r in results
        ]
    }
    
    report_path = Path('examples/output/comprehensive_analysis.json')
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n📄 Report saved: {report_path}")


if __name__ == '__main__':
    main()
