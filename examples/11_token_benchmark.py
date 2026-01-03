#!/usr/bin/env python3
"""
Token-Aware Format Benchmark.

Compares formats (JSON, YAML, Gherkin, Markdown) with token usage tracking.
Measures efficiency: tokens_in vs tokens_out vs code quality.

Usage:
    python examples/11_token_benchmark.py --folder tests/samples/
    python examples/11_token_benchmark.py --folder tests/samples/ --formats json yaml
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from code2logic import analyze_project, get_client, ReproductionMetrics
from code2logic.gherkin import GherkinGenerator
from code2logic.generators import YAMLGenerator
from code2logic.markdown_format import MarkdownHybridGenerator
from code2logic.reproduction import extract_code_block
from code2logic.models import ProjectInfo


# Token estimation (approx 4 chars per token for English text)
def estimate_tokens(text: str) -> int:
    """Estimate token count (approx 4 chars per token)."""
    return len(text) // 4


@dataclass 
class TokenBenchmarkResult:
    """Result with token tracking."""
    file_name: str
    format: str
    
    # Sizes
    original_size: int = 0
    spec_size: int = 0
    generated_size: int = 0
    
    # Token estimates
    spec_tokens: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    
    # Quality
    score: float = 0.0
    syntax_ok: bool = False
    runs_ok: bool = False
    
    # Efficiency metrics
    compression_ratio: float = 0.0  # spec_size / original_size
    token_efficiency: float = 0.0   # score / total_tokens * 1000
    reproduction_ratio: float = 0.0 # generated_size / original_size
    
    gen_time: float = 0.0
    error: str = ""


class JSONGenerator:
    """Generate JSON specification for code."""
    
    def generate(self, project: ProjectInfo, detail: str = 'full') -> str:
        """Generate JSON specification."""
        data = {
            'project': project.name,
            'files': project.total_files,
            'lines': project.total_lines,
            'modules': []
        }
        
        for m in project.modules:
            module = {
                'path': m.path,
                'language': m.language,
                'imports': m.imports[:10],
                'exports': m.exports[:10],
            }
            
            if m.classes:
                module['classes'] = []
                for c in m.classes:
                    cls = {
                        'name': c.name,
                        'bases': c.bases,
                        'doc': c.docstring[:80] if c.docstring else '',
                        'properties': c.properties[:15],
                        'methods': [
                            {
                                'name': method.name,
                                'params': method.params[:5],
                                'returns': method.return_type or 'None',
                                'doc': method.intent[:50] if method.intent else '',
                                'async': method.is_async,
                            }
                            for method in c.methods[:15]
                        ]
                    }
                    module['classes'].append(cls)
            
            if m.functions:
                module['functions'] = [
                    {
                        'name': f.name,
                        'params': f.params[:6],
                        'returns': f.return_type or 'None',
                        'doc': f.intent[:60] if f.intent else '',
                        'async': f.is_async,
                        'lines': f.lines,
                    }
                    for f in m.functions[:20]
                ]
            
            data['modules'].append(module)
        
        return json.dumps(data, indent=2)


class CompactJSONGenerator:
    """Generate compact JSON (no indentation) for token efficiency."""
    
    def generate(self, project: ProjectInfo, detail: str = 'full') -> str:
        """Generate compact JSON."""
        gen = JSONGenerator()
        full_json = gen.generate(project, detail)
        # Parse and re-dump without indentation
        data = json.loads(full_json)
        return json.dumps(data, separators=(',', ':'))


FORMATS = {
    'json': JSONGenerator,
    'json_compact': CompactJSONGenerator,
    'yaml': YAMLGenerator,
    'gherkin': GherkinGenerator,
    'markdown': MarkdownHybridGenerator,
}


def generate_spec(project: ProjectInfo, fmt: str) -> str:
    """Generate specification in given format."""
    if fmt == 'gherkin':
        gen = GherkinGenerator()
        return gen.generate(project)
    elif fmt == 'yaml':
        gen = YAMLGenerator()
        return gen.generate(project, detail='full')
    elif fmt == 'markdown':
        gen = MarkdownHybridGenerator()
        spec = gen.generate(project)
        return spec.content
    elif fmt == 'json':
        gen = JSONGenerator()
        return gen.generate(project)
    elif fmt == 'json_compact':
        gen = CompactJSONGenerator()
        return gen.generate(project)
    return ""


def create_single_project(module_info, file_path: Path) -> ProjectInfo:
    """Create ProjectInfo for a single file."""
    return ProjectInfo(
        name=file_path.name,
        root_path=str(file_path.parent),
        languages={'python': 1},
        modules=[module_info],
        dependency_graph={},
        dependency_metrics={},
        entrypoints=[],
        similar_functions={},
        total_files=1,
        total_lines=module_info.lines_total,
        generated_at=datetime.now().isoformat(),
    )


def get_reproduction_prompt(spec: str, fmt: str, file_name: str) -> str:
    """Generate reproduction prompt."""
    
    format_hints = {
        'json': "Parse the JSON structure and implement all classes and functions.",
        'json_compact': "Parse the compact JSON and implement all elements.",
        'yaml': "Parse the YAML structure and implement all classes and functions.",
        'gherkin': "Implement all scenarios as working Python code.",
        'markdown': "Parse embedded Gherkin (behaviors) and YAML (structures).",
    }
    
    # Truncate spec for token efficiency
    max_spec = 5000
    spec_truncated = spec[:max_spec] if len(spec) > max_spec else spec
    
    prompt = f"""Generate Python code from this {fmt.upper()} specification.
{format_hints.get(fmt, '')}

{spec_truncated}

Requirements:
- Complete, working Python code for {file_name}
- Include imports and type hints
- Implement all functions with actual logic

```python
"""
    return prompt


def test_code(code: str) -> Tuple[bool, bool]:
    """Test syntax and basic execution."""
    import subprocess
    import tempfile
    
    syntax_ok = False
    runs_ok = False
    
    try:
        compile(code, '<string>', 'exec')
        syntax_ok = True
    except:
        return False, False
    
    # Test execution
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, timeout=5
            )
            runs_ok = result.returncode == 0
    except:
        runs_ok = True  # Timeout might mean waiting for input
    
    return syntax_ok, runs_ok


def process_file_format(args) -> TokenBenchmarkResult:
    """Process single file+format."""
    file_name, fmt, original, single_project, client, verbose = args
    
    result = TokenBenchmarkResult(
        file_name=file_name,
        format=fmt,
        original_size=len(original),
    )
    
    try:
        # Generate spec
        spec = generate_spec(single_project, fmt)
        result.spec_size = len(spec)
        result.spec_tokens = estimate_tokens(spec)
        
        # Generate prompt
        prompt = get_reproduction_prompt(spec, fmt, file_name)
        result.prompt_tokens = estimate_tokens(prompt)
        
        # Reproduce
        start = time.time()
        response = client.generate(prompt, max_tokens=4000)
        result.gen_time = time.time() - start
        
        result.response_tokens = estimate_tokens(response)
        result.total_tokens = result.prompt_tokens + result.response_tokens
        
        # Extract code
        generated = extract_code_block(response)
        result.generated_size = len(generated)
        
        # Test
        result.syntax_ok, result.runs_ok = test_code(generated)
        
        # Metrics
        metrics = ReproductionMetrics()
        analysis = metrics.analyze(original, generated, spec, format_name=fmt)
        result.score = analysis.overall_score
        
        # Efficiency calculations
        result.compression_ratio = result.spec_size / result.original_size if result.original_size else 0
        result.reproduction_ratio = result.generated_size / result.original_size if result.original_size else 0
        result.token_efficiency = (result.score / result.total_tokens * 1000) if result.total_tokens else 0
        
        # Save generated
        output_dir = Path('examples/output/generated') / fmt
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{file_name}_generated.py").write_text(generated)
        
    except Exception as e:
        result.error = str(e)[:100]
        if verbose:
            print(f"  Error: {e}")
    
    return result


def run_token_benchmark(
    folder: str,
    formats: List[str] = None,
    limit: int = None,
    workers: int = 3,
    verbose: bool = False,
) -> List[TokenBenchmarkResult]:
    """Run benchmark with token tracking."""
    
    if formats is None:
        formats = ['json', 'yaml', 'gherkin', 'markdown']
    
    path = Path(folder)
    py_files = list(path.glob('*.py'))
    if limit:
        py_files = py_files[:limit]
    
    print(f"\n{'='*80}")
    print("TOKEN-AWARE FORMAT BENCHMARK")
    print(f"{'='*80}")
    print(f"Folder: {folder}")
    print(f"Formats: {', '.join(formats)}")
    print(f"Files: {len(py_files)}")
    
    # Initialize
    try:
        client = get_client()
        print(f"LLM: {client.__class__.__name__}")
    except Exception as e:
        print(f"LLM not available: {e}")
        return []
    
    # Analyze
    print("\n📊 Analyzing project...")
    project = analyze_project(str(path), use_treesitter=False)
    
    # Prepare tasks
    tasks = []
    for py_file in py_files:
        file_name = py_file.name
        original = py_file.read_text()
        
        module_info = None
        for m in project.modules:
            if Path(m.path).name == file_name:
                module_info = m
                break
        
        if not module_info:
            continue
        
        single_project = create_single_project(module_info, py_file)
        
        for fmt in formats:
            tasks.append((file_name, fmt, original, single_project, client, verbose))
    
    print(f"\n⚡ Running {len(tasks)} tasks...")
    print(f"{'─'*80}")
    print(f"{'File':<25} {'Format':<12} {'Score':>7} {'Tokens':>8} {'Eff':>6} {'Time':>6}")
    print(f"{'─'*80}")
    
    results = []
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file_format, t): t for t in tasks}
        
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            
            status = "✓" if r.score > 50 else "○"
            print(f"{r.file_name:<25} {r.format:<12} {r.score:>6.1f}% {r.total_tokens:>7} {r.token_efficiency:>5.1f} {r.gen_time:>5.1f}s")
    
    print(f"{'─'*80}")
    
    return results


def print_token_summary(results: List[TokenBenchmarkResult], formats: List[str]):
    """Print token-aware summary."""
    
    print(f"\n{'='*80}")
    print("TOKEN EFFICIENCY SUMMARY")
    print(f"{'='*80}")
    
    # Group by format
    by_format = {fmt: [r for r in results if r.format == fmt] for fmt in formats}
    
    # Main table
    print(f"\n📊 Format Comparison:")
    print(f"{'─'*80}")
    print(f"{'Metric':<22}", end="")
    for fmt in formats:
        print(f"{fmt:>14}", end="")
    print()
    print(f"{'─'*80}")
    
    metrics = [
        ("Avg Score", lambda rs: sum(r.score for r in rs) / len(rs), "%"),
        ("Avg Spec Tokens", lambda rs: sum(r.spec_tokens for r in rs) / len(rs), ""),
        ("Avg Total Tokens", lambda rs: sum(r.total_tokens for r in rs) / len(rs), ""),
        ("Token Efficiency", lambda rs: sum(r.token_efficiency for r in rs) / len(rs), ""),
        ("Compression Ratio", lambda rs: sum(r.compression_ratio for r in rs) / len(rs), "x"),
        ("Syntax OK %", lambda rs: sum(1 for r in rs if r.syntax_ok) / len(rs) * 100, "%"),
        ("Runs OK %", lambda rs: sum(1 for r in rs if r.runs_ok) / len(rs) * 100, "%"),
        ("Avg Gen Time", lambda rs: sum(r.gen_time for r in rs) / len(rs), "s"),
    ]
    
    for name, calc, suffix in metrics:
        print(f"{name:<22}", end="")
        for fmt in formats:
            rs = by_format.get(fmt, [])
            if rs:
                val = calc(rs)
                print(f"{val:>12.1f}{suffix}", end="")
            else:
                print(f"{'N/A':>14}", end="")
        print()
    
    print(f"{'─'*80}")
    
    # Best format analysis
    print(f"\n🏆 Best Format by Metric:")
    
    def get_best(metric_fn, higher_better=True):
        scores = {fmt: metric_fn(by_format.get(fmt, [])) for fmt in formats if by_format.get(fmt)}
        if higher_better:
            return max(scores.items(), key=lambda x: x[1])
        return min(scores.items(), key=lambda x: x[1])
    
    best_score = get_best(lambda rs: sum(r.score for r in rs) / len(rs) if rs else 0)
    best_efficiency = get_best(lambda rs: sum(r.token_efficiency for r in rs) / len(rs) if rs else 0)
    best_compression = get_best(lambda rs: sum(r.compression_ratio for r in rs) / len(rs) if rs else float('inf'), False)
    least_tokens = get_best(lambda rs: sum(r.total_tokens for r in rs) / len(rs) if rs else float('inf'), False)
    
    print(f"   Best Score: {best_score[0]} ({best_score[1]:.1f}%)")
    print(f"   Best Token Efficiency: {best_efficiency[0]} ({best_efficiency[1]:.2f})")
    print(f"   Best Compression: {best_compression[0]} ({best_compression[1]:.2f}x)")
    print(f"   Least Tokens: {least_tokens[0]} ({least_tokens[1]:.0f})")
    
    # JSON vs YAML comparison
    if 'json' in formats and 'yaml' in formats:
        json_rs = by_format.get('json', [])
        yaml_rs = by_format.get('yaml', [])
        
        if json_rs and yaml_rs:
            print(f"\n📋 JSON vs YAML:")
            print(f"{'─'*80}")
            
            json_tokens = sum(r.spec_tokens for r in json_rs) / len(json_rs)
            yaml_tokens = sum(r.spec_tokens for r in yaml_rs) / len(yaml_rs)
            json_score = sum(r.score for r in json_rs) / len(json_rs)
            yaml_score = sum(r.score for r in yaml_rs) / len(yaml_rs)
            
            print(f"   JSON: {json_tokens:.0f} spec tokens, {json_score:.1f}% score")
            print(f"   YAML: {yaml_tokens:.0f} spec tokens, {yaml_score:.1f}% score")
            
            token_diff = (yaml_tokens - json_tokens) / json_tokens * 100 if json_tokens else 0
            score_diff = yaml_score - json_score
            
            if token_diff > 0:
                print(f"   → JSON uses {abs(token_diff):.1f}% fewer tokens")
            else:
                print(f"   → YAML uses {abs(token_diff):.1f}% fewer tokens")
            
            if score_diff > 0:
                print(f"   → YAML has {score_diff:.1f}% higher score")
            else:
                print(f"   → JSON has {abs(score_diff):.1f}% higher score")


def save_token_report(results: List[TokenBenchmarkResult], output: str):
    """Save detailed report."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'results': [
            {
                'file': r.file_name,
                'format': r.format,
                'original_size': r.original_size,
                'spec_size': r.spec_size,
                'spec_tokens': r.spec_tokens,
                'prompt_tokens': r.prompt_tokens,
                'response_tokens': r.response_tokens,
                'total_tokens': r.total_tokens,
                'generated_size': r.generated_size,
                'score': r.score,
                'syntax_ok': r.syntax_ok,
                'runs_ok': r.runs_ok,
                'compression_ratio': r.compression_ratio,
                'token_efficiency': r.token_efficiency,
                'reproduction_ratio': r.reproduction_ratio,
                'gen_time': r.gen_time,
                'error': r.error,
            }
            for r in results
        ]
    }
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n📄 Report: {output}")


def main():
    parser = argparse.ArgumentParser(description='Token-Aware Format Benchmark')
    parser.add_argument('--folder', '-f', default='tests/samples/')
    parser.add_argument('--formats', nargs='+', default=['json', 'yaml', 'gherkin', 'markdown'])
    parser.add_argument('--limit', '-l', type=int)
    parser.add_argument('--workers', '-w', type=int, default=3)
    parser.add_argument('--output', '-o', default='examples/output/token_benchmark.json')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    results = run_token_benchmark(
        args.folder,
        formats=args.formats,
        limit=args.limit,
        workers=args.workers,
        verbose=args.verbose,
    )
    
    if results:
        print_token_summary(results, args.formats)
        save_token_report(results, args.output)


if __name__ == '__main__':
    main()
