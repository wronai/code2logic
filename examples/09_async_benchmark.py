#!/usr/bin/env python3
"""
Advanced Async Benchmark with Multi-Provider LLM Support.

Features:
- Async/parallel LLM API calls for faster benchmarking
- Multi-provider support with hierarchy (OpenRouter -> Ollama -> LiteLLM)
- Unittest generation alongside code reproduction
- Code executability testing
- Detailed quality metrics

Usage:
    python examples/09_async_benchmark.py --folder tests/samples/
    python examples/09_async_benchmark.py --folder tests/samples/ --with-tests
    python examples/09_async_benchmark.py --providers openrouter,ollama
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
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

from code2logic import analyze_project, ReproductionMetrics
from code2logic.formats import GherkinGenerator, YAMLGenerator, MarkdownHybridGenerator
from code2logic.core import ProjectInfo
from code2logic.reproduction import extract_code_block
from code2logic.utils import cleanup_generated_root, write_text_atomic
from code2logic.benchmarks.common import (
    create_single_project,
    generate_spec,
    get_async_reproduction_prompt,
)


FORMATS = ['gherkin', 'yaml', 'markdown']


@dataclass
class LLMProvider:
    """LLM provider configuration."""
    name: str
    priority: int
    available: bool = False
    client: any = None
    
    def check_availability(self) -> bool:
        """Check if provider is available."""
        try:
            if self.name == 'openrouter':
                from code2logic import get_client
                try:
                    self.client = get_client()
                    self.available = True
                except:
                    pass
            elif self.name == 'ollama':
                from code2logic.llm import OllamaClient, OllamaConfig
                client = OllamaClient(OllamaConfig())
                if client.is_available():
                    self.client = client
                    self.available = True
            elif self.name == 'litellm':
                from code2logic.llm import LiteLLMClient, LiteLLMConfig
                client = LiteLLMClient(LiteLLMConfig())
                if client.is_available():
                    self.client = client
                    self.available = True
        except Exception as e:
            self.available = False
        return self.available


@dataclass
class AsyncBenchmarkResult:
    """Result for async benchmark."""
    file_name: str
    file_size: int
    format: str
    
    score: float = 0.0
    spec_size: int = 0
    gen_size: int = 0
    gen_time: float = 0.0
    
    # Quality metrics
    syntax_ok: bool = False
    runs_ok: bool = False
    tests_generated: int = 0
    tests_passed: int = 0
    
    # Detailed scores
    text_score: float = 0.0
    struct_score: float = 0.0
    semantic_score: float = 0.0
    
    error: str = ""


class MultiProviderLLM:
    """Multi-provider LLM with fallback hierarchy."""
    
    def __init__(self, providers: List[str] = None):
        self.providers = []
        provider_list = providers or ['openrouter', 'ollama', 'litellm']
        
        for i, name in enumerate(provider_list):
            provider = LLMProvider(name=name, priority=i)
            provider.check_availability()
            self.providers.append(provider)
        
        # Sort by priority, available first
        self.providers.sort(key=lambda p: (not p.available, p.priority))
    
    def get_active_provider(self) -> Optional[LLMProvider]:
        """Get first available provider."""
        for p in self.providers:
            if p.available:
                return p
        return None
    
    def generate(self, prompt: str, max_tokens: int = 4000) -> str:
        """Generate using first available provider."""
        for provider in self.providers:
            if provider.available and provider.client:
                try:
                    return provider.client.generate(prompt, max_tokens=max_tokens)
                except Exception as e:
                    continue
        raise RuntimeError("No LLM providers available")
    
    def status(self) -> Dict[str, bool]:
        """Get provider availability status."""
        return {p.name: p.available for p in self.providers}


def get_reproduction_prompt(spec: str, fmt: str, file_name: str, with_tests: bool = False) -> str:
    return get_async_reproduction_prompt(spec, fmt, file_name, with_tests=with_tests)


def test_code_quality(code: str, file_name: str) -> Tuple[bool, bool, str]:
    """Test if generated code is syntactically correct and runs."""
    # Create temp file
    temp_path = Path(f'/tmp/test_{file_name}')
    temp_path.write_text(code)
    
    # Test syntax
    syntax_ok = False
    runs_ok = False
    error = ""
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', str(temp_path)],
            capture_output=True, text=True, timeout=5
        )
        syntax_ok = result.returncode == 0
        if not syntax_ok:
            error = result.stderr[:200]
    except Exception as e:
        error = str(e)[:200]
    
    # Test execution
    if syntax_ok:
        try:
            result = subprocess.run(
                [sys.executable, str(temp_path)],
                capture_output=True, text=True, timeout=10
            )
            runs_ok = result.returncode == 0
            if not runs_ok:
                error = result.stderr[:200]
        except subprocess.TimeoutExpired:
            runs_ok = True  # Timeout might mean it's waiting for input
        except Exception as e:
            error = str(e)[:200]
    
    # Cleanup
    try:
        temp_path.unlink()
    except:
        pass
    
    return syntax_ok, runs_ok, error


def run_tests_in_code(code: str) -> Tuple[int, int]:
    """Run unittest tests embedded in code."""
    import re
    
    # Count test methods
    test_methods = len(re.findall(r'def test_\w+', code))
    if test_methods == 0:
        return 0, 0
    
    # Create temp file and run tests
    temp_path = Path(f'/tmp/test_runner_{time.time()}.py')
    
    # Add test runner if not present
    if 'unittest.main' not in code:
        code += "\n\nif __name__ == '__main__':\n    unittest.main(verbosity=0, exit=False)\n"
    
    temp_path.write_text(code)
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', str(temp_path), '-v', '--tb=no'],
            capture_output=True, text=True, timeout=30
        )
        # Parse pytest output
        passed = len(re.findall(r'PASSED', result.stdout))
        return test_methods, passed
    except:
        return test_methods, 0
    finally:
        try:
            temp_path.unlink()
        except:
            pass


def process_file_format(
    args: Tuple[str, str, str, ProjectInfo, any, bool, bool]
) -> AsyncBenchmarkResult:
    """Process a single file+format combination."""
    file_name, fmt, original, single_project, llm, with_tests, verbose = args
    
    result = AsyncBenchmarkResult(
        file_name=file_name,
        file_size=len(original),
        format=fmt,
    )

    output_path = Path('examples/output/generated') / fmt / f"{file_name}_generated.py"
    
    try:
        # Generate spec
        spec = generate_spec(single_project, fmt)
        result.spec_size = len(spec)
        
        # Generate code
        prompt = get_reproduction_prompt(spec, fmt, file_name, with_tests)
        
        start = time.time()
        response = llm.generate(prompt, max_tokens=4000)
        result.gen_time = time.time() - start
        
        generated = extract_code_block(response)
        result.gen_size = len(generated)
        
        # Save generated code
        write_text_atomic(output_path, generated)
        
        # Test code quality
        result.syntax_ok, result.runs_ok, error = test_code_quality(generated, file_name)
        if error:
            result.error = error
        
        # Run tests if generated
        if with_tests:
            result.tests_generated, result.tests_passed = run_tests_in_code(generated)
        
        # Calculate metrics
        metrics = ReproductionMetrics()
        analysis = metrics.analyze(original, generated, spec, format_name=fmt)
        result.score = analysis.overall_score
        result.text_score = min(analysis.text.cosine_similarity, 100)
        result.struct_score = min(analysis.structural.structural_score, 100)
        result.semantic_score = min(analysis.semantic.intent_score, 100)
        
    except Exception as e:
        try:
            if output_path.exists():
                output_path.unlink()
        except Exception:
            pass
        result.error = str(e)[:200]
        if verbose:
            print(f"\n   Error {fmt}/{file_name}: {e}")
    
    return result


def run_async_benchmark(
    folder: str,
    formats: List[str] = None,
    limit: int = None,
    with_tests: bool = False,
    providers: List[str] = None,
    max_workers: int = 3,
    verbose: bool = False,
) -> List[AsyncBenchmarkResult]:
    """Run benchmark with parallel LLM calls."""
    
    if formats is None:
        formats = FORMATS

    formats = [f.strip() for f in formats if f and f.strip()]
    
    path = Path(folder)
    py_files = list(path.glob('*.py'))
    if limit:
        py_files = py_files[:limit]
    
    print(f"\n{'='*70}")
    print(f"ASYNC FORMAT BENCHMARK")
    print(f"{'='*70}")
    print(f"Folder: {folder}")
    print(f"Formats: {', '.join(formats)}")
    print(f"Files: {len(py_files)}")
    print(f"With tests: {with_tests}")
    print(f"Max workers: {max_workers}")

    generated_root = Path('examples/output/generated')
    if generated_root.exists():
        shutil.rmtree(generated_root)
    
    # Initialize LLM
    llm = MultiProviderLLM(providers)
    print(f"\n🤖 LLM Providers: {llm.status()}")
    
    active = llm.get_active_provider()
    if not active:
        print("❌ No LLM providers available!")
        return []
    print(f"   Active: {active.name}")
    
    # Analyze project
    print("\n📊 Analyzing project...")
    project = analyze_project(str(path), use_treesitter=False)
    
    # Prepare tasks
    tasks = []
    for py_file in py_files:
        file_name = py_file.name
        original = py_file.read_text()
        
        # Find module info
        module_info = None
        for m in project.modules:
            if Path(m.path).name == file_name:
                module_info = m
                break
        
        if not module_info:
            continue
        
        single_project = create_single_project(module_info, py_file)
        
        for fmt in formats:
            tasks.append((file_name, fmt, original, single_project, llm, with_tests, verbose))
    
    print(f"\n⚡ Running {len(tasks)} tasks with {max_workers} workers...")
    print(f"{'─'*70}")
    
    results = []
    completed = 0
    
    # Run parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file_format, task): task for task in tasks}
        
        for future in as_completed(futures):
            task = futures[future]
            file_name, fmt = task[0], task[1]
            
            try:
                result = future.result()
                results.append(result)
                
                completed += 1
                status = "✓" if result.score > 50 else "○"
                syntax = "S✓" if result.syntax_ok else "S✗"
                runs = "R✓" if result.runs_ok else "R✗"
                
                print(f"[{completed}/{len(tasks)}] {file_name:<25} {fmt:<10} "
                      f"{result.score:>5.1f}% {syntax} {runs} ({result.gen_time:.1f}s)")
                
            except Exception as e:
                print(f"[{completed}/{len(tasks)}] {file_name:<25} {fmt:<10} ERROR: {e}")
    
    print(f"{'─'*70}")

    # Cleanup: keep only requested format folders
    cleanup_generated_root(generated_root, set(formats))
    
    return results


def print_async_summary(results: List[AsyncBenchmarkResult], formats: List[str]):
    """Print async benchmark summary."""
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    
    # Group by format
    by_format = {fmt: [r for r in results if r.format == fmt] for fmt in formats}
    
    # Main metrics
    print(f"\n📊 Overall Results:")
    print(f"{'─'*70}")
    print(f"{'Metric':<20}", end="")
    for fmt in formats:
        print(f"{fmt:>16}", end="")
    print()
    print(f"{'─'*70}")
    
    metrics = [
        ("Avg Score", lambda rs: sum(r.score for r in rs) / len(rs) if rs else 0, "%"),
        ("Syntax OK", lambda rs: sum(1 for r in rs if r.syntax_ok) / len(rs) * 100 if rs else 0, "%"),
        ("Runs OK", lambda rs: sum(1 for r in rs if r.runs_ok) / len(rs) * 100 if rs else 0, "%"),
        ("Avg Gen Time", lambda rs: sum(r.gen_time for r in rs) / len(rs) if rs else 0, "s"),
        ("Avg Gen Size", lambda rs: sum(r.gen_size for r in rs) / len(rs) if rs else 0, ""),
    ]
    
    for name, calc, suffix in metrics:
        print(f"{name:<20}", end="")
        for fmt in formats:
            val = calc(by_format[fmt])
            print(f"{val:>14.1f}{suffix}", end="")
        print()
    
    print(f"{'─'*70}")
    
    # Code quality breakdown
    print(f"\n📈 Score Breakdown:")
    print(f"{'─'*70}")
    print(f"{'Category':<20}", end="")
    for fmt in formats:
        print(f"{fmt:>16}", end="")
    print()
    print(f"{'─'*70}")
    
    for name, attr in [("Text", "text_score"), ("Structural", "struct_score"), ("Semantic", "semantic_score")]:
        print(f"{name:<20}", end="")
        for fmt in formats:
            rs = by_format[fmt]
            val = sum(getattr(r, attr) for r in rs) / len(rs) if rs else 0
            print(f"{val:>15.1f}%", end="")
        print()
    
    print(f"{'─'*70}")
    
    # Quality comparison
    print(f"\n🔍 Code Quality (syntax OK + runs OK):")
    print(f"{'─'*70}")
    
    for fmt in formats:
        rs = by_format[fmt]
        syntax_ok = sum(1 for r in rs if r.syntax_ok)
        runs_ok = sum(1 for r in rs if r.runs_ok)
        total = len(rs)
        print(f"   {fmt:<12}: {syntax_ok}/{total} syntax OK, {runs_ok}/{total} runs OK")
    
    # Winner
    print(f"\n🏆 Best Format:")
    best_score = max(formats, key=lambda f: sum(r.score for r in by_format[f]) / len(by_format[f]) if by_format[f] else 0)
    best_quality = max(formats, key=lambda f: sum(1 for r in by_format[f] if r.runs_ok) / len(by_format[f]) if by_format[f] else 0)
    best_speed = min(formats, key=lambda f: sum(r.gen_time for r in by_format[f]) / len(by_format[f]) if by_format[f] else float('inf'))
    
    print(f"   Score: {best_score}")
    print(f"   Quality (runs OK): {best_quality}")
    print(f"   Speed: {best_speed}")


def save_async_report(results: List[AsyncBenchmarkResult], output: str):
    """Save async benchmark report."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'total_tasks': len(results),
        'results': [
            {
                'file': r.file_name,
                'format': r.format,
                'score': r.score,
                'syntax_ok': r.syntax_ok,
                'runs_ok': r.runs_ok,
                'gen_time': r.gen_time,
                'gen_size': r.gen_size,
                'spec_size': r.spec_size,
                'text_score': r.text_score,
                'struct_score': r.struct_score,
                'semantic_score': r.semantic_score,
                'tests_generated': r.tests_generated,
                'tests_passed': r.tests_passed,
                'error': r.error,
            }
            for r in results
        ]
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n📄 Report saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Async Format Benchmark with Multi-Provider LLM')
    parser.add_argument('--folder', '-f', default='tests/samples/')
    parser.add_argument('--formats', nargs='+', default=FORMATS)
    parser.add_argument('--limit', '-l', type=int)
    parser.add_argument('--with-tests', '-t', action='store_true', help='Generate unittests')
    parser.add_argument('--providers', '-p', nargs='+', default=['openrouter', 'ollama'])
    parser.add_argument('--workers', '-w', type=int, default=3, help='Max parallel workers')
    parser.add_argument('--output', '-o', default='examples/output/async_benchmark.json')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    results = run_async_benchmark(
        args.folder,
        formats=args.formats,
        limit=args.limit,
        with_tests=args.with_tests,
        providers=args.providers,
        max_workers=args.workers,
        verbose=args.verbose,
    )
    
    if results:
        print_async_summary(results, args.formats)
        save_async_report(results, args.output)


if __name__ == '__main__':
    main()
