#!/usr/bin/env python3
"""
Project Structure Benchmark.

Tests code2logic's ability to analyze and reproduce entire project structures.
Compares format efficiency across different languages and project types.

Usage:
    python examples/13_project_benchmark.py --project ~/github/wronai/contract/src
    python examples/13_project_benchmark.py --project tests/samples/ --formats yaml markdown
"""

import argparse
import json
import os
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
from code2logic.logicml import LogicMLGenerator
from code2logic.toon_format import TOONGenerator
from code2logic.reproduction import extract_code_block


FORMATS = ['yaml', 'toon', 'markdown', 'json', 'logicml']


@dataclass
class FileResult:
    """Result for single file."""
    path: str
    language: str
    original_size: int
    spec_size: int
    generated_size: int
    score: float
    syntax_ok: bool
    gen_time: float
    error: str = ""


@dataclass
class ProjectBenchmarkResult:
    """Result for entire project."""
    project_path: str
    format: str
    
    # Files
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    
    # Sizes
    total_original_size: int = 0
    total_spec_size: int = 0
    total_generated_size: int = 0
    
    # Metrics
    avg_score: float = 0.0
    syntax_ok_rate: float = 0.0
    
    # Languages
    languages: Dict[str, int] = field(default_factory=dict)
    
    # Time
    total_time: float = 0.0
    
    # Per-file results
    file_results: List[FileResult] = field(default_factory=list)


class JSONGenerator:
    """Simple JSON generator for comparison."""
    
    def generate(self, project, detail='full') -> str:
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
                'classes': [
                    {
                        'name': c.name,
                        'methods': [method.name for method in c.methods[:10]],
                        'properties': c.properties[:10],
                    }
                    for c in m.classes[:10]
                ],
                'functions': [
                    {
                        'name': f.name,
                        'params': f.params[:5],
                        'returns': f.return_type or '',
                    }
                    for f in m.functions[:15]
                ]
            }
            data['modules'].append(module)
        
        return json.dumps(data, indent=2)


def generate_spec(project, fmt: str) -> str:
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
    elif fmt == 'logicml':
        gen = LogicMLGenerator()
        spec = gen.generate(project)
        return spec.content
    return ""


def get_language_from_path(path: str) -> str:
    """Get language from file extension."""
    ext_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.jsx': 'javascript',
        '.go': 'go',
        '.rs': 'rust',
        '.java': 'java',
        '.rb': 'ruby',
        '.php': 'php',
    }
    ext = Path(path).suffix.lower()
    return ext_map.get(ext, 'unknown')


def reproduce_file(
    module_info,
    fmt: str,
    client,
    project_root: str,
    verbose: bool = False
) -> FileResult:
    """Reproduce a single file."""
    from code2logic.models import ProjectInfo
    
    # Build absolute path
    rel_path = module_info.path
    if not Path(rel_path).is_absolute():
        abs_path = Path(project_root) / rel_path
    else:
        abs_path = Path(rel_path)
    
    path = str(abs_path)
    language = module_info.language
    
    # Read original
    try:
        original = abs_path.read_text()
    except Exception as e:
        original = ""
        if verbose:
            print(f"   Cannot read {abs_path}: {e}")
    
    result = FileResult(
        path=path,
        language=language,
        original_size=len(original),
        spec_size=0,
        generated_size=0,
        score=0.0,
        syntax_ok=False,
        gen_time=0.0,
    )
    
    try:
        # Create single-file project
        single_project = ProjectInfo(
            name=Path(path).name,
            root_path=str(Path(path).parent),
            languages={language: 1},
            modules=[module_info],
            dependency_graph={},
            dependency_metrics={},
            entrypoints=[],
            similar_functions={},
            total_files=1,
            total_lines=module_info.lines_total,
            generated_at=datetime.now().isoformat(),
        )
        
        # Generate spec
        spec = generate_spec(single_project, fmt)
        result.spec_size = len(spec)
        
        # Reproduce
        prompt = f"""Generate {language} code from this {fmt.upper()} specification.

{spec[:5000]}

Requirements:
- Complete, working {language} code
- Include all imports
- Implement all functions and classes

```{language}
"""
        
        start = time.time()
        response = client.generate(prompt, max_tokens=4000)
        result.gen_time = time.time() - start
        
        generated = extract_code_block(response)
        result.generated_size = len(generated)
        
        # Test syntax (Python only for now)
        if language == 'python':
            try:
                compile(generated, '<string>', 'exec')
                result.syntax_ok = True
            except:
                result.syntax_ok = False
        else:
            # Basic check for other languages
            result.syntax_ok = len(generated) > 50
        
        # Calculate score
        if original and generated:
            metrics = ReproductionMetrics()
            analysis = metrics.analyze(original, generated, spec, format_name=fmt)
            result.score = analysis.overall_score
        
    except Exception as e:
        result.error = str(e)[:100]
        if verbose:
            print(f"   Error: {e}")
    
    return result


def run_project_benchmark(
    project_path: str,
    formats: List[str] = None,
    limit: int = None,
    workers: int = 3,
    verbose: bool = False,
) -> Dict[str, ProjectBenchmarkResult]:
    """Run benchmark on entire project."""
    
    if formats is None:
        formats = FORMATS
    
    print(f"\n{'='*80}")
    print("PROJECT STRUCTURE BENCHMARK")
    print(f"{'='*80}")
    print(f"Project: {project_path}")
    print(f"Formats: {', '.join(formats)}")
    
    # Initialize LLM
    try:
        client = get_client()
        print(f"LLM: {client.__class__.__name__}")
    except Exception as e:
        print(f"LLM not available: {e}")
        return {}
    
    # Analyze project
    print("\n📊 Analyzing project...")
    project = analyze_project(project_path, use_treesitter=False)
    
    print(f"   Files: {project.total_files}")
    print(f"   Lines: {project.total_lines}")
    print(f"   Languages: {project.languages}")
    
    # Filter modules
    modules = project.modules
    if limit:
        modules = modules[:limit]
    
    results = {}
    
    for fmt in formats:
        print(f"\n{'─'*80}")
        print(f"FORMAT: {fmt.upper()}")
        print(f"{'─'*80}")
        
        result = ProjectBenchmarkResult(
            project_path=project_path,
            format=fmt,
            total_files=len(modules),
        )
        
        # Process files
        for i, module in enumerate(modules):
            file_result = reproduce_file(module, fmt, client, project_path, verbose)
            result.file_results.append(file_result)
            
            # Update stats
            result.total_original_size += file_result.original_size
            result.total_spec_size += file_result.spec_size
            result.total_generated_size += file_result.generated_size
            result.total_time += file_result.gen_time
            
            if file_result.score > 0:
                result.successful_files += 1
            else:
                result.failed_files += 1
            
            # Track languages
            lang = file_result.language
            result.languages[lang] = result.languages.get(lang, 0) + 1
            
            # Print progress
            status = "✓" if file_result.score > 50 else "○"
            syntax = "S✓" if file_result.syntax_ok else "S✗"
            print(f"[{i+1}/{len(modules)}] {Path(file_result.path).name[:30]:<30} "
                  f"{file_result.score:>5.1f}% {syntax} ({file_result.gen_time:.1f}s)")
        
        # Calculate averages
        if result.file_results:
            scores = [r.score for r in result.file_results if r.score > 0]
            result.avg_score = sum(scores) / len(scores) if scores else 0
            result.syntax_ok_rate = sum(1 for r in result.file_results if r.syntax_ok) / len(result.file_results) * 100
        
        results[fmt] = result
    
    return results


def print_project_summary(results: Dict[str, ProjectBenchmarkResult]):
    """Print project benchmark summary."""
    
    print(f"\n{'='*80}")
    print("PROJECT BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    formats = list(results.keys())
    
    print(f"\n📊 Format Comparison:")
    print(f"{'─'*80}")
    print(f"{'Metric':<25}", end="")
    for fmt in formats:
        print(f"{fmt:>18}", end="")
    print()
    print(f"{'─'*80}")
    
    metrics = [
        ("Files Processed", lambda r: r.total_files, ""),
        ("Successful", lambda r: r.successful_files, ""),
        ("Avg Score", lambda r: r.avg_score, "%"),
        ("Syntax OK Rate", lambda r: r.syntax_ok_rate, "%"),
        ("Total Spec Size", lambda r: r.total_spec_size // 1024, "KB"),
        ("Total Gen Size", lambda r: r.total_generated_size // 1024, "KB"),
        ("Compression", lambda r: r.total_spec_size / r.total_original_size if r.total_original_size else 0, "x"),
        ("Total Time", lambda r: r.total_time, "s"),
    ]
    
    for name, calc, suffix in metrics:
        print(f"{name:<25}", end="")
        for fmt in formats:
            val = calc(results[fmt])
            print(f"{val:>16.1f}{suffix}", end="")
        print()
    
    print(f"{'─'*80}")
    
    # Language breakdown
    print(f"\n📁 Language Breakdown:")
    all_langs = set()
    for r in results.values():
        all_langs.update(r.languages.keys())
    
    print(f"{'─'*80}")
    print(f"{'Language':<25}", end="")
    for fmt in formats:
        print(f"{fmt:>18}", end="")
    print()
    print(f"{'─'*80}")
    
    for lang in sorted(all_langs):
        print(f"{lang:<25}", end="")
        for fmt in formats:
            # Calculate avg score for language
            lang_results = [r for r in results[fmt].file_results if r.language == lang]
            if lang_results:
                avg = sum(r.score for r in lang_results) / len(lang_results)
                print(f"{avg:>17.1f}%", end="")
            else:
                print(f"{'N/A':>18}", end="")
        print()
    
    print(f"{'─'*80}")
    
    # Best format
    print(f"\n🏆 Best Format:")
    best_score = max(formats, key=lambda f: results[f].avg_score)
    best_syntax = max(formats, key=lambda f: results[f].syntax_ok_rate)
    best_compression = min(formats, key=lambda f: results[f].total_spec_size / results[f].total_original_size if results[f].total_original_size else float('inf'))
    
    print(f"   Score: {best_score} ({results[best_score].avg_score:.1f}%)")
    print(f"   Syntax OK: {best_syntax} ({results[best_syntax].syntax_ok_rate:.1f}%)")
    print(f"   Compression: {best_compression}")


def save_project_report(results: Dict[str, ProjectBenchmarkResult], output: str):
    """Save project benchmark report."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'formats': {}
    }
    
    for fmt, result in results.items():
        data['formats'][fmt] = {
            'project_path': result.project_path,
            'total_files': result.total_files,
            'successful_files': result.successful_files,
            'avg_score': result.avg_score,
            'syntax_ok_rate': result.syntax_ok_rate,
            'total_spec_size': result.total_spec_size,
            'total_generated_size': result.total_generated_size,
            'total_time': result.total_time,
            'languages': result.languages,
            'files': [
                {
                    'path': r.path,
                    'language': r.language,
                    'score': r.score,
                    'syntax_ok': r.syntax_ok,
                    'gen_time': r.gen_time,
                }
                for r in result.file_results
            ]
        }
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n📄 Report saved: {output}")


def main():
    parser = argparse.ArgumentParser(description='Project Structure Benchmark')
    parser.add_argument('--project', '-p', required=True, help='Project path to benchmark')
    parser.add_argument('--formats', '-f', nargs='+', default=FORMATS)
    parser.add_argument('--limit', '-l', type=int, help='Limit files to process')
    parser.add_argument('--workers', '-w', type=int, default=3)
    parser.add_argument('--output', '-o', default='examples/output/project_benchmark.json')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    results = run_project_benchmark(
        args.project,
        formats=args.formats,
        limit=args.limit,
        workers=args.workers,
        verbose=args.verbose,
    )
    
    if results:
        print_project_summary(results)
        save_project_report(results, args.output)


if __name__ == '__main__':
    main()
