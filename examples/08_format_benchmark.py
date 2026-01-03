#!/usr/bin/env python3
"""
Format Benchmark: Markdown vs Gherkin vs YAML

Comprehensive benchmark comparing reproduction quality across formats.
Measures:
- Reproduction accuracy (structural, semantic, text)
- Token efficiency (spec size vs original)
- Generation time
- Success rate by file type

Usage:
    python examples/08_format_benchmark.py
    python examples/08_format_benchmark.py --folder code2logic/
    python examples/08_format_benchmark.py --samples tests/samples/
    python examples/08_format_benchmark.py --no-llm --quick
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from code2logic import (
    analyze_project,
    ReproductionMetrics,
    get_client,
)
from code2logic.gherkin import GherkinGenerator
from code2logic.generators import YAMLGenerator, JSONGenerator
from code2logic.markdown_format import MarkdownHybridGenerator
from code2logic.reproduction import extract_code_block
from code2logic.models import ProjectInfo


@dataclass
class BenchmarkResult:
    """Result for a single file benchmark."""
    file_name: str
    file_size: int
    language: str
    
    # Per-format metrics
    scores: Dict[str, float] = field(default_factory=dict)
    spec_sizes: Dict[str, int] = field(default_factory=dict)
    gen_sizes: Dict[str, int] = field(default_factory=dict)
    gen_times: Dict[str, float] = field(default_factory=dict)
    
    # Detailed metrics
    text_scores: Dict[str, float] = field(default_factory=dict)
    struct_scores: Dict[str, float] = field(default_factory=dict)
    semantic_scores: Dict[str, float] = field(default_factory=dict)
    
    # Code detail metrics
    orig_classes: int = 0
    orig_functions: int = 0
    orig_imports: int = 0
    gen_classes: Dict[str, int] = field(default_factory=dict)
    gen_functions: Dict[str, int] = field(default_factory=dict)
    gen_imports: Dict[str, int] = field(default_factory=dict)
    
    errors: Dict[str, str] = field(default_factory=dict)


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    total_files: int
    formats: List[str]
    
    # Aggregate scores
    avg_scores: Dict[str, float] = field(default_factory=dict)
    max_scores: Dict[str, float] = field(default_factory=dict)
    min_scores: Dict[str, float] = field(default_factory=dict)
    
    # Success rates (score > 50%)
    success_rates: Dict[str, float] = field(default_factory=dict)
    
    # Efficiency
    avg_compression: Dict[str, float] = field(default_factory=dict)
    avg_gen_time: Dict[str, float] = field(default_factory=dict)
    
    # Category scores
    avg_text: Dict[str, float] = field(default_factory=dict)
    avg_struct: Dict[str, float] = field(default_factory=dict)
    avg_semantic: Dict[str, float] = field(default_factory=dict)
    
    # Winner counts
    wins: Dict[str, int] = field(default_factory=dict)


FORMATS = ['gherkin', 'yaml', 'markdown']


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
        return gen.generate(project, detail='full')
    return ""


def reproduce_code(spec: str, fmt: str, file_name: str, client) -> Tuple[str, float]:
    """Reproduce code from spec using LLM. Returns (code, time)."""
    
    prompts = {
        'gherkin': f"""Generate Python code from this Gherkin/BDD specification.
Implement all scenarios as working code.

{spec[:5000]}

Generate complete Python code for {file_name}:""",
        
        'yaml': f"""Generate Python code from this YAML specification.
Match the structure exactly.

{spec[:5000]}

Generate complete Python code for {file_name}:""",
        
        'markdown': f"""Generate Python code from this Markdown specification.
It contains embedded Gherkin and YAML sections.

{spec[:5000]}

Generate complete Python code for {file_name}:""",
    }
    
    prompt = prompts.get(fmt, prompts['yaml'])
    
    start = time.time()
    try:
        response = client.generate(prompt, max_tokens=4000)
        code = extract_code_block(response)
    except Exception as e:
        code = f"# Error: {e}"
    elapsed = time.time() - start
    
    return code, elapsed


def generate_template(spec: str, fmt: str) -> str:
    """Fallback template generation."""
    import re
    
    classes = list(set(re.findall(r'class (\w+)', spec)))[:5]
    functions = list(set(re.findall(r'(?:Scenario|def|function):\s*(\w+)', spec, re.IGNORECASE)))[:10]
    
    code = '''from dataclasses import dataclass
from typing import Optional, List, Dict

'''
    
    for cls in classes:
        code += f'''@dataclass
class {cls}:
    """Generated class."""
    pass

'''
    
    for func in functions:
        if func not in classes:
            code += f'''def {func}():
    """Generated function."""
    pass

'''
    
    return code


def count_code_elements(code: str) -> Dict[str, int]:
    """Count code elements in generated code."""
    import re
    
    classes = len(re.findall(r'^class \w+', code, re.MULTILINE))
    functions = len(re.findall(r'^def \w+', code, re.MULTILINE))
    async_funcs = len(re.findall(r'^async def \w+', code, re.MULTILINE))
    imports = len(re.findall(r'^(?:from|import) ', code, re.MULTILINE))
    decorators = len(re.findall(r'^@\w+', code, re.MULTILINE))
    docstrings = len(re.findall(r'"""[^"]+"""', code))
    type_hints = len(re.findall(r': \w+[\[\],\s\w]*(?:=|$|\))', code))
    
    return {
        'classes': classes,
        'functions': functions + async_funcs,
        'imports': imports,
        'decorators': decorators,
        'docstrings': docstrings,
        'type_hints': type_hints,
    }


def save_generated_code(output_dir: Path, file_name: str, fmt: str, spec: str, generated: str):
    """Save generated spec and code for inspection."""
    fmt_dir = output_dir / 'generated' / fmt
    fmt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save spec
    spec_ext = {'gherkin': '.feature', 'yaml': '.yaml', 'markdown': '.md', 'json': '.json'}
    spec_path = fmt_dir / f"{file_name}{spec_ext.get(fmt, '.txt')}"
    spec_path.write_text(spec)
    
    # Save generated code
    code_path = fmt_dir / f"{file_name}_generated.py"
    code_path.write_text(generated)


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


def run_benchmark(
    folder: str,
    formats: List[str] = None,
    limit: int = None,
    no_llm: bool = False,
    verbose: bool = False,
) -> Tuple[List[BenchmarkResult], BenchmarkSummary]:
    """Run benchmark on folder."""
    
    if formats is None:
        formats = FORMATS
    
    path = Path(folder)
    py_files = list(path.glob('*.py'))
    if limit:
        py_files = py_files[:limit]
    
    print(f"\n{'='*70}")
    print(f"FORMAT BENCHMARK: {folder}")
    print(f"{'='*70}")
    print(f"Formats: {', '.join(formats)}")
    print(f"Files: {len(py_files)}")
    
    # Analyze project (use_treesitter=False for better dataclass detection)
    print("\n📊 Analyzing project...")
    project = analyze_project(str(path), use_treesitter=False)
    
    # Initialize
    client = None
    if not no_llm:
        try:
            client = get_client()
            print(f"🤖 Using LLM: {client.__class__.__name__}")
        except Exception as e:
            print(f"⚠️  LLM not available: {e}")
            no_llm = True
    
    if no_llm:
        print("📝 Using template generation (--no-llm)")
    
    results = []
    metrics = ReproductionMetrics(verbose=verbose)
    
    print(f"\n{'─'*70}")
    print(f"{'File':<25} {'Size':>8} ", end="")
    for fmt in formats:
        print(f"{fmt:>12}", end="")
    print(f" {'Winner':>12}")
    print(f"{'─'*70}")
    
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
        
        result = BenchmarkResult(
            file_name=file_name,
            file_size=len(original),
            language='python',
        )
        
        # Count original code elements
        orig_elements = count_code_elements(original)
        result.orig_classes = orig_elements['classes']
        result.orig_functions = orig_elements['functions']
        result.orig_imports = orig_elements['imports']
        
        # Create single-file project
        single_project = create_single_project(module_info, py_file)
        
        print(f"{file_name:<25} {len(original):>7} ", end="", flush=True)
        
        best_score = 0
        best_fmt = None
        
        for fmt in formats:
            # Generate spec
            spec = generate_spec(single_project, fmt)
            result.spec_sizes[fmt] = len(spec)
            
            # Reproduce
            if no_llm:
                generated = generate_template(spec, fmt)
                gen_time = 0.0
            else:
                generated, gen_time = reproduce_code(spec, fmt, file_name, client)
            
            result.gen_sizes[fmt] = len(generated)
            result.gen_times[fmt] = gen_time
            
            # Count code elements
            gen_elements = count_code_elements(generated)
            result.gen_classes[fmt] = gen_elements['classes']
            result.gen_functions[fmt] = gen_elements['functions']
            result.gen_imports[fmt] = gen_elements['imports']
            
            # Save generated files for inspection
            output_dir = Path('examples/output')
            save_generated_code(output_dir, file_name, fmt, spec, generated)
            
            # Calculate metrics
            try:
                analysis = metrics.analyze(
                    original, generated, spec,
                    format_name=fmt,
                    source_file=file_name,
                )
                result.scores[fmt] = analysis.overall_score
                result.text_scores[fmt] = min(analysis.text.cosine_similarity, 100)
                result.struct_scores[fmt] = min(analysis.structural.structural_score, 100)
                result.semantic_scores[fmt] = min(analysis.semantic.intent_score, 100)
                
                if analysis.overall_score > best_score:
                    best_score = analysis.overall_score
                    best_fmt = fmt
                    
            except Exception as e:
                if verbose:
                    print(f"\n   Error: {e}")
                result.errors[fmt] = str(e)
                result.scores[fmt] = 0.0
            
            # Print score
            score = result.scores.get(fmt, 0)
            marker = "✓" if score > 50 else " "
            print(f"{score:>10.1f}%{marker}", end="", flush=True)
        
        print(f" {best_fmt or 'none':>12}")
        results.append(result)
    
    print(f"{'─'*70}")
    
    # Calculate summary
    summary = calculate_summary(results, formats)
    
    return results, summary


def calculate_summary(results: List[BenchmarkResult], formats: List[str]) -> BenchmarkSummary:
    """Calculate benchmark summary."""
    summary = BenchmarkSummary(
        total_files=len(results),
        formats=formats,
    )
    
    for fmt in formats:
        scores = [r.scores.get(fmt, 0) for r in results if fmt in r.scores]
        spec_sizes = [r.spec_sizes.get(fmt, 0) for r in results]
        file_sizes = [r.file_size for r in results]
        gen_times = [r.gen_times.get(fmt, 0) for r in results]
        text_scores = [r.text_scores.get(fmt, 0) for r in results if fmt in r.text_scores]
        struct_scores = [r.struct_scores.get(fmt, 0) for r in results if fmt in r.struct_scores]
        semantic_scores = [r.semantic_scores.get(fmt, 0) for r in results if fmt in r.semantic_scores]
        
        if scores:
            summary.avg_scores[fmt] = sum(scores) / len(scores)
            summary.max_scores[fmt] = max(scores)
            summary.min_scores[fmt] = min(scores)
            summary.success_rates[fmt] = len([s for s in scores if s > 50]) / len(scores) * 100
        
        if spec_sizes and file_sizes:
            compressions = [s / f for s, f in zip(spec_sizes, file_sizes) if f > 0]
            summary.avg_compression[fmt] = sum(compressions) / len(compressions) if compressions else 0
        
        if gen_times:
            summary.avg_gen_time[fmt] = sum(gen_times) / len(gen_times)
        
        if text_scores:
            summary.avg_text[fmt] = sum(text_scores) / len(text_scores)
        if struct_scores:
            summary.avg_struct[fmt] = sum(struct_scores) / len(struct_scores)
        if semantic_scores:
            summary.avg_semantic[fmt] = sum(semantic_scores) / len(semantic_scores)
        
        # Count wins
        summary.wins[fmt] = 0
    
    # Calculate wins
    for r in results:
        if r.scores:
            best_fmt = max(r.scores.items(), key=lambda x: x[1])[0]
            summary.wins[best_fmt] = summary.wins.get(best_fmt, 0) + 1
    
    return summary


def print_summary(summary: BenchmarkSummary, results: List[BenchmarkResult] = None):
    """Print benchmark summary."""
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    
    # Main table
    print(f"\n📊 Overall Results ({summary.total_files} files):")
    print(f"{'─'*70}")
    print(f"{'Metric':<20}", end="")
    for fmt in summary.formats:
        print(f"{fmt:>16}", end="")
    print()
    print(f"{'─'*70}")
    
    rows = [
        ("Avg Score", summary.avg_scores, "%", 1),
        ("Max Score", summary.max_scores, "%", 1),
        ("Min Score", summary.min_scores, "%", 1),
        ("Success Rate", summary.success_rates, "%", 0),
        ("Wins", summary.wins, "", 0),
        ("Avg Gen Time", summary.avg_gen_time, "s", 2),
    ]
    
    for label, data, suffix, decimals in rows:
        print(f"{label:<20}", end="")
        for fmt in summary.formats:
            val = data.get(fmt, 0)
            if decimals == 0:
                print(f"{val:>15.0f}{suffix}", end="")
            elif decimals == 1:
                print(f"{val:>14.1f}{suffix}", end="")
            else:
                print(f"{val:>14.2f}{suffix}", end="")
        print()
    
    print(f"{'─'*70}")
    
    # Compression section with best indicator
    print(f"\n📦 Compression (lower = better):")
    print(f"{'─'*70}")
    best_compression = min(summary.avg_compression.items(), key=lambda x: x[1])[0] if summary.avg_compression else None
    print(f"{'Format':<20}", end="")
    for fmt in summary.formats:
        comp = summary.avg_compression.get(fmt, 0)
        marker = " ✓" if fmt == best_compression else ""
        print(f"{comp:>13.2f}x{marker}", end="")
    print()
    print(f"{'─'*70}")
    
    # Category breakdown
    print(f"\n📈 Score Breakdown:")
    print(f"{'─'*70}")
    print(f"{'Category':<20}", end="")
    for fmt in summary.formats:
        print(f"{fmt:>16}", end="")
    print()
    print(f"{'─'*70}")
    
    categories = [
        ("Text Similarity", summary.avg_text),
        ("Structural", summary.avg_struct),
        ("Semantic", summary.avg_semantic),
    ]
    
    for label, data in categories:
        print(f"{label:<20}", end="")
        for fmt in summary.formats:
            val = data.get(fmt, 0)
            print(f"{val:>15.1f}%", end="")
        print()
    
    print(f"{'─'*70}")
    
    # Code element comparison
    if results:
        print(f"\n🔍 Code Element Reproduction (classes/functions):")
        print(f"{'─'*70}")
        print(f"{'File':<20} {'Original':>10} ", end="")
        for fmt in summary.formats:
            print(f"{fmt:>14}", end="")
        print()
        print(f"{'─'*70}")
        
        for r in results[:8]:  # Limit display
            orig = f"{r.orig_classes}c/{r.orig_functions}f"
            print(f"{r.file_name:<20} {orig:>10} ", end="")
            for fmt in summary.formats:
                gen_c = r.gen_classes.get(fmt, 0)
                gen_f = r.gen_functions.get(fmt, 0)
                # Mark match
                c_match = "✓" if gen_c == r.orig_classes else ""
                f_match = "✓" if gen_f == r.orig_functions else ""
                print(f"{gen_c}c{c_match}/{gen_f}f{f_match}".rjust(14), end="")
            print()
        print(f"{'─'*70}")
    
    # Winner
    print(f"\n🏆 Best Format by Category:")
    
    metrics_to_check = [
        ("Overall Score", summary.avg_scores, True),
        ("Success Rate", summary.success_rates, True),
        ("Efficiency", summary.avg_compression, False),
        ("Text Similarity", summary.avg_text, True),
        ("Structural", summary.avg_struct, True),
        ("Semantic", summary.avg_semantic, True),
        ("Total Wins", summary.wins, True),
    ]
    
    for name, data, higher_better in metrics_to_check:
        if data:
            if higher_better:
                best = max(data.items(), key=lambda x: x[1])
            else:
                best = min(data.items(), key=lambda x: x[1])
            print(f"   {name:<20}: {best[0]} ({best[1]:.1f})")


def save_report(results: List[BenchmarkResult], summary: BenchmarkSummary, output: str):
    """Save benchmark report."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'total_files': summary.total_files,
        'formats': summary.formats,
        'summary': {
            'avg_scores': summary.avg_scores,
            'max_scores': summary.max_scores,
            'min_scores': summary.min_scores,
            'success_rates': summary.success_rates,
            'avg_compression': summary.avg_compression,
            'avg_gen_time': summary.avg_gen_time,
            'avg_text': summary.avg_text,
            'avg_struct': summary.avg_struct,
            'avg_semantic': summary.avg_semantic,
            'wins': summary.wins,
        },
        'results': [
            {
                'file': r.file_name,
                'size': r.file_size,
                'scores': r.scores,
                'spec_sizes': r.spec_sizes,
                'gen_times': r.gen_times,
                'text_scores': r.text_scores,
                'struct_scores': r.struct_scores,
                'semantic_scores': r.semantic_scores,
            }
            for r in results
        ]
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Also save markdown report
    md_path = path.with_suffix('.md')
    save_markdown_report(summary, results, md_path)
    
    print(f"\n📄 Reports saved:")
    print(f"   JSON: {path}")
    print(f"   Markdown: {md_path}")


def save_markdown_report(summary: BenchmarkSummary, results: List[BenchmarkResult], path: Path):
    """Save markdown report."""
    lines = [
        "# Format Benchmark Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"\nFiles analyzed: {summary.total_files}",
        "",
        "## Summary",
        "",
        "| Metric | " + " | ".join(summary.formats) + " |",
        "| --- | " + " | ".join(["---"] * len(summary.formats)) + " |",
    ]
    
    rows = [
        ("Avg Score", summary.avg_scores, "%.1f%%"),
        ("Success Rate", summary.success_rates, "%.0f%%"),
        ("Wins", summary.wins, "%d"),
        ("Compression", summary.avg_compression, "%.2fx"),
    ]
    
    for label, data, fmt_str in rows:
        row = f"| {label} |"
        for f in summary.formats:
            val = data.get(f, 0)
            row += f" {fmt_str % val} |"
        lines.append(row)
    
    lines.extend([
        "",
        "## Score Breakdown",
        "",
        "| Category | " + " | ".join(summary.formats) + " |",
        "| --- | " + " | ".join(["---"] * len(summary.formats)) + " |",
    ])
    
    categories = [
        ("Text", summary.avg_text),
        ("Structural", summary.avg_struct),
        ("Semantic", summary.avg_semantic),
    ]
    
    for label, data in categories:
        row = f"| {label} |"
        for f in summary.formats:
            val = data.get(f, 0)
            row += f" {val:.1f}% |"
        lines.append(row)
    
    lines.extend([
        "",
        "## Per-File Results",
        "",
        "| File | " + " | ".join(summary.formats) + " | Best |",
        "| --- | " + " | ".join(["---"] * len(summary.formats)) + " | --- |",
    ])
    
    for r in sorted(results, key=lambda x: -max(x.scores.values()) if x.scores else 0):
        row = f"| {r.file_name} |"
        best_fmt = ""
        best_score = 0
        for f in summary.formats:
            score = r.scores.get(f, 0)
            marker = "✓" if score > 50 else ""
            row += f" {score:.1f}%{marker} |"
            if score > best_score:
                best_score = score
                best_fmt = f
        row += f" {best_fmt} |"
        lines.append(row)
    
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Format Benchmark: Markdown vs Gherkin vs YAML')
    parser.add_argument('--folder', '-f', default='code2logic/', help='Folder to benchmark')
    parser.add_argument('--samples', '-s', help='Additional samples folder')
    parser.add_argument('--formats', nargs='+', default=FORMATS, help='Formats to test')
    parser.add_argument('--limit', '-l', type=int, help='Limit files')
    parser.add_argument('--no-llm', action='store_true', help='Use templates only')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick run (limit 3)')
    parser.add_argument('--output', '-o', default='examples/output/benchmark.json')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    if args.quick:
        args.limit = 3
    
    # Run benchmark
    results, summary = run_benchmark(
        args.folder,
        formats=args.formats,
        limit=args.limit,
        no_llm=args.no_llm,
        verbose=args.verbose,
    )
    
    # Add samples if provided
    if args.samples:
        sample_results, sample_summary = run_benchmark(
            args.samples,
            formats=args.formats,
            limit=args.limit,
            no_llm=args.no_llm,
            verbose=args.verbose,
        )
        results.extend(sample_results)
        summary = calculate_summary(results, args.formats)
    
    # Print and save
    print_summary(summary, results)
    save_report(results, summary, args.output)


if __name__ == '__main__':
    main()
