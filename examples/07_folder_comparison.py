#!/usr/bin/env python3
"""
Folder-level Format Comparison for Code2Logic.

Compares reproduction quality across formats (Gherkin, YAML, Markdown hybrid)
for an entire folder of code files.

Usage:
    python examples/07_folder_comparison.py code2logic/
    python examples/07_folder_comparison.py code2logic/ --no-llm
    python examples/07_folder_comparison.py code2logic/ --limit 5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from code2logic import (
    analyze_project,
    ReproductionMetrics,
    compare_formats,
    get_client,
)
from code2logic.gherkin import GherkinGenerator
from code2logic.generators import YAMLGenerator, JSONGenerator
from code2logic.markdown_format import MarkdownHybridGenerator
from code2logic.reproduction import extract_code_block


@dataclass
class FileResult:
    """Result for a single file reproduction."""
    file_name: str
    original_chars: int
    scores: Dict[str, float] = field(default_factory=dict)
    spec_sizes: Dict[str, int] = field(default_factory=dict)
    generated_sizes: Dict[str, int] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Full comparison report."""
    folder: str
    timestamp: str
    formats: List[str]
    files_analyzed: int
    results: List[FileResult]
    summary: Dict[str, Dict[str, float]]
    best_format: Dict[str, str]
    recommendations: List[str]


def generate_specs(project, formats: List[str]) -> Dict[str, str]:
    """Generate specifications in multiple formats."""
    specs = {}
    
    if 'gherkin' in formats:
        gen = GherkinGenerator()
        specs['gherkin'] = gen.generate(project)
    
    if 'yaml' in formats:
        gen = YAMLGenerator()
        specs['yaml'] = gen.generate(project, detail='full')
    
    if 'json' in formats:
        gen = JSONGenerator()
        specs['json'] = gen.generate(project, detail='full')
    
    if 'markdown' in formats:
        gen = MarkdownHybridGenerator()
        spec = gen.generate(project)
        specs['markdown'] = spec.content
    
    return specs


def reproduce_from_spec(spec: str, format_name: str, file_name: str, client) -> str:
    """Reproduce code from specification using LLM."""
    
    format_instructions = {
        'gherkin': "This is a Gherkin/BDD specification. Generate Python code that implements all scenarios.",
        'yaml': "This is a YAML specification. Generate Python code that matches the structure.",
        'json': "This is a JSON specification. Generate Python code that matches the structure.",
        'markdown': "This is a Markdown specification with embedded Gherkin and YAML. Generate Python code.",
    }
    
    prompt = f"""{format_instructions.get(format_name, 'Generate code from this specification.')}

Target file: {file_name}

Specification:
{spec[:6000]}

Generate complete, working Python code for {file_name}.
Include all imports, classes, and functions.
Output only the Python code."""

    try:
        response = client.generate(prompt, max_tokens=4000)
        return extract_code_block(response)
    except Exception as e:
        return f"# Error: {e}"


def analyze_folder(
    folder_path: str,
    formats: List[str] = None,
    limit: int = None,
    no_llm: bool = False,
    verbose: bool = False,
) -> ComparisonReport:
    """Analyze and compare reproduction of an entire folder."""
    
    if formats is None:
        formats = ['gherkin', 'yaml', 'markdown']
    
    path = Path(folder_path)
    
    print(f"\n{'='*60}")
    print(f"FOLDER COMPARISON: {folder_path}")
    print(f"{'='*60}")
    print(f"Formats: {', '.join(formats)}")
    
    # Get Python files
    py_files = list(path.glob('*.py'))
    if limit:
        py_files = py_files[:limit]
    
    print(f"Files to analyze: {len(py_files)}")
    
    # Analyze project
    print("\n1. Analyzing project structure...")
    project = analyze_project(str(path))
    
    # Generate specs for entire project
    print("\n2. Generating specifications...")
    project_specs = generate_specs(project, formats)
    
    for fmt, spec in project_specs.items():
        print(f"   {fmt}: {len(spec)} chars ({len(spec)//4} tokens)")
    
    # Initialize client if using LLM
    client = None
    if not no_llm:
        try:
            client = get_client()
            print(f"\n3. Using LLM: {client.__class__.__name__}")
        except Exception as e:
            print(f"\n⚠️  LLM not available: {e}")
            print("   Running in template mode...")
            no_llm = True
    
    # Process each file
    results = []
    metrics = ReproductionMetrics(verbose=verbose)
    
    print(f"\n4. Reproducing {len(py_files)} files...")
    print("-" * 60)
    
    for i, py_file in enumerate(py_files):
        file_name = py_file.name
        print(f"\n[{i+1}/{len(py_files)}] {file_name}")
        
        original = py_file.read_text()
        result = FileResult(
            file_name=file_name,
            original_chars=len(original),
        )
        
        # Find module info for this file
        module_info = None
        for m in project.modules:
            if Path(m.path).name == file_name:
                module_info = m
                break
        
        if not module_info:
            print(f"   ⚠️  Module info not found, skipping")
            continue
        
        # Create single-file project for spec generation
        from code2logic.models import ProjectInfo
        single_project = ProjectInfo(
            name=file_name,
            root_path=str(py_file.parent),
            languages={'python': 1},
            modules=[module_info],
            dependency_graph={},
            dependency_metrics={},
            entrypoints=[],
            similar_functions={},
            total_files=1,
            total_lines=module_info.lines_total,
            generated_at='',
        )
        
        file_specs = generate_specs(single_project, formats)
        
        for fmt in formats:
            spec = file_specs.get(fmt, '')
            result.spec_sizes[fmt] = len(spec)
            
            print(f"   {fmt}: spec={len(spec)} chars", end="")
            
            if no_llm:
                # Template fallback
                generated = generate_template(spec, fmt)
            else:
                generated = reproduce_from_spec(spec, fmt, file_name, client)
            
            result.generated_sizes[fmt] = len(generated)
            print(f", generated={len(generated)} chars", end="")
            
            # Calculate metrics
            try:
                analysis = metrics.analyze(
                    original, generated, spec,
                    format_name=fmt,
                    source_file=file_name,
                )
                result.scores[fmt] = analysis.overall_score
                print(f", score={analysis.overall_score:.1f}%")
            except Exception as e:
                result.errors[fmt] = str(e)
                result.scores[fmt] = 0.0
                print(f", error: {e}")
        
        results.append(result)
    
    # Generate summary
    summary = calculate_summary(results, formats)
    best = find_best_format(summary)
    recommendations = generate_recommendations(summary, best)
    
    report = ComparisonReport(
        folder=folder_path,
        timestamp=datetime.now().isoformat(),
        formats=formats,
        files_analyzed=len(results),
        results=results,
        summary=summary,
        best_format=best,
        recommendations=recommendations,
    )
    
    return report


def generate_template(spec: str, fmt: str) -> str:
    """Generate template code when LLM not available."""
    import re
    
    classes = re.findall(r'class (\w+)', spec)
    functions = re.findall(r'(?:Scenario|def|function):\s*(\w+)', spec, re.IGNORECASE)
    
    code = '''from dataclasses import dataclass
from typing import Optional, List, Dict

'''
    
    unique_classes = list(set(classes))[:5]
    unique_functions = list(set(functions))[:10]
    
    for cls in unique_classes:
        code += f'''@dataclass
class {cls}:
    """Generated class."""
    pass

'''
    
    for func in unique_functions:
        if func not in unique_classes:
            code += f'''def {func}():
    """Generated function."""
    pass

'''
    
    return code


def calculate_summary(results: List[FileResult], formats: List[str]) -> Dict:
    """Calculate summary statistics."""
    summary = {}
    
    for fmt in formats:
        scores = [r.scores.get(fmt, 0) for r in results if fmt in r.scores]
        spec_sizes = [r.spec_sizes.get(fmt, 0) for r in results if fmt in r.spec_sizes]
        gen_sizes = [r.generated_sizes.get(fmt, 0) for r in results if fmt in r.generated_sizes]
        
        if scores:
            summary[fmt] = {
                'avg_score': sum(scores) / len(scores),
                'max_score': max(scores),
                'min_score': min(scores),
                'avg_spec_size': sum(spec_sizes) / len(spec_sizes) if spec_sizes else 0,
                'avg_generated_size': sum(gen_sizes) / len(gen_sizes) if gen_sizes else 0,
                'success_rate': len([s for s in scores if s > 50]) / len(scores) * 100,
            }
    
    return summary


def find_best_format(summary: Dict) -> Dict[str, str]:
    """Find best format for each metric."""
    best = {}
    
    metrics = ['avg_score', 'max_score', 'success_rate']
    
    for metric in metrics:
        best_fmt = None
        best_val = -1
        for fmt, stats in summary.items():
            if stats.get(metric, 0) > best_val:
                best_val = stats[metric]
                best_fmt = fmt
        best[metric] = best_fmt
    
    # Best efficiency (smallest spec size with good score)
    best_efficiency = None
    best_ratio = 0
    for fmt, stats in summary.items():
        if stats['avg_spec_size'] > 0:
            ratio = stats['avg_score'] / (stats['avg_spec_size'] / 1000)
            if ratio > best_ratio:
                best_ratio = ratio
                best_efficiency = fmt
    best['efficiency'] = best_efficiency
    
    return best


def generate_recommendations(summary: Dict, best: Dict) -> List[str]:
    """Generate recommendations based on analysis."""
    recs = []
    
    if best.get('avg_score'):
        recs.append(f"Best overall format: {best['avg_score']} (highest average score)")
    
    if best.get('efficiency'):
        recs.append(f"Most efficient format: {best['efficiency']} (best score/size ratio)")
    
    # Check if markdown hybrid is competitive
    if 'markdown' in summary:
        md_score = summary['markdown']['avg_score']
        other_scores = [s['avg_score'] for f, s in summary.items() if f != 'markdown']
        if other_scores and md_score >= max(other_scores) * 0.9:
            recs.append("Markdown hybrid format is competitive and may offer best readability")
    
    # Format-specific recommendations
    for fmt, stats in summary.items():
        if stats['success_rate'] < 50:
            recs.append(f"⚠️ {fmt} has low success rate ({stats['success_rate']:.0f}%)")
    
    return recs


def print_report(report: ComparisonReport):
    """Print comparison report."""
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print("="*60)
    
    # Summary table
    print(f"\n📊 Summary ({report.files_analyzed} files):")
    print("-" * 60)
    print(f"{'Format':<12} {'Avg Score':>10} {'Max':>8} {'Success%':>10} {'Spec Size':>12}")
    print("-" * 60)
    
    for fmt in report.formats:
        if fmt in report.summary:
            s = report.summary[fmt]
            print(f"{fmt:<12} {s['avg_score']:>9.1f}% {s['max_score']:>7.1f}% {s['success_rate']:>9.0f}% {s['avg_spec_size']:>11.0f}")
    
    print("-" * 60)
    
    # Best formats
    print(f"\n🏆 Best Format by Category:")
    for category, fmt in report.best_format.items():
        print(f"   {category}: {fmt}")
    
    # File-by-file results
    print(f"\n📁 Per-File Results:")
    print("-" * 60)
    
    for result in sorted(report.results, key=lambda r: -max(r.scores.values()) if r.scores else 0)[:10]:
        print(f"\n   {result.file_name} ({result.original_chars} chars)")
        for fmt, score in result.scores.items():
            indicator = "✓" if score > 50 else "✗"
            print(f"      {fmt}: {score:.1f}% {indicator}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    for rec in report.recommendations:
        print(f"   • {rec}")


def save_report(report: ComparisonReport, output_path: str):
    """Save report to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict for JSON
    data = {
        'folder': report.folder,
        'timestamp': report.timestamp,
        'formats': report.formats,
        'files_analyzed': report.files_analyzed,
        'summary': report.summary,
        'best_format': report.best_format,
        'recommendations': report.recommendations,
        'results': [
            {
                'file': r.file_name,
                'original_chars': r.original_chars,
                'scores': r.scores,
                'spec_sizes': r.spec_sizes,
                'generated_sizes': r.generated_sizes,
            }
            for r in report.results
        ]
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n📄 Report saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description='Compare reproduction formats for a folder')
    parser.add_argument('folder', nargs='?', default='code2logic/')
    parser.add_argument('--formats', '-f', nargs='+', default=['gherkin', 'yaml', 'markdown'])
    parser.add_argument('--limit', '-l', type=int, help='Limit number of files')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM, use templates')
    parser.add_argument('--output', '-o', default='examples/output/folder_comparison.json')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    
    report = analyze_folder(
        args.folder,
        formats=args.formats,
        limit=args.limit,
        no_llm=args.no_llm,
        verbose=args.verbose,
    )
    
    print_report(report)
    save_report(report, args.output)


if __name__ == '__main__':
    main()
