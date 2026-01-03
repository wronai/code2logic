#!/usr/bin/env python3
"""
Metrics Analysis Example - Detailed reproduction quality analysis.

Usage:
    python 06_metrics.py tests/samples/sample_dataclasses.py
    python 06_metrics.py code2logic/models.py --compare-formats
    python 06_metrics.py tests/samples/ --batch
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from code2logic import (
    ReproductionMetrics,
    analyze_reproduction,
    compare_formats,
    generate_file_gherkin,
    generate_file_yaml,
    generate_file_json,
    reproduce_file,
    get_client,
)
from code2logic.reproduction import extract_code_block


def analyze_single(source_path: str, verbose: bool = False):
    """Analyze single file with detailed metrics."""
    path = Path(source_path)
    original = path.read_text()
    
    print(f"\nAnalyzing: {source_path}")
    print("="*60)
    
    # Generate spec and reproduce
    spec = generate_file_gherkin(source_path)
    
    client = get_client()
    prompt = f"""Generate Python code from this Gherkin specification:

{spec}

Generate complete, working Python code."""
    
    response = client.generate(prompt, max_tokens=4000)
    generated = extract_code_block(response)
    
    # Analyze with metrics
    metrics = ReproductionMetrics(verbose=verbose)
    result = metrics.analyze(
        original, generated, spec, 
        format_name='gherkin',
        source_file=source_path,
    )
    
    # Print results
    print(f"\n📊 Overall Score: {result.overall_score:.1f}% (Grade: {result.quality_grade})")
    
    print(f"\n📝 Text Metrics:")
    print(f"   Character Similarity: {result.text.char_similarity:.1f}%")
    print(f"   Line Similarity:      {result.text.line_similarity:.1f}%")
    print(f"   Word Similarity:      {result.text.word_similarity:.1f}%")
    print(f"   Jaccard Similarity:   {result.text.jaccard_similarity:.1f}%")
    print(f"   Cosine Similarity:    {result.text.cosine_similarity:.1f}%")
    print(f"   Diff Changes:         {result.text.diff_changed} lines")
    
    print(f"\n🏗️ Structural Metrics:")
    print(f"   Classes:    {result.structural.classes_original} → {result.structural.classes_generated} {'✓' if result.structural.classes_match else '✗'}")
    print(f"   Functions:  {result.structural.functions_original} → {result.structural.functions_generated} {'✓' if result.structural.functions_match else '✗'}")
    print(f"   Methods:    {result.structural.methods_original} → {result.structural.methods_generated} {'✓' if result.structural.methods_match else '✗'}")
    print(f"   Imports:    {result.structural.imports_original} → {result.structural.imports_generated} {'✓' if result.structural.imports_match else '✗'}")
    print(f"   Score:      {result.structural.structural_score:.1f}%")
    
    print(f"\n🎯 Semantic Metrics:")
    print(f"   Naming Similarity:  {result.semantic.naming_similarity:.1f}%")
    print(f"   Type Hints:         {result.semantic.type_hints_present:.1f}%")
    print(f"   Docstrings:         {result.semantic.docstring_present:.1f}%")
    print(f"   Signatures:         {result.semantic.signature_match:.1f}%")
    print(f"   Intent Score:       {result.semantic.intent_score:.1f}%")
    
    print(f"\n📦 Format Efficiency:")
    print(f"   Spec Size:       {result.format.spec_chars} chars ({result.format.spec_tokens} tokens)")
    print(f"   Compression:     {result.format.compression_ratio:.2f}x")
    print(f"   Efficiency:      {result.format.efficiency_score:.2f}")
    
    print(f"\n💡 Recommendations:")
    for rec in result.recommendations:
        print(f"   • {rec}")
    
    return result


def compare_all_formats(source_path: str):
    """Compare reproduction across all formats."""
    path = Path(source_path)
    original = path.read_text()
    
    print(f"\nComparing formats for: {source_path}")
    print("="*60)
    
    # Generate specs
    formats = {
        'gherkin': generate_file_gherkin(source_path),
        'yaml': generate_file_yaml(source_path),
        'json': generate_file_json(source_path),
    }
    
    client = get_client()
    results = {}
    
    for fmt, spec in formats.items():
        print(f"\n  Testing {fmt}...", end=" ", flush=True)
        
        prompt = f"""Generate Python code from this {fmt} specification:

{spec[:4000]}

Generate complete, working Python code."""
        
        try:
            response = client.generate(prompt, max_tokens=4000)
            generated = extract_code_block(response)
            results[fmt] = (spec, generated)
            print(f"✓ ({len(generated)} chars)")
        except Exception as e:
            print(f"✗ ({e})")
            results[fmt] = (spec, "")
    
    # Compare
    comparison = compare_formats(original, results)
    
    # Print comparison
    print(f"\n📊 Format Comparison:")
    print("-"*60)
    print(f"{'Format':<12} {'Overall':>10} {'Grade':>6} {'Text':>10} {'Struct':>10} {'Semantic':>10}")
    print("-"*60)
    
    for fmt, summary in comparison['summary'].items():
        print(f"{fmt:<12} {summary['overall']:>9.1f}% {summary['grade']:>6} "
              f"{summary['text']:>9.1f}% {summary['structural']:>9.1f}% {summary['semantic']:>9.1f}%")
    
    print("-"*60)
    print(f"\n🏆 Best Format by Category:")
    for category, fmt in comparison['best'].items():
        print(f"   {category}: {fmt}")
    
    return comparison


def batch_analyze(project_path: str):
    """Analyze all Python files in a directory."""
    path = Path(project_path)
    files = list(path.glob('*.py'))
    
    print(f"\nBatch Analysis: {project_path}")
    print(f"Found {len(files)} Python files")
    print("="*60)
    
    all_results = []
    
    for file_path in files[:5]:  # Limit to 5 for demo
        try:
            result = analyze_single(str(file_path), verbose=False)
            all_results.append({
                'file': file_path.name,
                'score': result.overall_score,
                'grade': result.quality_grade,
            })
        except Exception as e:
            print(f"Error analyzing {file_path.name}: {e}")
    
    # Summary
    if all_results:
        avg_score = sum(r['score'] for r in all_results) / len(all_results)
        print(f"\n📈 Batch Summary:")
        print(f"   Files analyzed: {len(all_results)}")
        print(f"   Average score:  {avg_score:.1f}%")
        
        print(f"\n   By file:")
        for r in sorted(all_results, key=lambda x: -x['score']):
            print(f"     {r['file']}: {r['score']:.1f}% ({r['grade']})")


def main():
    parser = argparse.ArgumentParser(description='Reproduction metrics analysis')
    parser.add_argument('source', nargs='?', default='tests/samples/sample_dataclasses.py')
    parser.add_argument('--compare-formats', '-c', action='store_true')
    parser.add_argument('--batch', '-b', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--output', '-o', default='examples/output/metrics_report.md', help='Save report to file')
    args = parser.parse_args()
    
    print("="*60)
    print("CODE2LOGIC - METRICS ANALYSIS")
    print("="*60)
    
    if args.batch:
        batch_analyze(args.source)
    elif args.compare_formats:
        result = compare_all_formats(args.source)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2))
            print(f"\nSaved to: {args.output}")
    else:
        result = analyze_single(args.source, args.verbose)
        if args.output:
            Path(args.output).write_text(result.to_report())
            print(f"\nReport saved to: {args.output}")


if __name__ == '__main__':
    main()
