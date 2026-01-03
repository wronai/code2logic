#!/usr/bin/env python3
"""
Reproduction Benchmark Example.

Tests reproduction quality across different output formats
to find the best format for code generation from analysis.

Usage:
    python benchmark_reproduction.py
    python benchmark_reproduction.py --files code2logic/models.py tests/samples/sample_functions.py
    python benchmark_reproduction.py --formats gherkin csv json
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from code2logic import run_benchmark, ReproductionBenchmark


# Default test files
DEFAULT_FILES = [
    "tests/samples/sample_dataclasses.py",
    "tests/samples/sample_functions.py",
    "tests/samples/sample_class.py",
]


def main():
    parser = argparse.ArgumentParser(description='Run reproduction benchmark')
    parser.add_argument('--files', '-f', nargs='+', default=DEFAULT_FILES,
                        help='Files to benchmark')
    parser.add_argument('--formats', nargs='+', 
                        default=['gherkin', 'csv', 'json', 'yaml', 'markdown'],
                        help='Formats to test')
    parser.add_argument('--output', '-o', default='benchmark_results',
                        help='Output directory')
    parser.add_argument('--model', '-m', help='Model to use')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick test with gherkin only')
    args = parser.parse_args()
    
    # Verify files exist
    files = []
    for f in args.files:
        if Path(f).exists():
            files.append(f)
        else:
            print(f"Warning: File not found: {f}")
    
    if not files:
        print("No valid files to benchmark!")
        sys.exit(1)
    
    # Quick mode
    formats = ['gherkin'] if args.quick else args.formats
    
    print("="*60)
    print("REPRODUCTION BENCHMARK")
    print("="*60)
    print(f"\nFiles: {len(files)}")
    print(f"Formats: {', '.join(formats)}")
    print(f"Output: {args.output}/")
    
    # Run benchmark
    from code2logic import get_client
    client = get_client(model=args.model)
    
    benchmark = ReproductionBenchmark(client)
    
    all_results = []
    for file_path in files:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {file_path}")
        print("="*60)
        
        result = benchmark.run_single(file_path, formats)
        all_results.append(result)
        
        print(f"\n  Best: {result.best_format} ({result.best_similarity:.1f}%)")
    
    # Generate summary
    summary = benchmark._generate_summary(all_results)
    
    # Save results
    benchmark._save_results(Path(args.output), all_results, summary)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBest format overall: {summary['best_format']}")
    print(f"Average similarity:  {summary['best_average']:.1f}%")
    print("\nBy format:")
    for fmt, score in sorted(summary['average_by_format'].items(), key=lambda x: -x[1]):
        print(f"  {fmt:<12} {score:.1f}%")
    
    print(f"\nResults saved to: {args.output}/BENCHMARK_REPORT.md")


if __name__ == '__main__':
    main()
