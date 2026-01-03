#!/usr/bin/env python3
"""
Project-level Code Reproduction Example.

Reproduces entire projects with multi-file support.

Usage:
    python project_reproduction.py tests/samples/
    python project_reproduction.py code2logic/ --parallel
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from code2logic import ProjectReproducer, reproduce_project


def main():
    parser = argparse.ArgumentParser(description='Project-level code reproduction')
    parser.add_argument('project', help='Project path to reproduce')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--target', '-t', help='Target language for all files')
    parser.add_argument('--parallel', '-p', action='store_true', help='Use parallel processing')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Max parallel workers')
    args = parser.parse_args()
    
    print("="*60)
    print("PROJECT REPRODUCTION")
    print("="*60)
    print(f"\nProject: {args.project}")
    
    reproducer = ProjectReproducer(
        max_workers=args.workers,
        target_lang=args.target,
    )
    
    result = reproducer.reproduce_project(
        args.project,
        output_dir=args.output,
        parallel=args.parallel,
    )
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"  Total files:      {result.total_files}")
    print(f"  Successful:       {result.successful_files}")
    print(f"  Failed:           {result.failed_files}")
    print(f"  Avg compression:  {result.avg_compression:.2f}x")
    print(f"  Avg similarity:   {result.avg_similarity:.1f}%")
    print(f"  Avg structural:   {result.avg_structural:.1f}%")
    
    if result.by_language:
        print(f"\n  By Language:")
        for lang, data in sorted(result.by_language.items()):
            print(f"    {lang}: {data['count']} files, {data['similarity']:.1f}% similarity")


if __name__ == '__main__':
    main()
