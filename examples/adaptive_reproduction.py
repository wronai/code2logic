#!/usr/bin/env python3
"""
Adaptive Code Reproduction Example.

Uses LLM capability detection to optimize format and chunking.

Usage:
    python adaptive_reproduction.py tests/samples/sample_dataclasses.py
    python adaptive_reproduction.py code2logic/models.py --output results/
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

from code2logic import AdaptiveReproducer, get_llm_capabilities


def main():
    parser = argparse.ArgumentParser(description='Adaptive code reproduction')
    parser.add_argument('source', help='Source file to reproduce')
    parser.add_argument('--output', '-o', default='adaptive_output', help='Output directory')
    parser.add_argument('--model', '-m', help='Model name')
    parser.add_argument('--show-caps', action='store_true', help='Show LLM capabilities')
    args = parser.parse_args()
    
    if args.show_caps:
        from code2logic import LLM_CAPABILITIES
        print("LLM Capabilities Database:")
        print("="*60)
        for model, caps in LLM_CAPABILITIES.items():
            print(f"\n{model}:")
            print(f"  Context: {caps['context_size']} tokens")
            print(f"  Quality: {caps['code_quality']}")
            print(f"  Best formats: {', '.join(caps['best_formats'])}")
        return
    
    print("="*60)
    print("ADAPTIVE CODE REPRODUCTION")
    print("="*60)
    
    reproducer = AdaptiveReproducer(model=args.model)
    
    print(f"\nSource: {args.source}")
    print(f"Model: {reproducer.model}")
    print(f"Capabilities: {reproducer.capabilities}")
    
    result = reproducer.reproduce(args.source, args.output)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print("="*60)
    print(f"  Format used:    {result.format_used}")
    print(f"  Chunks:         {result.chunks_used}")
    print(f"  Similarity:     {result.similarity:.1f}%")
    print(f"  Structural:     {result.structural_score:.1f}%")
    print(f"  Compression:    {result.compression_ratio:.2f}x")
    print(f"  Efficiency:     {result.efficiency_score:.2f}")
    print(f"\n  Output: {args.output}/")


if __name__ == '__main__':
    main()
