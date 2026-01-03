#!/usr/bin/env python3
"""
Universal Code Reproduction Example.

Language-agnostic reproduction using UCLR (Universal Code Logic Representation).

Usage:
    python universal_reproduction.py tests/samples/sample_dataclasses.py
    python universal_reproduction.py tests/samples/sample_typescript.ts --target python
    python universal_reproduction.py tests/samples/sample_go.go --no-llm
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

from code2logic import (
    UniversalReproducer,
    UniversalParser,
    Language,
)


def main():
    parser = argparse.ArgumentParser(description='Universal code reproduction')
    parser.add_argument('source', help='Source file to reproduce')
    parser.add_argument('--target', '-t', help='Target language (python, typescript, go, sql)')
    parser.add_argument('--output', '-o', default='universal_output', help='Output directory')
    parser.add_argument('--no-llm', action='store_true', help='Use template generation instead of LLM')
    parser.add_argument('--show-logic', action='store_true', help='Show extracted logic')
    parser.add_argument('--show-compact', action='store_true', help='Show compact UCLR format')
    args = parser.parse_args()
    
    print("="*60)
    print("UNIVERSAL CODE REPRODUCTION")
    print("="*60)
    
    # Parse source
    uparser = UniversalParser()
    logic = uparser.parse(args.source)
    
    print(f"\nSource: {args.source}")
    print(f"Detected language: {logic.source_language.value}")
    print(f"Elements found: {len(logic.elements)}")
    print(f"Imports: {len(logic.imports)}")
    
    if args.show_logic:
        print(f"\n{'='*60}")
        print("EXTRACTED LOGIC")
        print("="*60)
        for elem in logic.elements:
            print(f"\n[{elem.type.value}] {elem.name}")
            if elem.attributes:
                print(f"  Attributes: {len(elem.attributes)}")
                for attr in elem.attributes[:5]:
                    print(f"    - {attr['name']}: {attr.get('type', 'any')}")
            if elem.children:
                print(f"  Methods: {len(elem.children)}")
                for child in elem.children[:5]:
                    print(f"    - {child.name}()")
    
    if args.show_compact:
        print(f"\n{'='*60}")
        print("COMPACT UCLR FORMAT")
        print("="*60)
        compact = logic.to_compact()
        print(compact[:2000])
        if len(compact) > 2000:
            print(f"\n... ({len(compact) - 2000} more chars)")
        print(f"\nTotal: {len(compact)} chars")
        return
    
    # Reproduce
    reproducer = UniversalReproducer()
    
    print(f"\nReproducing...")
    result = reproducer.reproduce(
        args.source,
        target_lang=args.target,
        output_dir=args.output,
        use_llm=not args.no_llm,
    )
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print("="*60)
    print(f"  Source language:  {result['source_language']}")
    print(f"  Target language:  {result['target_language']}")
    print(f"  Source chars:     {result['source_chars']}")
    print(f"  Logic chars:      {result['logic_chars']}")
    print(f"  Generated chars:  {result['generated_chars']}")
    print(f"  Compression:      {result['compression_ratio']:.2f}x")
    print(f"  Similarity:       {result['similarity']:.1f}%")
    print(f"  Structural:       {result['structural_score']:.1f}%")
    print(f"\n  Output: {args.output}/")


if __name__ == '__main__':
    main()
