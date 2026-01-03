#!/usr/bin/env python3
"""
OpenRouter Code Reproduction Example (Simplified).

Uses code2logic's built-in reproduction utilities.

Usage:
    python openrouter_reproduction.py code2logic/models.py
    python openrouter_reproduction.py code2logic/models.py --dry-run
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic import (
    OpenRouterClient,
    CodeReproducer,
    generate_file_gherkin,
    Config,
)


def main():
    parser = argparse.ArgumentParser(description='Code reproduction with OpenRouter')
    parser.add_argument('source', help='Source file to reproduce')
    parser.add_argument('--output', '-o', default='reproduction_output', help='Output directory')
    parser.add_argument('--model', '-m', help='OpenRouter model')
    parser.add_argument('--dry-run', action='store_true', help='Generate Gherkin only')
    parser.add_argument('--list-models', action='store_true', help='List recommended models')
    args = parser.parse_args()
    
    if args.list_models:
        print("Recommended models:")
        for model, desc in OpenRouterClient.list_recommended_models():
            print(f"  {model:<45} - {desc}")
        return
    
    print("="*60)
    print("CODE2LOGIC - CODE REPRODUCTION")
    print("="*60)
    
    config = Config()
    model = args.model or config.get_model('openrouter')
    print(f"\nSource: {args.source}")
    print(f"Model: {model}")
    
    # Dry run - just show Gherkin
    if args.dry_run:
        gherkin = generate_file_gherkin(args.source)
        print(f"\nGherkin ({len(gherkin)} chars):\n")
        print(gherkin[:2000])
        if len(gherkin) > 2000:
            print(f"\n... ({len(gherkin) - 2000} more chars)")
        return
    
    # Full reproduction
    try:
        client = OpenRouterClient(model=model)
        reproducer = CodeReproducer(client)
        results = reproducer.reproduce_file(args.source, args.output)
        
        comp = results['comparison']
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"  Similarity:       {comp['similarity_percent']}%")
        print(f"  Structural Match: {comp['structural_score']}%")
        print(f"  Output: {args.output}/")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
