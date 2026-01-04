#!/usr/bin/env python3
"""
LLM Integration Example - Use with OpenRouter or Ollama.

Usage:
    python 05_llm_integration.py
    python 05_llm_integration.py --provider ollama
    python 05_llm_integration.py --list-models
"""

import sys
import argparse
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from code2logic import (
    get_client,
    OpenRouterClient,
    OllamaLocalClient,
    LLM_CAPABILITIES,
    suggest_refactoring,
)


def main():
    parser = argparse.ArgumentParser(description='LLM integration')
    parser.add_argument('project', nargs='?', default='code2logic/')
    parser.add_argument(
        '--provider',
        '-p',
        default=os.environ.get('CODE2LOGIC_DEFAULT_PROVIDER', 'auto'),
        choices=['auto', 'openrouter', 'ollama', 'litellm'],
    )
    parser.add_argument('--model', '-m', help='Model name')
    parser.add_argument('--list-models', action='store_true')
    args = parser.parse_args()
    
    print("="*60)
    print("CODE2LOGIC - LLM INTEGRATION")
    print("="*60)
    
    # List models
    if args.list_models:
        print("\nSupported Models:")
        print("-"*40)
        for model, caps in LLM_CAPABILITIES.items():
            if model != 'default':
                print(f"\n  {model}")
                print(f"    Context: {caps['context_size']} tokens")
                print(f"    Quality: {caps['code_quality']}")
                print(f"    Formats: {', '.join(caps['best_formats'])}")
        return
    
    # Get client
    print(f"\nProvider: {args.provider}")
    
    try:
        if args.provider == 'auto':
            if args.model:
                print("Note: --model is ignored when --provider=auto")
            client = get_client('auto')
        else:
            client = get_client(args.provider, args.model)

        provider_name = getattr(client, 'provider', None) or client.__class__.__name__
        print(f"Selected: {provider_name}")
        print(f"Model: {getattr(client, 'model', 'default')}")
        print(f"Available: {client.is_available()}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Generate suggestions
    print(f"\nAnalyzing {args.project}...")
    
    report = suggest_refactoring(args.project, use_llm=True, client=client)
    
    print(f"\n{'='*60}")
    print("REFACTORING SUGGESTIONS")
    print("="*60)
    print(f"\nDuplicates: {len(report.duplicates)}")
    print(f"Quality issues: {len(report.quality_issues)}")
    
    if report.suggestions:
        print(f"\nLLM Suggestions:")
        print("-"*40)
        for sug in report.suggestions:
            print(sug.suggestion[:500])


if __name__ == '__main__':
    main()
