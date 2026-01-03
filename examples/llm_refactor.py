#!/usr/bin/env python3
"""
LLM Refactoring Suggestions Example (Simplified).

Uses code2logic's built-in LLM clients and analysis.

Usage:
    python llm_refactor.py /path/to/project
    python llm_refactor.py /path/to/project --provider openrouter
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic import (
    analyze_project,
    CSVGenerator,
    get_client,
    analyze_code_quality,
)


def format_issues_for_llm(issues: dict, max_issues: int = 10) -> str:
    """Format issues for LLM context."""
    lines = ["Code Quality Issues:\n"]
    count = 0
    for category, issue_list in issues.items():
        for issue in issue_list[:5]:
            lines.append(f"- {category}: {issue.get('path')}::{issue.get('name', 'N/A')} "
                        f"(severity: {issue.get('severity', 'medium')})")
            count += 1
            if count >= max_issues:
                break
        if count >= max_issues:
            break
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='LLM-powered refactoring suggestions')
    parser.add_argument('project', help='Project path to analyze')
    parser.add_argument('--provider', '-p', choices=['ollama', 'openrouter', 'litellm'],
                        default='ollama', help='LLM provider')
    parser.add_argument('--model', '-m', help='Model to use')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM analysis')
    args = parser.parse_args()
    
    print(f"Analyzing {args.project}...")
    project = analyze_project(args.project)
    
    # Get code quality issues
    issues = analyze_code_quality(project)
    
    print(f"\nFound {sum(len(v) for v in issues.values())} issues")
    
    if args.no_llm:
        # Just print issues
        for category, issue_list in issues.items():
            print(f"\n{category}:")
            for issue in issue_list[:5]:
                print(f"  - {issue.get('path')}::{issue.get('name', 'N/A')}")
        return
    
    # Generate CSV for context
    csv_gen = CSVGenerator()
    csv_output = csv_gen.generate(project, detail='minimal')
    
    # Get LLM client
    try:
        client = get_client(args.provider, args.model)
        print(f"Using {args.provider} for suggestions...")
    except Exception as e:
        print(f"LLM not available: {e}")
        return
    
    # Generate suggestions
    issues_text = format_issues_for_llm(issues)
    
    prompt = f"""Analyze this codebase and suggest refactoring improvements.

{issues_text}

Code structure (CSV):
{csv_output[:3000]}

Provide 3-5 specific, actionable refactoring suggestions with:
1. What to refactor
2. Why it's needed  
3. How to implement it
4. Estimated effort (low/medium/high)
"""

    system = "You are a senior software architect specializing in code quality and refactoring."
    
    print("\nGenerating suggestions...\n")
    response = client.generate(prompt, system=system, max_tokens=2000)
    
    print("="*60)
    print("REFACTORING SUGGESTIONS")
    print("="*60)
    print(response)


if __name__ == '__main__':
    main()
