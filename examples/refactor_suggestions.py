#!/usr/bin/env python3
"""
Example: LLM-powered refactoring suggestions using Ollama.

This script demonstrates how to:
1. Analyze a project with code2logic
2. Identify potential issues (complexity, duplicates, long files)
3. Use Ollama to generate specific refactoring suggestions

Requirements:
    pip install httpx
    ollama serve
    ollama pull qwen2.5-coder:7b
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic import analyze_project, CSVGenerator

OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen2.5-coder:7b"


def check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        response = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def generate_with_ollama(prompt: str, system: str = None) -> str:
    """Generate text using Ollama."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 2000}
    }
    if system:
        payload["system"] = system

    response = httpx.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "")


def find_issues(project) -> list:
    """Find potential code issues."""
    issues = []

    # Track duplicates
    hash_groups = defaultdict(list)
    sig_groups = defaultdict(list)
    intent_groups = defaultdict(list)

    for m in project.modules:
        # Long files
        if m.lines_code > 500:
            issues.append({
                'type': 'long_file',
                'severity': 'medium',
                'path': m.path,
                'lines': m.lines_code,
                'suggestion': 'Consider splitting into multiple modules'
            })

        # Long functions
        for f in m.functions:
            if f.lines > 50:
                issues.append({
                    'type': 'long_function',
                    'severity': 'medium',
                    'path': m.path,
                    'name': f.name,
                    'lines': f.lines,
                    'suggestion': 'Consider breaking into smaller functions'
                })

            # Track for duplicates
            import hashlib
            sig = f"({','.join(f.params)})->{f.return_type or ''}"
            h = hashlib.md5(f"{f.name}:{sig}".encode()).hexdigest()[:8]

            hash_groups[h].append(f"{m.path}::{f.name}")
            sig_groups[sig].append(f"{m.path}::{f.name}")

            if f.intent:
                key = f.intent.lower()[:50]
                intent_groups[key].append(f"{m.path}::{f.name}")

        # Classes
        for c in m.classes:
            if len(c.methods) > 20:
                issues.append({
                    'type': 'large_class',
                    'severity': 'medium',
                    'path': m.path,
                    'name': c.name,
                    'methods': len(c.methods),
                    'suggestion': 'Consider splitting into smaller classes (SRP)'
                })

            for method in c.methods:
                sig = f"({','.join(method.params)})->{method.return_type or ''}"
                import hashlib
                h = hashlib.md5(f"{method.name}:{sig}".encode()).hexdigest()[:8]
                hash_groups[h].append(f"{m.path}::{c.name}.{method.name}")

    # Add duplicate issues
    for h, funcs in hash_groups.items():
        if len(funcs) > 1:
            issues.append({
                'type': 'duplicate',
                'severity': 'high',
                'hash': h,
                'functions': funcs,
                'suggestion': 'Extract to shared utility function'
            })

    # Similar signatures (potential DRY violations)
    for sig, funcs in sig_groups.items():
        if len(funcs) > 3:
            issues.append({
                'type': 'similar_signature',
                'severity': 'low',
                'signature': sig,
                'count': len(funcs),
                'suggestion': 'Consider generic implementation'
            })

    return issues


def get_llm_suggestions(issues: list, project) -> str:
    """Get detailed suggestions from LLM."""
    # Build context
    context_parts = []

    context_parts.append(f"Project: {project.name}")
    context_parts.append(f"Total files: {project.total_files}")
    context_parts.append(f"Total lines: {project.total_lines}")
    context_parts.append(f"Languages: {', '.join(project.languages.keys())}")
    context_parts.append("")

    context_parts.append("Issues found:")
    for i, issue in enumerate(issues[:15], 1):
        context_parts.append(f"\n{i}. [{issue['type'].upper()}] {issue.get('severity', 'medium')} severity")

        if 'path' in issue:
            context_parts.append(f"   Path: {issue['path']}")
        if 'name' in issue:
            context_parts.append(f"   Name: {issue['name']}")
        if 'lines' in issue:
            context_parts.append(f"   Lines: {issue['lines']}")
        if 'functions' in issue:
            context_parts.append(f"   Functions: {', '.join(issue['functions'][:5])}")

        context_parts.append(f"   Initial suggestion: {issue['suggestion']}")

    context = '\n'.join(context_parts)

    system = """You are an expert software architect reviewing code for refactoring opportunities.
Provide specific, actionable suggestions with code examples where helpful.
Prioritize by impact and ease of implementation."""

    prompt = f"""Analyze these code issues and provide detailed refactoring suggestions:

{context}

For each issue:
1. Explain the specific problem
2. Describe the recommended refactoring pattern
3. Provide a concrete code example if applicable
4. Estimate effort (low/medium/high)
5. List potential risks

Focus on the most impactful changes first."""

    return generate_with_ollama(prompt, system)


def main():
    """Main refactoring analysis."""
    if len(sys.argv) < 2:
        print("Usage: python refactor_suggestions.py /path/to/project")
        sys.exit(1)

    project_path = sys.argv[1]
    use_llm = "--no-llm" not in sys.argv

    print(f"Analyzing project: {project_path}")
    project = analyze_project(project_path)

    print(f"Found {project.total_files} files, {project.total_lines} lines")

    # Find issues
    issues = find_issues(project)

    print("\n" + "=" * 70)
    print("REFACTORING ANALYSIS")
    print("=" * 70)

    # Group by type
    by_type = defaultdict(list)
    for issue in issues:
        by_type[issue['type']].append(issue)

    print(f"\nTotal issues found: {len(issues)}")
    print("\nBy type:")
    for t, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {t}: {len(items)}")

    # Print top issues
    print("\n" + "-" * 70)
    print("TOP ISSUES")
    print("-" * 70)

    high_severity = [i for i in issues if i.get('severity') == 'high']
    for issue in high_severity[:10]:
        print(f"\n[{issue['type'].upper()}] - HIGH SEVERITY")
        for k, v in issue.items():
            if k not in ('type', 'severity'):
                print(f"  {k}: {v}")

    # Get LLM suggestions
    if use_llm and check_ollama():
        print("\n" + "-" * 70)
        print("LLM REFACTORING SUGGESTIONS")
        print("-" * 70)

        suggestions = get_llm_suggestions(issues, project)
        print(suggestions)
    elif use_llm:
        print("\n⚠️  Ollama not running. Start with: ollama serve")
        print("   Skipping LLM suggestions. Use --no-llm to suppress this warning.")

    # Save results
    output = {
        'project': {
            'name': project.name,
            'files': project.total_files,
            'lines': project.total_lines,
        },
        'issues': issues,
        'summary': {
            'total': len(issues),
            'by_type': {t: len(items) for t, items in by_type.items()},
            'high_severity': len(high_severity),
        }
    }

    output_file = "refactoring_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()