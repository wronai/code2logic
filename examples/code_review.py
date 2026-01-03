#!/usr/bin/env python3
"""
Example: Automated Code Review with LLM.

Performs automated code review using code2logic analysis + LLM.
Checks for:
- Code quality issues
- Security vulnerabilities
- Performance problems
- Best practice violations

Usage:
    python code_review.py /path/to/project
    python code_review.py /path/to/project --strict
    python code_review.py /path/to/project --focus security
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from code2logic import analyze_project, CSVGenerator


OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen2.5-coder:7b"

# Security patterns to check
SECURITY_PATTERNS = {
    'sql_injection': ['execute', 'raw', 'cursor.execute'],
    'command_injection': ['os.system', 'subprocess.call', 'eval', 'exec'],
    'path_traversal': ['open(', 'Path(', 'os.path.join'],
    'hardcoded_secrets': ['password', 'secret', 'api_key', 'token'],
    'insecure_random': ['random.', 'randint'],
}

# Performance anti-patterns
PERFORMANCE_PATTERNS = {
    'n_plus_one': ['for.*in.*:', '.all()', '.filter('],
    'large_memory': ['readlines()', 'list(', '.to_list()'],
    'blocking_io': ['requests.get', 'urllib.urlopen', 'open('],
}


def check_ollama() -> bool:
    """Check if Ollama is running."""
    if not HTTPX_AVAILABLE:
        return False
    try:
        response = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def generate_with_ollama(prompt: str, system: str = None) -> str:
    """Generate with Ollama."""
    if not HTTPX_AVAILABLE:
        return "httpx not installed"
    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 2000}
    }
    if system:
        payload["system"] = system
    
    try:
        response = httpx.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"Error: {e}"


def analyze_code_quality(project) -> Dict[str, List[Dict]]:
    """Analyze code quality issues."""
    issues = defaultdict(list)
    
    for module in project.modules:
        # Complexity analysis
        for func in module.functions:
            if func.complexity > 10:
                issues['high_complexity'].append({
                    'path': module.path,
                    'name': func.name,
                    'complexity': func.complexity,
                    'severity': 'high' if func.complexity > 15 else 'medium',
                    'suggestion': 'Consider breaking into smaller functions',
                })
            
            # Long functions
            if func.lines > 50:
                issues['long_function'].append({
                    'path': module.path,
                    'name': func.name,
                    'lines': func.lines,
                    'severity': 'medium',
                    'suggestion': 'Consider extracting helper functions',
                })
            
            # Missing docstrings
            if not func.docstring and not func.name.startswith('_'):
                issues['missing_docstring'].append({
                    'path': module.path,
                    'name': func.name,
                    'severity': 'low',
                    'suggestion': 'Add docstring explaining purpose and parameters',
                })
            
            # Too many parameters
            if len(func.params) > 5:
                issues['too_many_params'].append({
                    'path': module.path,
                    'name': func.name,
                    'params': len(func.params),
                    'severity': 'medium',
                    'suggestion': 'Consider using a config object or dataclass',
                })
        
        # Large classes
        for cls in module.classes:
            if len(cls.methods) > 20:
                issues['god_class'].append({
                    'path': module.path,
                    'name': cls.name,
                    'methods': len(cls.methods),
                    'severity': 'high',
                    'suggestion': 'Consider splitting into smaller classes (SRP)',
                })
        
        # Long files
        if module.lines_code > 500:
            issues['long_file'].append({
                'path': module.path,
                'lines': module.lines_code,
                'severity': 'medium',
                'suggestion': 'Consider splitting into multiple modules',
            })
    
    return dict(issues)


def analyze_security(project) -> Dict[str, List[Dict]]:
    """Analyze potential security issues."""
    issues = defaultdict(list)
    
    for module in project.modules:
        for func in module.functions:
            func_calls = ' '.join(func.calls or [])
            
            for issue_type, patterns in SECURITY_PATTERNS.items():
                for pattern in patterns:
                    if pattern.lower() in func_calls.lower() or pattern.lower() in func.name.lower():
                        issues[issue_type].append({
                            'path': module.path,
                            'name': func.name,
                            'pattern': pattern,
                            'severity': 'high' if issue_type in ['sql_injection', 'command_injection'] else 'medium',
                            'suggestion': f'Review for potential {issue_type.replace("_", " ")}',
                        })
    
    return dict(issues)


def analyze_performance(project) -> Dict[str, List[Dict]]:
    """Analyze potential performance issues."""
    issues = defaultdict(list)
    
    for module in project.modules:
        for func in module.functions:
            # Check async usage
            if func.is_async:
                if 'time.sleep' in ' '.join(func.calls or []):
                    issues['blocking_in_async'].append({
                        'path': module.path,
                        'name': func.name,
                        'severity': 'high',
                        'suggestion': 'Use asyncio.sleep instead of time.sleep in async functions',
                    })
            
            # Check for potential N+1 queries (heuristic)
            if func.complexity > 5 and any('query' in c.lower() or 'get' in c.lower() for c in (func.calls or [])):
                issues['potential_n_plus_one'].append({
                    'path': module.path,
                    'name': func.name,
                    'severity': 'medium',
                    'suggestion': 'Review for N+1 query patterns',
                })
    
    return dict(issues)


def generate_review_report(
    quality_issues: Dict,
    security_issues: Dict,
    performance_issues: Dict,
    project_name: str,
) -> str:
    """Generate markdown review report."""
    lines = [
        f"# Code Review Report: {project_name}",
        "",
        "## Summary",
        "",
    ]
    
    total_issues = sum(len(v) for v in quality_issues.values())
    total_issues += sum(len(v) for v in security_issues.values())
    total_issues += sum(len(v) for v in performance_issues.values())
    
    high = sum(1 for issues in [quality_issues, security_issues, performance_issues] 
               for v in issues.values() for i in v if i.get('severity') == 'high')
    medium = sum(1 for issues in [quality_issues, security_issues, performance_issues]
                 for v in issues.values() for i in v if i.get('severity') == 'medium')
    low = total_issues - high - medium
    
    lines.append(f"- **Total Issues:** {total_issues}")
    lines.append(f"- **High Severity:** {high} 🔴")
    lines.append(f"- **Medium Severity:** {medium} 🟡")
    lines.append(f"- **Low Severity:** {low} 🟢")
    lines.append("")
    
    # Quality Issues
    if quality_issues:
        lines.append("## Code Quality")
        lines.append("")
        for issue_type, issues in quality_issues.items():
            lines.append(f"### {issue_type.replace('_', ' ').title()} ({len(issues)})")
            lines.append("")
            for i in issues[:5]:
                sev = "🔴" if i['severity'] == 'high' else "🟡" if i['severity'] == 'medium' else "🟢"
                location = f"`{i['path']}::{i['name']}`" if 'name' in i else f"`{i['path']}`"
                lines.append(f"- {sev} {location}")
                lines.append(f"  - {i['suggestion']}")
            if len(issues) > 5:
                lines.append(f"  - ... and {len(issues) - 5} more")
            lines.append("")
    
    # Security Issues
    if security_issues:
        lines.append("## Security Concerns")
        lines.append("")
        for issue_type, issues in security_issues.items():
            lines.append(f"### {issue_type.replace('_', ' ').title()} ({len(issues)})")
            lines.append("")
            for i in issues[:5]:
                sev = "🔴" if i['severity'] == 'high' else "🟡"
                location = f"`{i['path']}::{i['name']}`" if 'name' in i else f"`{i['path']}`"
                lines.append(f"- {sev} {location}")
                lines.append(f"  - Pattern: `{i.get('pattern', 'N/A')}`")
                lines.append(f"  - {i['suggestion']}")
            lines.append("")
    
    # Performance Issues
    if performance_issues:
        lines.append("## Performance")
        lines.append("")
        for issue_type, issues in performance_issues.items():
            lines.append(f"### {issue_type.replace('_', ' ').title()} ({len(issues)})")
            lines.append("")
            for i in issues[:5]:
                sev = "🔴" if i['severity'] == 'high' else "🟡"
                location = f"`{i['path']}::{i['name']}`" if 'name' in i else f"`{i['path']}`"
                lines.append(f"- {sev} {location}")
                lines.append(f"  - {i['suggestion']}")
            lines.append("")
    
    return '\n'.join(lines)


def get_llm_review(project, quality_issues, security_issues) -> str:
    """Get LLM-powered code review."""
    csv_gen = CSVGenerator()
    context = csv_gen.generate(project, detail='standard')[:4000]
    
    issues_summary = []
    for issue_type, issues in quality_issues.items():
        if issues:
            issues_summary.append(f"- {issue_type}: {len(issues)} issues")
    for issue_type, issues in security_issues.items():
        if issues:
            issues_summary.append(f"- {issue_type}: {len(issues)} potential issues")
    
    system = """You are an expert code reviewer. Analyze the code structure and identified issues.
Provide specific, actionable recommendations prioritized by impact.
Focus on the most critical issues first."""

    prompt = f"""Review this codebase:

{context}

Issues identified:
{chr(10).join(issues_summary)}

Provide:
1. Top 3 critical issues to fix immediately
2. Architectural improvements (if any)
3. Quick wins (easy fixes with high impact)"""

    return generate_with_ollama(prompt, system)


def main():
    """Run code review."""
    if len(sys.argv) < 2:
        print("Usage: python code_review.py /path/to/project [options]")
        print("")
        print("Options:")
        print("  --strict           Include low-severity issues")
        print("  --focus AREA       Focus area: quality, security, performance, all")
        print("  --output FILE      Output file (default: code_review.md)")
        print("  --no-llm           Skip LLM analysis")
        sys.exit(1)
    
    project_path = sys.argv[1]
    strict = '--strict' in sys.argv
    focus = 'all'
    output_file = 'code_review.md'
    use_llm = '--no-llm' not in sys.argv
    
    if '--focus' in sys.argv:
        idx = sys.argv.index('--focus')
        focus = sys.argv[idx + 1]
    
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        output_file = sys.argv[idx + 1]
    
    print(f"Analyzing: {project_path}")
    project = analyze_project(project_path)
    print(f"Found {project.total_files} files, {project.total_lines} lines")
    
    print("\nRunning code review...")
    
    quality_issues = {}
    security_issues = {}
    performance_issues = {}
    
    if focus in ('quality', 'all'):
        quality_issues = analyze_code_quality(project)
    
    if focus in ('security', 'all'):
        security_issues = analyze_security(project)
    
    if focus in ('performance', 'all'):
        performance_issues = analyze_performance(project)
    
    # Generate report
    report = generate_review_report(quality_issues, security_issues, performance_issues, project.name)
    
    # LLM review
    if use_llm and check_ollama():
        print("\nGetting LLM review...")
        llm_review = get_llm_review(project, quality_issues, security_issues)
        report += "\n## LLM Review\n\n" + llm_review
    elif use_llm:
        print("\n⚠️  Ollama not running. Skipping LLM review.")
    
    # Save report
    Path(output_file).write_text(report)
    print(f"\nReport saved to: {output_file}")
    
    # Summary
    total = sum(len(v) for v in quality_issues.values())
    total += sum(len(v) for v in security_issues.values())
    total += sum(len(v) for v in performance_issues.values())
    
    print(f"\nTotal issues found: {total}")


if __name__ == '__main__':
    main()
