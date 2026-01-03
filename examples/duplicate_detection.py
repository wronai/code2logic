#!/usr/bin/env python3
"""
Example: Duplicate Detection and Deduplication Report.

Finds duplicates in codebases using multiple strategies:
1. Hash-based (exact duplicates)
2. Signature-based (same interface, different names)
3. Intent-based (semantic duplicates)
4. Category+Domain (consolidation candidates)

Generates actionable deduplication report for refactoring.

Usage:
    python duplicate_detection.py /path/to/project
    python duplicate_detection.py /path/to/project --threshold 0.8
    python duplicate_detection.py /path/to/project --output report.md
"""

import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic import analyze_project, CSVGenerator


def compute_signature_hash(params: List[str], return_type: str) -> str:
    """Compute hash of function signature (ignoring name)."""
    # Normalize params (remove names, keep types)
    normalized_params = []
    for p in params:
        if ':' in p:
            normalized_params.append(p.split(':')[1].strip())
        else:
            normalized_params.append(p)
    
    sig = f"({','.join(normalized_params)})->{return_type or 'void'}"
    return hashlib.md5(sig.encode()).hexdigest()[:8]


def compute_name_hash(name: str, params: List[str]) -> str:
    """Compute hash for exact duplicate detection."""
    sig = f"{name}({','.join(params)})"
    return hashlib.md5(sig.encode()).hexdigest()[:8]


def normalize_intent(intent: str) -> str:
    """Normalize intent for comparison."""
    if not intent:
        return ''
    
    # Lowercase, remove punctuation, normalize whitespace
    intent = intent.lower().strip()
    intent = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in intent)
    intent = ' '.join(intent.split())
    
    # Remove common words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall'}
    
    words = [w for w in intent.split() if w not in stop_words]
    return ' '.join(words[:10])  # Keep first 10 meaningful words


def find_duplicates(project) -> Dict[str, Any]:
    """Find all types of duplicates in project."""
    results = {
        'exact_duplicates': [],
        'signature_duplicates': [],
        'intent_duplicates': [],
        'consolidation_candidates': [],
        'statistics': {},
    }
    
    # Collect all functions with metadata
    functions = []
    
    for module in project.modules:
        for func in module.functions:
            functions.append({
                'path': module.path,
                'name': func.name,
                'full_name': f"{module.path}::{func.name}",
                'params': func.params,
                'return_type': func.return_type,
                'intent': func.intent,
                'lines': func.lines,
                'is_async': func.is_async,
                'category': categorize(func.name),
                'domain': extract_domain(module.path),
            })
        
        for cls in module.classes:
            for method in cls.methods:
                functions.append({
                    'path': module.path,
                    'name': method.name,
                    'full_name': f"{module.path}::{cls.name}.{method.name}",
                    'class': cls.name,
                    'params': method.params,
                    'return_type': method.return_type,
                    'intent': method.intent,
                    'lines': method.lines,
                    'is_async': method.is_async,
                    'category': categorize(method.name),
                    'domain': extract_domain(module.path),
                })
    
    # 1. Exact duplicates (same name + signature)
    name_hash_groups = defaultdict(list)
    for f in functions:
        h = compute_name_hash(f['name'], f['params'])
        name_hash_groups[h].append(f)
    
    for h, group in name_hash_groups.items():
        if len(group) > 1:
            results['exact_duplicates'].append({
                'hash': h,
                'count': len(group),
                'functions': [f['full_name'] for f in group],
                'suggestion': 'Extract to shared utility',
                'effort': 'low',
            })
    
    # 2. Signature duplicates (same signature, different names)
    sig_hash_groups = defaultdict(list)
    for f in functions:
        h = compute_signature_hash(f['params'], f['return_type'])
        sig_hash_groups[h].append(f)
    
    for h, group in sig_hash_groups.items():
        if len(group) > 2:
            # Get unique names
            names = set(f['name'].split('.')[-1] for f in group)
            if len(names) > 1:  # Different names with same signature
                results['signature_duplicates'].append({
                    'signature_hash': h,
                    'count': len(group),
                    'names': list(names)[:5],
                    'functions': [f['full_name'] for f in group[:10]],
                    'suggestion': 'Consider generic implementation',
                    'effort': 'medium',
                })
    
    # 3. Intent duplicates (semantic similarity)
    intent_groups = defaultdict(list)
    for f in functions:
        if f['intent']:
            normalized = normalize_intent(f['intent'])
            if normalized:
                intent_groups[normalized].append(f)
    
    for intent, group in intent_groups.items():
        if len(group) > 1:
            results['intent_duplicates'].append({
                'intent': intent[:50],
                'count': len(group),
                'functions': [f['full_name'] for f in group[:10]],
                'suggestion': 'Review for consolidation',
                'effort': 'high',
            })
    
    # 4. Consolidation candidates (same category + domain)
    cat_domain_groups = defaultdict(list)
    for f in functions:
        key = f"{f['category']}:{f['domain']}"
        cat_domain_groups[key].append(f)
    
    for key, group in cat_domain_groups.items():
        if len(group) > 5:
            category, domain = key.split(':')
            results['consolidation_candidates'].append({
                'category': category,
                'domain': domain,
                'count': len(group),
                'functions': [f['name'] for f in group[:10]],
                'suggestion': f'Consider {category} service for {domain}',
                'effort': 'high',
            })
    
    # Statistics
    results['statistics'] = {
        'total_functions': len(functions),
        'exact_duplicate_groups': len(results['exact_duplicates']),
        'exact_duplicate_functions': sum(d['count'] for d in results['exact_duplicates']),
        'signature_duplicate_groups': len(results['signature_duplicates']),
        'intent_duplicate_groups': len(results['intent_duplicates']),
        'consolidation_areas': len(results['consolidation_candidates']),
    }
    
    return results


def categorize(name: str) -> str:
    """Categorize function by name."""
    name_lower = name.lower()
    
    if any(v in name_lower for v in ('get', 'fetch', 'find', 'load', 'read')):
        return 'read'
    if any(v in name_lower for v in ('create', 'add', 'insert', 'new')):
        return 'create'
    if any(v in name_lower for v in ('update', 'set', 'modify')):
        return 'update'
    if any(v in name_lower for v in ('delete', 'remove', 'clear')):
        return 'delete'
    if any(v in name_lower for v in ('validate', 'check', 'verify')):
        return 'validate'
    if any(v in name_lower for v in ('convert', 'transform', 'parse')):
        return 'transform'
    
    return 'other'


def extract_domain(path: str) -> str:
    """Extract domain from path."""
    parts = path.lower().replace('\\', '/').split('/')
    domains = ['auth', 'user', 'order', 'payment', 'api', 'service',
               'model', 'validation', 'generator', 'parser', 'test']
    
    for part in parts:
        for domain in domains:
            if domain in part:
                return domain
    
    return parts[-2] if len(parts) > 1 else 'core'


def generate_report(results: Dict[str, Any], project_name: str) -> str:
    """Generate markdown deduplication report."""
    lines = [
        f"# Duplicate Detection Report: {project_name}",
        "",
        "## Summary",
        "",
        f"- Total functions analyzed: {results['statistics']['total_functions']}",
        f"- Exact duplicate groups: {results['statistics']['exact_duplicate_groups']}",
        f"- Exact duplicate functions: {results['statistics']['exact_duplicate_functions']}",
        f"- Signature duplicate groups: {results['statistics']['signature_duplicate_groups']}",
        f"- Intent duplicate groups: {results['statistics']['intent_duplicate_groups']}",
        f"- Consolidation areas: {results['statistics']['consolidation_areas']}",
        "",
    ]
    
    # Exact duplicates
    if results['exact_duplicates']:
        lines.append("## Exact Duplicates (HIGH PRIORITY)")
        lines.append("")
        lines.append("These functions have identical names and signatures in multiple locations.")
        lines.append("")
        
        for dup in results['exact_duplicates'][:10]:
            lines.append(f"### Hash: {dup['hash']} ({dup['count']} occurrences)")
            lines.append(f"**Suggestion:** {dup['suggestion']} (Effort: {dup['effort']})")
            lines.append("")
            for f in dup['functions'][:5]:
                lines.append(f"- `{f}`")
            lines.append("")
    
    # Signature duplicates
    if results['signature_duplicates']:
        lines.append("## Signature Duplicates (MEDIUM PRIORITY)")
        lines.append("")
        lines.append("Different function names with identical signatures - potential for generic implementation.")
        lines.append("")
        
        for dup in results['signature_duplicates'][:10]:
            lines.append(f"### {', '.join(dup['names'])} ({dup['count']} occurrences)")
            lines.append(f"**Suggestion:** {dup['suggestion']} (Effort: {dup['effort']})")
            lines.append("")
            for f in dup['functions'][:5]:
                lines.append(f"- `{f}`")
            lines.append("")
    
    # Intent duplicates
    if results['intent_duplicates']:
        lines.append("## Semantic Duplicates (REVIEW)")
        lines.append("")
        lines.append("Functions with similar business intent - may be doing the same thing.")
        lines.append("")
        
        for dup in results['intent_duplicates'][:10]:
            lines.append(f"### Intent: \"{dup['intent']}...\" ({dup['count']} occurrences)")
            lines.append(f"**Suggestion:** {dup['suggestion']} (Effort: {dup['effort']})")
            lines.append("")
            for f in dup['functions'][:5]:
                lines.append(f"- `{f}`")
            lines.append("")
    
    # Consolidation candidates
    if results['consolidation_candidates']:
        lines.append("## Consolidation Opportunities")
        lines.append("")
        lines.append("Areas with many related functions that could be organized better.")
        lines.append("")
        
        for area in sorted(results['consolidation_candidates'], 
                          key=lambda x: -x['count'])[:10]:
            lines.append(f"### {area['category'].title()} operations in {area['domain']} ({area['count']} functions)")
            lines.append(f"**Suggestion:** {area['suggestion']}")
            lines.append("")
            lines.append("Functions: " + ', '.join(area['functions'][:5]))
            lines.append("")
    
    # Action items
    lines.append("## Recommended Actions")
    lines.append("")
    lines.append("1. **Immediate:** Fix exact duplicates - extract to shared utilities")
    lines.append("2. **Short-term:** Review signature duplicates for generic implementations")
    lines.append("3. **Long-term:** Consolidate related functions into services")
    lines.append("")
    
    return '\n'.join(lines)


def main():
    """Run duplicate detection."""
    if len(sys.argv) < 2:
        print("Usage: python duplicate_detection.py /path/to/project [options]")
        print("")
        print("Options:")
        print("  --output FILE    Output report file (default: stdout)")
        print("  --json           Output as JSON instead of Markdown")
        sys.exit(1)
    
    project_path = sys.argv[1]
    output_file = None
    as_json = '--json' in sys.argv
    
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        output_file = sys.argv[idx + 1]
    
    print(f"Analyzing: {project_path}", file=sys.stderr)
    
    # Analyze project
    project = analyze_project(project_path)
    print(f"Found {project.total_files} files, {project.total_lines} lines", file=sys.stderr)
    
    # Find duplicates
    print("Detecting duplicates...", file=sys.stderr)
    results = find_duplicates(project)
    
    # Generate output
    if as_json:
        output = json.dumps(results, indent=2)
    else:
        output = generate_report(results, project.name)
    
    # Write output
    if output_file:
        Path(output_file).write_text(output)
        print(f"Report saved to: {output_file}", file=sys.stderr)
    else:
        print(output)
    
    # Summary
    stats = results['statistics']
    print(f"\nDuplicate Summary:", file=sys.stderr)
    print(f"  Exact duplicates: {stats['exact_duplicate_functions']}", file=sys.stderr)
    print(f"  Signature duplicates: {stats['signature_duplicate_groups']} groups", file=sys.stderr)
    print(f"  Semantic duplicates: {stats['intent_duplicate_groups']} groups", file=sys.stderr)


if __name__ == '__main__':
    main()
