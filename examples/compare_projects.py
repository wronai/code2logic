#!/usr/bin/env python3
"""
Example: Compare two projects for duplicates and similarities.

This script demonstrates how to:
1. Analyze two projects
2. Compare them for identical elements
3. Find semantic similarities using LLM

Usage:
    python compare_projects.py /path/to/project1 /path/to/project2
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic import analyze_project, CSVGenerator


def compute_hash(name: str, signature: str) -> str:
    """Compute hash for function comparison."""
    import hashlib
    content = f"{name}:{signature}"
    return hashlib.md5(content.encode()).hexdigest()[:8]


def extract_elements(project) -> dict:
    """Extract all elements from project with hashes."""
    elements = {}

    for m in project.modules:
        for f in m.functions:
            sig = f"({','.join(f.params)})->{f.return_type or ''}"
            h = compute_hash(f.name, sig)
            elements[h] = {
                'path': m.path,
                'name': f.name,
                'signature': sig,
                'intent': f.intent,
                'type': 'function',
            }

        for c in m.classes:
            for method in c.methods:
                sig = f"({','.join(method.params)})->{method.return_type or ''}"
                h = compute_hash(method.name, sig)
                elements[h] = {
                    'path': m.path,
                    'name': f"{c.name}.{method.name}",
                    'signature': sig,
                    'intent': method.intent,
                    'type': 'method',
                }

    return elements


def compare_projects(project1, project2) -> dict:
    """Compare two projects."""
    elements1 = extract_elements(project1)
    elements2 = extract_elements(project2)

    hashes1 = set(elements1.keys())
    hashes2 = set(elements2.keys())

    identical = hashes1 & hashes2
    only_in_1 = hashes1 - hashes2
    only_in_2 = hashes2 - hashes1

    # Find similar by name (different signature)
    names1 = {e['name'].split('.')[-1]: h for h, e in elements1.items()}
    names2 = {e['name'].split('.')[-1]: h for h, e in elements2.items()}

    similar_names = set(names1.keys()) & set(names2.keys())
    similar_but_different = []

    for name in similar_names:
        h1, h2 = names1[name], names2[name]
        if h1 != h2:  # Same name, different signature
            similar_but_different.append({
                'name': name,
                'project1': elements1[h1],
                'project2': elements2[h2],
            })

    return {
        'project1': {
            'name': project1.name,
            'files': project1.total_files,
            'lines': project1.total_lines,
            'elements': len(elements1),
        },
        'project2': {
            'name': project2.name,
            'files': project2.total_files,
            'lines': project2.total_lines,
            'elements': len(elements2),
        },
        'identical': {
            'count': len(identical),
            'elements': [
                {
                    'hash': h,
                    'project1': elements1[h],
                    'project2': elements2[h],
                }
                for h in list(identical)[:20]
            ]
        },
        'only_in_project1': {
            'count': len(only_in_1),
            'elements': [elements1[h] for h in list(only_in_1)[:20]]
        },
        'only_in_project2': {
            'count': len(only_in_2),
            'elements': [elements2[h] for h in list(only_in_2)[:20]]
        },
        'similar_but_different': {
            'count': len(similar_but_different),
            'elements': similar_but_different[:20]
        }
    }


def print_comparison(result: dict):
    """Print comparison results."""
    print("\n" + "=" * 70)
    print("PROJECT COMPARISON RESULTS")
    print("=" * 70)

    p1 = result['project1']
    p2 = result['project2']

    print(f"\nProject 1: {p1['name']}")
    print(f"  Files: {p1['files']}, Lines: {p1['lines']}, Elements: {p1['elements']}")

    print(f"\nProject 2: {p2['name']}")
    print(f"  Files: {p2['files']}, Lines: {p2['lines']}, Elements: {p2['elements']}")

    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)

    identical = result['identical']
    print(f"\n✓ Identical elements: {identical['count']}")
    if identical['elements']:
        print("  Top matches:")
        for e in identical['elements'][:5]:
            p1_e = e['project1']
            p2_e = e['project2']
            print(f"    - {p1_e['name']}: {p1_e['path']} <-> {p2_e['path']}")

    only1 = result['only_in_project1']
    print(f"\n→ Only in Project 1: {only1['count']}")
    if only1['elements']:
        for e in only1['elements'][:5]:
            print(f"    - {e['name']} ({e['path']})")

    only2 = result['only_in_project2']
    print(f"\n← Only in Project 2: {only2['count']}")
    if only2['elements']:
        for e in only2['elements'][:5]:
            print(f"    - {e['name']} ({e['path']})")

    similar = result['similar_but_different']
    print(f"\n~ Similar but different: {similar['count']}")
    if similar['elements']:
        print("  (Same name, different signature)")
        for e in similar['elements'][:5]:
            print(f"    - {e['name']}:")
            print(f"        P1: {e['project1']['signature']}")
            print(f"        P2: {e['project2']['signature']}")

    print("\n" + "=" * 70)

    # Summary
    total_overlap = identical['count'] + similar['count']
    total_unique = only1['count'] + only2['count']

    print("SUMMARY")
    print("=" * 70)
    print(f"Total overlap: {total_overlap} ({100 * total_overlap / (p1['elements'] + p2['elements']):.1f}% average)")
    print(f"Total unique: {total_unique}")

    if identical['count'] > 10:
        print("\n⚠️  High number of identical functions detected!")
        print("   Consider consolidating shared utilities.")

    if similar['count'] > 5:
        print("\n⚠️  Similar functions with different signatures detected!")
        print("   Review for potential API inconsistencies.")


def main():
    """Main comparison."""
    if len(sys.argv) < 3:
        print("Usage: python compare_projects.py /path/to/project1 /path/to/project2")
        sys.exit(1)

    path1 = sys.argv[1]
    path2 = sys.argv[2]

    print(f"Analyzing project 1: {path1}")
    project1 = analyze_project(path1)

    print(f"Analyzing project 2: {path2}")
    project2 = analyze_project(path2)

    # Compare
    result = compare_projects(project1, project2)

    # Print results
    print_comparison(result)

    # Save to file
    output_file = "comparison_result.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    main()