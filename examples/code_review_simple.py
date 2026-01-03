#!/usr/bin/env python3
"""
Code Review Example (Simplified).

Uses code2logic's built-in code review utilities.

Usage:
    python code_review_simple.py /path/to/project
    python code_review_simple.py /path/to/project --focus security
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic import analyze_project, CodeReviewer


def main():
    parser = argparse.ArgumentParser(description='Automated code review')
    parser.add_argument('project', help='Project path to review')
    parser.add_argument('--focus', choices=['all', 'quality', 'security', 'performance'], 
                        default='all', help='Review focus')
    parser.add_argument('--output', '-o', help='Output file for report')
    args = parser.parse_args()
    
    print(f"Analyzing {args.project}...")
    project = analyze_project(args.project)
    
    print(f"Reviewing {project.total_files} files...")
    reviewer = CodeReviewer()
    results = reviewer.review(project, focus=args.focus)
    
    report = reviewer.generate_report(results, project.name)
    
    if args.output:
        Path(args.output).write_text(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)
    
    # Summary
    summary = results['summary']
    print(f"\n{'='*40}")
    print(f"Total issues: {summary['total_issues']}")
    print(f"  High: {summary['by_severity'].get('high', 0)}")
    print(f"  Medium: {summary['by_severity'].get('medium', 0)}")
    print(f"  Low: {summary['by_severity'].get('low', 0)}")


if __name__ == '__main__':
    main()
