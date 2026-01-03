"""
Command-line interface for Code2Logic.

Usage:
    code2logic /path/to/project
    code2logic /path/to/project -f csv -o output.csv
    code2logic /path/to/project -f yaml
    code2logic /path/to/project -f json --flat
"""

import argparse
import os
import sys
import subprocess

from . import __version__


def ensure_dependencies():
    """Auto-install optional dependencies for best results."""
    packages = {
        'tree-sitter': 'tree_sitter',
        'tree-sitter-python': 'tree_sitter_python', 
        'tree-sitter-javascript': 'tree_sitter_javascript',
        'tree-sitter-typescript': 'tree_sitter_typescript',
        'networkx': 'networkx',
        'rapidfuzz': 'rapidfuzz',
        'pyyaml': 'yaml',
    }
    
    missing = []
    for pkg_name, import_name in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)
    
    if missing:
        print(f"Installing dependencies for best results: {', '.join(missing)}", file=sys.stderr)
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-q',
                '--break-system-packages', *missing
            ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            print("Dependencies installed successfully!", file=sys.stderr)
        except subprocess.CalledProcessError:
            # Try without --break-system-packages
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', '-q', *missing
                ], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                print("Dependencies installed successfully!", file=sys.stderr)
            except subprocess.CalledProcessError:
                print(f"Warning: Could not install some dependencies. "
                      f"Install manually: pip install {' '.join(missing)}", file=sys.stderr)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='code2logic',
        description='Convert source code to logical representation for LLM analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  code2logic /path/to/project                    # Standard Markdown
  code2logic /path/to/project -f csv             # CSV (best for LLM, ~50%% smaller)
  code2logic /path/to/project -f yaml            # YAML (human-readable)
  code2logic /path/to/project -f json --flat     # Flat JSON (for comparisons)
  code2logic /path/to/project -f compact         # Ultra-compact text

Output formats (token efficiency):
  csv      - Best for LLM (~20K tokens/100 files) - flat table
  compact  - Good for LLM (~25K tokens/100 files) - minimal text
  json     - Standard (~35K tokens/100 files) - nested/flat
  yaml     - Readable (~35K tokens/100 files) - nested/flat  
  markdown - Documentation (~55K tokens/100 files)

Detail levels (columns in csv/json/yaml):
  minimal  - path, type, name, signature (4 columns)
  standard - + intent, category, domain, imports (8 columns)
  full     - + calls, lines, complexity, hash (16 columns)
'''
    )
    
    parser.add_argument(
        'path',
        nargs='?',
        default=None,
        help='Path to the project directory'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['markdown', 'compact', 'json', 'yaml', 'csv', 'gherkin'],
        default='markdown',
        help='Output format (default: markdown)'
    )
    parser.add_argument(
        '-d', '--detail',
        choices=['minimal', 'standard', 'full'],
        default='standard',
        help='Detail level - columns to include (default: standard)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: stdout)'
    )
    parser.add_argument(
        '--flat',
        action='store_true',
        help='Use flat structure (for json/yaml) - better for comparisons'
    )
    parser.add_argument(
        '--no-install',
        action='store_true',
        help='Skip auto-installation of dependencies'
    )
    parser.add_argument(
        '--no-treesitter',
        action='store_true',
        help='Disable Tree-sitter (use fallback parser)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show library availability status and exit'
    )
    
    args = parser.parse_args()
    
    # Auto-install dependencies unless disabled
    if not args.no_install and not args.status:
        ensure_dependencies()
    
    # Import after potential installation
    from .analyzer import ProjectAnalyzer, get_library_status
    from .generators import (
        MarkdownGenerator, CompactGenerator, JSONGenerator,
        YAMLGenerator, CSVGenerator
    )
    from .gherkin import GherkinGenerator
    
    # Status check
    if args.status:
        status = get_library_status()
        print("Library Status:")
        for lib, available in status.items():
            symbol = "✓" if available else "✗"
            print(f"  {lib}: {symbol}")
        sys.exit(0)
    
    # Path is required for analysis
    if args.path is None:
        print("Error: path is required", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # Validate path
    if not os.path.exists(args.path):
        print(f"Error: Path does not exist: {args.path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isdir(args.path):
        print(f"Error: Path is not a directory: {args.path}", file=sys.stderr)
        sys.exit(1)
    
    # Analyze
    if args.verbose:
        print(f"Analyzing project: {args.path}", file=sys.stderr)
    
    analyzer = ProjectAnalyzer(
        args.path,
        use_treesitter=not args.no_treesitter,
        verbose=args.verbose
    )
    project = analyzer.analyze()
    
    if args.verbose:
        print(f"Found {project.total_files} files, {project.total_lines} lines", file=sys.stderr)
    
    # Generate output
    if args.format == 'markdown':
        generator = MarkdownGenerator()
        output = generator.generate(project, args.detail)
    elif args.format == 'compact':
        generator = CompactGenerator()
        output = generator.generate(project)
    elif args.format == 'json':
        generator = JSONGenerator()
        output = generator.generate(project, flat=args.flat, detail=args.detail)
    elif args.format == 'yaml':
        generator = YAMLGenerator()
        output = generator.generate(project, flat=args.flat, detail=args.detail)
    elif args.format == 'csv':
        generator = CSVGenerator()
        output = generator.generate(project, detail=args.detail)
    elif args.format == 'gherkin':
        generator = GherkinGenerator()
        output = generator.generate(project, detail=args.detail)
    
    # Write output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        if args.verbose:
            print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
