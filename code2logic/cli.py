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
import time
import logging
from datetime import datetime

from . import __version__


# Colors for terminal output
class Colors:
    BLUE = '\033[34m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    CYAN = '\033[36m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    NC = '\033[0m'  # No Color


class Logger:
    """Enhanced logger for CLI output."""
    
    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose
        self.debug = debug
        self.start_time = time.time()
        self._step = 0
    
    def _elapsed(self) -> str:
        """Get elapsed time string."""
        elapsed = time.time() - self.start_time
        return f"{elapsed:.2f}s"
    
    def info(self, msg: str):
        """Print info message."""
        print(f"{Colors.BLUE}ℹ{Colors.NC} {msg}", file=sys.stderr)
    
    def success(self, msg: str):
        """Print success message."""
        print(f"{Colors.GREEN}✓{Colors.NC} {msg}", file=sys.stderr)
    
    def warning(self, msg: str):
        """Print warning message."""
        print(f"{Colors.YELLOW}⚠{Colors.NC} {msg}", file=sys.stderr)
    
    def error(self, msg: str):
        """Print error message."""
        print(f"{Colors.RED}✗{Colors.NC} {msg}", file=sys.stderr)
    
    def step(self, msg: str):
        """Print step message with counter."""
        self._step += 1
        if self.verbose:
            print(f"{Colors.CYAN}[{self._step}]{Colors.NC} {msg} {Colors.DIM}({self._elapsed()}){Colors.NC}", file=sys.stderr)
    
    def detail(self, msg: str):
        """Print detail message (only in verbose mode)."""
        if self.verbose:
            print(f"    {Colors.DIM}{msg}{Colors.NC}", file=sys.stderr)
    
    def debug_msg(self, msg: str):
        """Print debug message (only in debug mode)."""
        if self.debug:
            print(f"{Colors.DIM}[DEBUG] {msg}{Colors.NC}", file=sys.stderr)
    
    def stats(self, label: str, value):
        """Print statistics."""
        if self.verbose:
            print(f"    {Colors.BOLD}{label}:{Colors.NC} {value}", file=sys.stderr)
    
    def separator(self):
        """Print separator line."""
        if self.verbose:
            print(f"{Colors.DIM}{'─' * 50}{Colors.NC}", file=sys.stderr)
    
    def header(self, msg: str):
        """Print header."""
        if self.verbose:
            print(f"\n{Colors.BOLD}{Colors.BLUE}{msg}{Colors.NC}", file=sys.stderr)
            print(f"{Colors.DIM}{'═' * len(msg)}{Colors.NC}", file=sys.stderr)


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
    cli_start = time.time()
    
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
        help='Verbose output with progress info'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug output (very verbose)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress all output except errors'
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
    
    # Initialize logger
    log = Logger(verbose=args.verbose, debug=args.debug)
    
    if args.verbose and not args.quiet:
        log.header("CODE2LOGIC")
        log.detail(f"Version: {__version__}")
        log.detail(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto-install dependencies unless disabled
    if not args.no_install and not args.status:
        if args.verbose:
            log.step("Checking dependencies...")
        ensure_dependencies()
        if args.verbose:
            log.detail("Dependencies OK")
    
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
        log.error(f"Path does not exist: {args.path}")
        sys.exit(1)
    
    if not os.path.isdir(args.path):
        log.error(f"Path is not a directory: {args.path}")
        sys.exit(1)
    
    # Analyze
    if args.verbose:
        log.step(f"Analyzing project: {args.path}")
        log.detail(f"Parser: {'Tree-sitter' if not args.no_treesitter else 'Fallback regex'}")
    
    analyze_start = time.time()
    analyzer = ProjectAnalyzer(
        args.path,
        use_treesitter=not args.no_treesitter,
        verbose=args.debug
    )
    project = analyzer.analyze()
    analyze_time = time.time() - analyze_start
    
    if args.verbose:
        log.success(f"Analysis complete ({analyze_time:.2f}s)")
        log.separator()
        log.stats("Files", project.total_files)
        log.stats("Lines", f"{project.total_lines:,}")
        log.stats("Languages", ', '.join(project.languages.keys()))
        log.stats("Modules", len(project.modules))
        
        total_functions = sum(len(m.functions) for m in project.modules)
        total_classes = sum(len(m.classes) for m in project.modules)
        log.stats("Functions", total_functions)
        log.stats("Classes", total_classes)
        
        if project.entrypoints:
            log.stats("Entrypoints", ', '.join(project.entrypoints[:3]))
        
        log.separator()
    
    # Generate output
    if args.verbose:
        log.step(f"Generating {args.format} output (detail: {args.detail})")
    
    gen_start = time.time()
    
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
    
    gen_time = time.time() - gen_start
    
    if args.verbose:
        output_size = len(output)
        tokens_approx = output_size // 4
        log.success(f"Output generated ({gen_time:.2f}s)")
        log.stats("Size", f"{output_size:,} chars (~{tokens_approx:,} tokens)")
        log.stats("Lines", output.count('\n') + 1)
    
    # Write output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        if args.verbose:
            log.success(f"Output written to: {args.output}")
    else:
        if not args.quiet:
            print(output)
    
    # Final summary
    if args.verbose:
        total_time = time.time() - cli_start
        log.separator()
        log.info(f"Total time: {total_time:.2f}s")


if __name__ == '__main__':
    main()
