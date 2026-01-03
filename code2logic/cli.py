"""
Command-line interface for code2logic with auto-dependency installation.
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

from .analyzer import ProjectAnalyzer
from .generators import (
    CSVGenerator,
    YAMLGenerator,
    JSONGenerator,
    CompactGenerator,
    MarkdownGenerator,
)
from .mcp_server import MCPServer

logger = logging.getLogger(__name__)


class DependencyManager:
    """Manages automatic dependency installation."""
    
    REQUIRED_PACKAGES = [
        'networkx',
        'tree-sitter', 
        'tree-sitter-python',
        'tree-sitter-javascript',
        'tree-sitter-java',
        'pyyaml',
        'litellm',
        'ollama',
    ]
    
    @classmethod
    def check_and_install_dependencies(cls) -> None:
        """Check if required packages are installed and install them if needed."""
        missing_packages = []
        
        for package in cls.REQUIRED_PACKAGES:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Installing missing dependencies: {', '.join(missing_packages)}")
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install'
                ] + missing_packages)
                print("Dependencies installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install dependencies: {e}")
                sys.exit(1)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Analyze code projects and generate logical representations'
    )
    
    parser.add_argument(
        'project_path',
        help='Path to the project directory to analyze'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='output',
        help='Output directory or file (default: output)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['csv', 'yaml', 'json', 'compact', 'markdown', 'all'],
        default='json',
        help='Output format (default: json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--mcp',
        action='store_true',
        help='Start MCP server for Claude Desktop integration'
    )
    
    parser.add_argument(
        '--mcp-port',
        type=int,
        default=8080,
        help='Port for MCP server (default: 8080)'
    )
    
    parser.add_argument(
        '--no-install',
        action='store_true',
        help='Skip automatic dependency installation'
    )
    
    return parser


def get_generator(format_name: str) -> 'BaseGenerator':
    """Get the appropriate generator for the format."""
    generators = {
        'csv': CSVGenerator(),
        'yaml': YAMLGenerator(),
        'json': JSONGenerator(),
        'compact': CompactGenerator(),
        'markdown': MarkdownGenerator(),
    }
    
    if format_name not in generators:
        raise ValueError(f"Unsupported format: {format_name}")
    
    return generators[format_name]


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Install dependencies if needed
    if not args.no_install:
        DependencyManager.check_and_install_dependencies()
    
    # Start MCP server if requested
    if args.mcp:
        server = MCPServer(port=args.mcp_port)
        print(f"Starting MCP server on port {args.mcp_port}")
        server.start()
        return
    
    # Validate project path
    project_path = Path(args.project_path)
    if not project_path.exists():
        print(f"Error: Project path '{project_path}' does not exist")
        sys.exit(1)
    
    try:
        # Analyze project
        analyzer = ProjectAnalyzer(str(project_path))
        project = analyzer.analyze()
        
        # Generate output
        if args.format == 'all':
            formats = ['csv', 'yaml', 'json', 'compact', 'markdown']
            for fmt in formats:
                generator = get_generator(fmt)
                output_path = f"{args.output}.{fmt}"
                analyzer.generate_output(generator, output_path)
                print(f"Generated {fmt} output: {output_path}")
        else:
            generator = get_generator(args.format)
            output_path = args.output
            if not output_path.endswith(f'.{args.format}'):
                output_path = f"{output_path}.{args.format}"
            analyzer.generate_output(generator, output_path)
            print(f"Generated {args.format} output: {output_path}")
        
        print(f"Analysis complete! Analyzed {len(project.modules)} modules")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
