"""
Main project analyzer orchestrating all analysis components.

Provides the high-level API for analyzing codebases.
"""

import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional, List, Dict

from .models import ProjectInfo, ModuleInfo
from .parsers import TreeSitterParser, UniversalParser, TREE_SITTER_AVAILABLE
from .dependency import DependencyAnalyzer, NETWORKX_AVAILABLE
from .similarity import SimilarityDetector, RAPIDFUZZ_AVAILABLE
from .intent import NLTK_AVAILABLE, SPACY_AVAILABLE


class ProjectAnalyzer:
    """
    Main class for analyzing software projects.
    
    Orchestrates:
    - File scanning and language detection
    - AST parsing (Tree-sitter or fallback)
    - Dependency graph building and analysis
    - Similar function detection
    - Entry point identification
    
    Example:
        >>> analyzer = ProjectAnalyzer("/path/to/project")
        >>> project = analyzer.analyze()
        >>> print(f"Found {project.total_files} files")
        
    With options:
        >>> analyzer = ProjectAnalyzer(
        ...     "/path/to/project",
        ...     use_treesitter=True,
        ...     verbose=True
        ... )
    """
    
    # Language extension mapping
    LANGUAGE_EXTENSIONS: Dict[str, str] = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.c': 'cpp',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.h': 'cpp',
        '.hpp': 'cpp',
        '.php': 'php',
        '.rb': 'ruby',
        '.kt': 'kotlin',
        '.swift': 'swift',
    }
    
    # Directories to ignore
    IGNORE_DIRS: set = {
        '.git', '.svn', '.hg',
        'node_modules', '__pycache__', '.venv', 'venv', 'env',
        'target', 'build', 'dist', 'out', '.next',
        '.idea', '.vscode', '.pytest_cache',
        'vendor', 'packages', '.tox', 'coverage',
        '.mypy_cache', '.ruff_cache', '.cache',
    }
    
    # Files to ignore
    IGNORE_FILES: set = {
        '.gitignore', '.dockerignore', 
        'package-lock.json', 'yarn.lock',
        'Pipfile.lock', 'poetry.lock', 
        'Cargo.lock', 'pnpm-lock.yaml',
    }
    
    def __init__(
        self,
        root_path: str,
        use_treesitter: bool = True,
        verbose: bool = False,
        include_private: bool = False,
    ):
        """
        Initialize the project analyzer.
        
        Args:
            root_path: Path to the project root directory
            use_treesitter: Whether to use Tree-sitter for parsing
            verbose: Whether to print status messages
            include_private: Whether to include private functions/classes
        """
        self.root_path = Path(root_path).resolve()
        self.verbose = verbose
        self.include_private = include_private
        self.modules: List[ModuleInfo] = []
        self.languages: Dict[str, int] = defaultdict(int)
        
        # Initialize parsers
        self.ts_parser = (
            TreeSitterParser() 
            if use_treesitter and TREE_SITTER_AVAILABLE 
            else None
        )
        self.fallback_parser = UniversalParser()
        
        # Initialize analyzers
        self.dep_analyzer = DependencyAnalyzer()
        self.sim_detector = SimilarityDetector()
        
        if verbose:
            self._print_status()
    
    def _print_status(self):
        """Print library availability status."""
        parts = []
        parts.append("TS✓" if TREE_SITTER_AVAILABLE else "TS✗")
        parts.append("NX✓" if NETWORKX_AVAILABLE else "NX✗")
        parts.append("RF✓" if RAPIDFUZZ_AVAILABLE else "RF✗")
        parts.append("NLP✓" if (SPACY_AVAILABLE or NLTK_AVAILABLE) else "NLP✗")
        print(f"Libs: {' '.join(parts)}", file=sys.stderr)
    
    def analyze(self) -> ProjectInfo:
        """
        Analyze the project.
        
        Returns:
            ProjectInfo with complete analysis results
        """
        # Scan and parse files
        self._scan_files()
        
        # Build dependency graph
        dep_graph = self.dep_analyzer.build_graph(self.modules)
        dep_metrics = self.dep_analyzer.analyze_metrics()
        
        # Detect entry points
        entrypoints = self._detect_entrypoints()
        
        # Find similar functions
        similar = self.sim_detector.find_similar_functions(self.modules)
        
        return ProjectInfo(
            name=self.root_path.name,
            root_path=str(self.root_path),
            languages=dict(self.languages),
            modules=self.modules,
            dependency_graph=dep_graph,
            dependency_metrics=dep_metrics,
            entrypoints=entrypoints,
            similar_functions=similar,
            total_files=len(self.modules),
            total_lines=sum(m.lines_total for m in self.modules),
            generated_at=datetime.now().isoformat()
        )
    
    def _scan_files(self):
        """Scan and parse all source files."""
        for fp in self.root_path.rglob('*'):
            if not fp.is_file():
                continue
            
            # Skip ignored directories
            if any(d in fp.parts for d in self.IGNORE_DIRS):
                continue
            
            # Skip ignored files
            if fp.name in self.IGNORE_FILES:
                continue
            
            # Check extension
            ext = fp.suffix.lower()
            if ext not in self.LANGUAGE_EXTENSIONS:
                continue
            
            language = self.LANGUAGE_EXTENSIONS[ext]
            self.languages[language] += 1
            
            # Read file
            try:
                content = fp.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            
            rel_path = str(fp.relative_to(self.root_path))
            
            # Try Tree-sitter first, then fallback
            module = None
            if self.ts_parser and self.ts_parser.is_available(language):
                module = self.ts_parser.parse(rel_path, content, language)
            
            if module is None:
                module = self.fallback_parser.parse(rel_path, content, language)
            
            if module:
                self.modules.append(module)
    
    def _detect_entrypoints(self) -> List[str]:
        """Detect project entry points."""
        eps = []
        
        # From dependency analyzer (nodes with no incoming edges)
        if self.dep_analyzer.graph is not None:
            eps.extend(self.dep_analyzer.get_entrypoints())
        
        # Common entry point file names
        main_files = {
            'main.py', 'app.py', 'server.py', '__main__.py', 'run.py',
            'main.js', 'app.js', 'server.js', 'index.js',
            'main.ts', 'app.ts', 'server.ts', 'index.ts',
            'main.go', 'main.rs', 'Main.java',
        }
        
        for m in self.modules:
            fn = Path(m.path).name
            parent = str(Path(m.path).parent)
            
            if fn in main_files and m.path not in eps:
                eps.append(m.path)
            elif fn in ('index.js', 'index.ts') and parent in ('.', 'src') and m.path not in eps:
                eps.append(m.path)
        
        return eps[:10]
    
    def get_statistics(self) -> Dict:
        """
        Get analysis statistics.
        
        Returns:
            Dict with analysis statistics
        """
        return {
            'total_files': len(self.modules),
            'total_lines': sum(m.lines_total for m in self.modules),
            'total_code_lines': sum(m.lines_code for m in self.modules),
            'languages': dict(self.languages),
            'total_classes': sum(len(m.classes) for m in self.modules),
            'total_functions': sum(len(m.functions) for m in self.modules),
        }


def analyze_project(
    path: str,
    use_treesitter: bool = True,
    verbose: bool = False,
) -> ProjectInfo:
    """
    Convenience function to analyze a project.
    
    Args:
        path: Path to the project directory
        use_treesitter: Whether to use Tree-sitter for parsing
        verbose: Whether to print status messages
        
    Returns:
        ProjectInfo with analysis results
        
    Example:
        >>> from code2logic import analyze_project
        >>> project = analyze_project("/path/to/project")
        >>> print(f"Analyzed {project.total_files} files")
    """
    analyzer = ProjectAnalyzer(path, use_treesitter=use_treesitter, verbose=verbose)
    return analyzer.analyze()


def get_library_status() -> Dict[str, bool]:
    """
    Get availability status of optional libraries.
    
    Returns:
        Dict mapping library name to availability status
    """
    return {
        'tree_sitter': TREE_SITTER_AVAILABLE,
        'networkx': NETWORKX_AVAILABLE,
        'rapidfuzz': RAPIDFUZZ_AVAILABLE,
        'nltk': NLTK_AVAILABLE,
        'spacy': SPACY_AVAILABLE,
    }
