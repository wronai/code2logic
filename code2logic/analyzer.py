"""
Main project analyzer for code2logic.

This module contains the core ProjectAnalyzer class that orchestrates
the analysis of code projects using various parsers and generators.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import Project, Module, Function, Class, Dependency
from .parsers import TreeSitterParser, FallbackParser
from .dependency import DependencyAnalyzer
from .similarity import SimilarityDetector
from .generators import BaseGenerator

logger = logging.getLogger(__name__)


class ProjectAnalyzer:
    """Main analyzer class for code projects."""
    
    def __init__(self, project_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the analyzer.
        
        Args:
            project_path: Path to the project directory
            config: Optional configuration dictionary
        """
        self.project_path = Path(project_path)
        self.config = config or {}
        self.project: Optional[Project] = None
        
        # Initialize parsers
        self.tree_sitter_parser = TreeSitterParser()
        self.fallback_parser = FallbackParser()
        
        # Initialize analyzers
        self.dependency_analyzer = DependencyAnalyzer()
        self.similarity_detector = SimilarityDetector()
        
    def analyze(self) -> Project:
        """
        Analyze the project and return a Project object.
        
        Returns:
            Project object containing analysis results
        """
        logger.info(f"Starting analysis of project: {self.project_path}")
        
        # Discover source files
        source_files = self._discover_source_files()
        
        # Parse files and extract structure
        modules = []
        for file_path in source_files:
            try:
                module = self._parse_file(file_path)
                if module:
                    modules.append(module)
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")
        
        # Analyze dependencies
        dependencies = self.dependency_analyzer.analyze_dependencies(modules)
        
        # Detect similarities
        similarities = self.similarity_detector.detect_similarities(modules)
        
        # Create project object
        self.project = Project(
            name=self.project_path.name,
            path=str(self.project_path),
            modules=modules,
            dependencies=dependencies,
            similarities=similarities,
            metadata=self._extract_metadata()
        )
        
        logger.info(f"Analysis complete. Found {len(modules)} modules")
        return self.project
    
    def _discover_source_files(self) -> List[Path]:
        """Discover source files in the project."""
        source_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h'}
        source_files = []
        
        for root, dirs, files in os.walk(self.project_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {
                'node_modules', '__pycache__', 'venv', 'env', 'build', 'dist'
            }]
            
            for file in files:
                if Path(file).suffix in source_extensions:
                    source_files.append(Path(root) / file)
        
        return source_files
    
    def _parse_file(self, file_path: Path) -> Optional[Module]:
        """Parse a single file and extract its structure."""
        try:
            # Try Tree-sitter first
            module = self.tree_sitter_parser.parse_file(file_path)
            if module:
                return module
        except Exception:
            pass
        
        # Fallback to simple parser
        try:
            return self.fallback_parser.parse_file(file_path)
        except Exception as e:
            logger.error(f"Failed to parse {file_path} with fallback parser: {e}")
            return None
    
    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract project metadata."""
        metadata = {}
        
        # Look for common project files
        for filename in ['pyproject.toml', 'package.json', 'requirements.txt']:
            file_path = self.project_path / filename
            if file_path.exists():
                metadata[filename] = self._read_project_file(file_path)
        
        return metadata
    
    def _read_project_file(self, file_path: Path) -> Dict[str, Any]:
        """Read and parse project configuration files."""
        # Simple implementation - can be extended
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {'content': content[:1000]}  # Truncate for metadata
        except Exception:
            return {}
    
    def generate_output(self, generator: BaseGenerator, output_path: str) -> None:
        """
        Generate output using the specified generator.
        
        Args:
            generator: Generator instance to use
            output_path: Path for the output file
        """
        if not self.project:
            raise ValueError("Project not analyzed yet. Call analyze() first.")
        
        generator.generate(self.project, output_path)
        logger.info(f"Generated output: {output_path}")
