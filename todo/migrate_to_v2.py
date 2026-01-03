#!/usr/bin/env python3
"""
Code2Logic Migration Script v1 → v2

Reorganizes flat structure into modular architecture.
Run from project root: python migrate_to_v2.py

Safe operations:
- Creates new directory structure
- Copies files to new locations
- Does NOT delete original files
- Generates migration report
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Migration mapping: old_file → new_location
MIGRATION_MAP = {
    # Core
    'models.py': 'core/models.py',
    'analyzer.py': 'core/analyzer.py',
    'dependency.py': 'core/dependency.py',
    'parsers.py': 'core/parsers/legacy.py',  # Will be split later
    'base.py': 'core/parsers/base.py',
    
    # Formats
    'generators.py': 'formats/legacy_generators.py',  # Will be split
    'gherkin.py': 'formats/gherkin.py',
    'markdown_format.py': 'formats/markdown.py',
    'file_formats.py': 'formats/file_formats.py',
    
    # Reproduction
    'reproduction.py': 'reproduction/reproducer.py',
    'chunked_reproduction.py': 'reproduction/chunked.py',
    'project_reproducer.py': 'reproduction/project.py',
    'metrics.py': 'reproduction/metrics.py',
    'similarity.py': 'reproduction/similarity.py',
    'universal.py': 'reproduction/universal.py',
    
    # LLM
    'llm_clients.py': 'llm/clients.py',
    'llm.py': 'llm/legacy.py',
    'intent.py': 'llm/intent.py',
    
    # Tools
    'benchmark.py': 'tools/benchmark.py',
    'code_review.py': 'tools/review.py',
    'refactor.py': 'tools/refactor.py',
    'adaptive.py': 'tools/adaptive.py',
    
    # Integrations
    'mcp_server.py': 'integrations/mcp.py',
    
    # CLI
    'cli.py': 'cli/main.py',
    
    # Config
    'config.py': 'config.py',  # Stays at root
}

# New directories to create
NEW_DIRS = [
    'core',
    'core/parsers',
    'formats',
    'reproduction',
    'llm',
    'tools',
    'integrations',
    'cli',
    'cli/commands',
    'tests',
    'tests/core',
    'tests/formats',
    'tests/reproduction',
]

# Init files content
INIT_TEMPLATES = {
    'core/__init__.py': '''"""Core analysis components."""
from .models import ProjectInfo, ModuleInfo, FunctionInfo, ClassInfo
from .analyzer import ProjectAnalyzer

__all__ = ['ProjectInfo', 'ModuleInfo', 'FunctionInfo', 'ClassInfo', 'ProjectAnalyzer']
''',
    
    'formats/__init__.py': '''"""Output format generators."""
from .base import BaseGenerator, FormatSpec, FORMATS, register_format

# Import all generators to register them
from .logicml import LogicMLGenerator
from .yaml import YAMLGenerator
from .gherkin import GherkinGenerator
from .markdown import MarkdownGenerator

DEFAULT_FORMAT = 'logicml'

def get_generator(format_name: str = None) -> BaseGenerator:
    """Get format generator by name."""
    name = format_name or DEFAULT_FORMAT
    if name not in FORMATS:
        raise ValueError(f"Unknown format: {name}. Available: {list(FORMATS.keys())}")
    return FORMATS[name]()

__all__ = ['BaseGenerator', 'FormatSpec', 'FORMATS', 'get_generator', 'DEFAULT_FORMAT']
''',
    
    'reproduction/__init__.py': '''"""Code reproduction from logic specifications."""
from .reproducer import UniversalReproducer
from .project import ProjectReproducer

__all__ = ['UniversalReproducer', 'ProjectReproducer']
''',
    
    'llm/__init__.py': '''"""LLM client integrations."""
from .base import BaseLLMClient
from .clients import get_client

__all__ = ['BaseLLMClient', 'get_client']
''',
    
    'tools/__init__.py': '''"""Development tools and utilities."""
from .benchmark import run_benchmark
from .review import CodeReviewer

__all__ = ['run_benchmark', 'CodeReviewer']
''',
    
    'cli/__init__.py': '''"""Command-line interface."""
from .main import main

__all__ = ['main']
''',
    
    'cli/commands/__init__.py': '''"""CLI commands."""
''',
}

# Base generator template
BASE_GENERATOR_TEMPLATE = '''"""Base class for format generators."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type, TypeVar, Generic, Any

try:
    from ..core.models import ProjectInfo
except ImportError:
    ProjectInfo = Any

T = TypeVar('T')


@dataclass
class FormatSpec(Generic[T]):
    """Base specification output."""
    content: str
    token_estimate: int
    metadata: T = None


class BaseGenerator(ABC):
    """Abstract base for all format generators."""
    
    FORMAT_NAME: str = "base"
    FILE_EXTENSION: str = ".txt"
    TOKEN_EFFICIENCY: float = 1.0  # Relative to YAML baseline
    REPRODUCTION_FIDELITY: float = 0.9  # Expected 0.0 - 1.0
    
    @abstractmethod
    def generate(self, project: 'ProjectInfo', detail: str = 'standard') -> FormatSpec:
        """Generate format specification.
        
        Args:
            project: Analyzed project info
            detail: Detail level ('minimal', 'standard', 'full')
            
        Returns:
            FormatSpec with content and metadata
        """
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // 4


# Format registry
FORMATS: Dict[str, Type[BaseGenerator]] = {}


def register_format(cls: Type[BaseGenerator]) -> Type[BaseGenerator]:
    """Decorator to register a format generator.
    
    Usage:
        @register_format
        class MyGenerator(BaseGenerator):
            FORMAT_NAME = "myformat"
    """
    if hasattr(cls, 'FORMAT_NAME'):
        FORMATS[cls.FORMAT_NAME] = cls
    return cls
'''


def create_directory_structure(base_path: Path):
    """Create new directory structure."""
    print("📁 Creating directory structure...")
    
    for dir_name in NEW_DIRS:
        dir_path = base_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {dir_name}/")


def create_init_files(base_path: Path):
    """Create __init__.py files."""
    print("\n📝 Creating __init__.py files...")
    
    for file_path, content in INIT_TEMPLATES.items():
        full_path = base_path / file_path
        full_path.write_text(content)
        print(f"   ✓ {file_path}")


def create_base_generator(base_path: Path):
    """Create base generator class."""
    print("\n🔧 Creating base generator...")
    
    base_file = base_path / 'formats' / 'base.py'
    base_file.write_text(BASE_GENERATOR_TEMPLATE)
    print(f"   ✓ formats/base.py")


def migrate_files(source_path: Path, target_path: Path):
    """Migrate files to new locations."""
    print("\n📦 Migrating files...")
    
    migrated = []
    skipped = []
    
    for old_name, new_path in MIGRATION_MAP.items():
        old_file = source_path / old_name
        new_file = target_path / new_path
        
        if old_file.exists():
            # Ensure target directory exists
            new_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(old_file, new_file)
            migrated.append((old_name, new_path))
            print(f"   ✓ {old_name} → {new_path}")
        else:
            skipped.append(old_name)
    
    return migrated, skipped


def generate_report(base_path: Path, migrated: list, skipped: list):
    """Generate migration report."""
    report = f"""# Code2Logic Migration Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- Files migrated: {len(migrated)}
- Files skipped (not found): {len(skipped)}

## Migrated Files

| Original | New Location |
|----------|--------------|
"""
    
    for old, new in migrated:
        report += f"| {old} | {new} |\n"
    
    if skipped:
        report += "\n## Skipped Files\n\n"
        for name in skipped:
            report += f"- {name}\n"
    
    report += """
## Next Steps

1. Update imports in migrated files
2. Split legacy files (generators.py, parsers.py) into individual modules
3. Add LogicML generator to formats/
4. Update CLI to use new structure
5. Run tests to verify migration
6. Delete original files after verification

## Import Updates Needed

Replace:
```python
from .models import ProjectInfo
from .generators import YAMLGenerator
from .llm_clients import get_client
```

With:
```python
from .core.models import ProjectInfo
from .formats.yaml import YAMLGenerator
from .llm.clients import get_client
```
"""
    
    report_path = base_path / 'MIGRATION_REPORT.md'
    report_path.write_text(report)
    print(f"\n📋 Report saved: {report_path}")


def main():
    """Run migration."""
    print("=" * 50)
    print("Code2Logic v1 → v2 Migration")
    print("=" * 50)
    
    # Paths
    source_path = Path('.')  # Current directory (original code2logic)
    target_path = Path('code2logic_v2')  # New structure
    
    # Check if source has expected files
    if not (source_path / 'models.py').exists() and not (source_path / 'code2logic').exists():
        print("⚠️  Run this script from the code2logic package directory")
        print("   or the parent directory containing code2logic/")
        return
    
    # Adjust source if needed
    if (source_path / 'code2logic').exists():
        source_path = source_path / 'code2logic'
    
    print(f"\nSource: {source_path.absolute()}")
    print(f"Target: {target_path.absolute()}")
    
    # Create new structure
    target_path.mkdir(exist_ok=True)
    create_directory_structure(target_path)
    create_init_files(target_path)
    create_base_generator(target_path)
    
    # Migrate files
    migrated, skipped = migrate_files(source_path, target_path)
    
    # Generate report
    generate_report(target_path, migrated, skipped)
    
    print("\n" + "=" * 50)
    print("✅ Migration complete!")
    print("=" * 50)
    print(f"\nNext: Review {target_path}/MIGRATION_REPORT.md")


if __name__ == '__main__':
    main()
