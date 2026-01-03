"""
Tests for the ProjectAnalyzer class.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from code2logic.analyzer import ProjectAnalyzer, analyze_project, get_library_status
from code2logic.models import ProjectInfo, ModuleInfo


class TestProjectAnalyzer:
    """Test cases for ProjectAnalyzer."""
    
    def test_init(self, temp_project_dir):
        """Test ProjectAnalyzer initialization."""
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        
        assert analyzer.root_path == Path(temp_project_dir).resolve()
        assert analyzer.fallback_parser is not None
        assert analyzer.dep_analyzer is not None
        assert analyzer.sim_detector is not None
    
    def test_init_with_verbose(self, temp_project_dir, capsys):
        """Test ProjectAnalyzer initialization with verbose mode."""
        analyzer = ProjectAnalyzer(str(temp_project_dir), verbose=True)
        
        captured = capsys.readouterr()
        assert "Libs:" in captured.err
    
    def test_analyze_returns_project_info(self, sample_project):
        """Test that analyze returns ProjectInfo."""
        analyzer = ProjectAnalyzer(str(sample_project))
        project = analyzer.analyze()
        
        assert isinstance(project, ProjectInfo)
        assert project.name == sample_project.name
        assert project.root_path == str(sample_project.resolve())
    
    def test_analyze_finds_source_files(self, sample_project):
        """Test that analysis finds Python files."""
        analyzer = ProjectAnalyzer(str(sample_project))
        project = analyzer.analyze()
        
        # Should find main.py and utils.py from sample_project fixture
        assert project.total_files >= 2
        paths = [m.path for m in project.modules]
        assert any("main.py" in p for p in paths)
        assert any("utils.py" in p for p in paths)
    
    def test_ignores_non_source_files(self, temp_project_dir):
        """Test that non-source files are ignored."""
        # Create non-source files
        (temp_project_dir / "README.md").write_text("# README")
        (temp_project_dir / "config.json").write_text("{}")
        (temp_project_dir / "script.sh").write_text("echo hello")
        (temp_project_dir / "test.py").write_text("print('hello')")
        
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        project = analyzer.analyze()
        
        # Should only find .py file
        for module in project.modules:
            assert module.path.endswith('.py')
    
    def test_ignores_common_dirs(self, temp_project_dir):
        """Test that common directories are ignored."""
        # Create directories that should be ignored
        (temp_project_dir / "node_modules").mkdir()
        (temp_project_dir / "__pycache__").mkdir()
        (temp_project_dir / "venv").mkdir()
        (temp_project_dir / ".git").mkdir()
        
        # Add files in ignored directories
        (temp_project_dir / "node_modules" / "test.py").write_text("print('test')")
        (temp_project_dir / "__pycache__" / "test.py").write_text("print('test')")
        
        # Add a regular file
        (temp_project_dir / "main.py").write_text("print('main')")
        
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        project = analyzer.analyze()
        
        # Should not include files from ignored directories
        for module in project.modules:
            assert "node_modules" not in module.path
            assert "__pycache__" not in module.path
            assert "venv" not in module.path
    
    def test_analyze_extracts_functions(self, temp_project_dir):
        """Test that analysis extracts functions."""
        code = '''
def hello():
    """Say hello."""
    return "hello"

def add(a, b):
    return a + b
'''
        (temp_project_dir / "funcs.py").write_text(code)
        
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        project = analyzer.analyze()
        
        assert len(project.modules) >= 1
        module = next((m for m in project.modules if "funcs.py" in m.path), None)
        assert module is not None
        assert len(module.functions) >= 2
        
        func_names = [f.name for f in module.functions]
        assert "hello" in func_names
        assert "add" in func_names
    
    def test_analyze_extracts_classes(self, temp_project_dir):
        """Test that analysis extracts classes."""
        code = '''
class MyClass:
    """A test class."""
    
    def __init__(self):
        self.value = 0
    
    def method(self):
        return self.value
'''
        (temp_project_dir / "classes.py").write_text(code)
        
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        project = analyzer.analyze()
        
        module = next((m for m in project.modules if "classes.py" in m.path), None)
        assert module is not None
        assert len(module.classes) >= 1
        
        cls = module.classes[0]
        assert cls.name == "MyClass"
        assert len(cls.methods) >= 2  # __init__ and method
    
    def test_analyze_extracts_imports(self, temp_project_dir):
        """Test that analysis extracts imports."""
        code = '''
import os
import sys
from pathlib import Path
from typing import List, Dict

def func():
    pass
'''
        (temp_project_dir / "imports.py").write_text(code)
        
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        project = analyzer.analyze()
        
        module = next((m for m in project.modules if "imports.py" in m.path), None)
        assert module is not None
        assert len(module.imports) >= 2
    
    def test_analyze_counts_lines(self, temp_project_dir):
        """Test that analysis counts lines."""
        code = '''# Comment
def func():
    pass

class MyClass:
    pass
'''
        (temp_project_dir / "lines.py").write_text(code)
        
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        project = analyzer.analyze()
        
        module = next((m for m in project.modules if "lines.py" in m.path), None)
        assert module is not None
        assert module.lines_total > 0
    
    def test_analyze_empty_project(self, temp_project_dir):
        """Test analysis of empty project."""
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        project = analyzer.analyze()
        
        assert isinstance(project, ProjectInfo)
        assert project.total_files == 0
        assert len(project.modules) == 0
    
    def test_detect_entrypoints(self, temp_project_dir):
        """Test entry point detection."""
        (temp_project_dir / "main.py").write_text("print('main')")
        (temp_project_dir / "app.py").write_text("print('app')")
        (temp_project_dir / "utils.py").write_text("print('utils')")
        
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        project = analyzer.analyze()
        
        # main.py and app.py should be detected as entry points
        assert any("main.py" in ep for ep in project.entrypoints)
    
    def test_get_statistics(self, sample_project):
        """Test get_statistics method."""
        analyzer = ProjectAnalyzer(str(sample_project))
        analyzer.analyze()
        
        stats = analyzer.get_statistics()
        
        assert 'total_files' in stats
        assert 'total_lines' in stats
        assert 'languages' in stats
        assert stats['total_files'] >= 2
    
    def test_language_detection(self, temp_project_dir):
        """Test language detection from file extensions."""
        (temp_project_dir / "test.py").write_text("print('python')")
        (temp_project_dir / "test.js").write_text("console.log('js')")
        
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        project = analyzer.analyze()
        
        assert 'python' in project.languages
        assert 'javascript' in project.languages


class TestAnalyzeProjectFunction:
    """Test the analyze_project convenience function."""
    
    def test_analyze_project(self, sample_project):
        """Test analyze_project function."""
        project = analyze_project(str(sample_project))
        
        assert isinstance(project, ProjectInfo)
        assert project.total_files >= 2


class TestGetLibraryStatus:
    """Test the get_library_status function."""
    
    def test_get_library_status(self):
        """Test get_library_status function."""
        status = get_library_status()
        
        assert 'tree_sitter' in status
        assert 'networkx' in status
        assert 'rapidfuzz' in status
        assert 'nltk' in status
        assert 'spacy' in status
        assert all(isinstance(v, bool) for v in status.values())
