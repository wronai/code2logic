"""
Tests for the ProjectAnalyzer class.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from code2logic.analyzer import ProjectAnalyzer
from code2logic.models import Module, Function, Class, Project


class TestProjectAnalyzer:
    """Test cases for ProjectAnalyzer."""
    
    def test_init(self, temp_project_dir):
        """Test ProjectAnalyzer initialization."""
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        
        assert analyzer.project_path == Path(temp_project_dir)
        assert analyzer.project is None
        assert analyzer.tree_sitter_parser is not None
        assert analyzer.fallback_parser is not None
        assert analyzer.dependency_analyzer is not None
        assert analyzer.similarity_detector is not None
    
    def test_init_with_config(self, temp_project_dir):
        """Test ProjectAnalyzer initialization with config."""
        config = {"debug": True, "max_depth": 10}
        analyzer = ProjectAnalyzer(str(temp_project_dir), config)
        
        assert analyzer.config == config
    
    def test_discover_source_files(self, sample_project):
        """Test source file discovery."""
        analyzer = ProjectAnalyzer(str(sample_project))
        files = analyzer._discover_source_files()
        
        assert len(files) >= 2  # main.py and utils.py
        assert any(f.name == "main.py" for f in files)
        assert any(f.name == "utils.py" for f in files)
    
    def test_discover_source_files_filters_non_source(self, temp_project_dir):
        """Test that non-source files are filtered out."""
        # Create non-source files
        (temp_project_dir / "README.md").write_text("# README")
        (temp_project_dir / "config.json").write_text("{}")
        (temp_project_dir / "script.sh").write_text("echo hello")
        
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        files = analyzer._discover_source_files()
        
        # Should not include non-source files
        assert not any(f.suffix in ['.md', '.json', '.sh'] for f in files)
    
    def test_discover_source_files_ignores_common_dirs(self, temp_project_dir):
        """Test that common directories are ignored."""
        # Create directories that should be ignored
        (temp_project_dir / "node_modules").mkdir()
        (temp_project_dir / "__pycache__").mkdir()
        (temp_project_dir / "venv").mkdir()
        (temp_project_dir / ".git").mkdir()
        
        # Add files in ignored directories
        ((temp_project_dir / "node_modules" / "test.py")).write_text("print('test')")
        ((temp_project_dir / "__pycache__" / "test.py")).write_text("print('test')")
        
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        files = analyzer._discover_source_files()
        
        # Should not include files from ignored directories
        assert not any("node_modules" in str(f) for f in files)
        assert not any("__pycache__" in str(f) for f in files)
    
    @patch('code2logic.analyzer.TreeSitterParser')
    @patch('code2logic.analyzer.FallbackParser')
    def test_parse_file_tree_sitter_success(self, mock_fallback, mock_tree_sitter, sample_project):
        """Test successful parsing with Tree-sitter."""
        # Setup mock
        mock_parser = Mock()
        mock_module = Module(
            name="test",
            path="test.py",
            functions=[],
            classes=[],
            imports=[]
        )
        mock_parser.parse_file.return_value = mock_module
        mock_tree_sitter.return_value = mock_parser
        
        analyzer = ProjectAnalyzer(str(sample_project))
        
        # Test parsing
        test_file = sample_project / "main.py"
        result = analyzer._parse_file(test_file)
        
        assert result is not None
        assert result.name == "test"
        mock_parser.parse_file.assert_called_once_with(test_file)
        mock_fallback.return_value.parse_file.assert_not_called()
    
    @patch('code2logic.analyzer.TreeSitterParser')
    @patch('code2logic.analyzer.FallbackParser')
    def test_parse_file_fallback_success(self, mock_fallback, mock_tree_sitter, sample_project):
        """Test fallback parsing when Tree-sitter fails."""
        # Setup mocks
        mock_tree_parser = Mock()
        mock_tree_parser.parse_file.side_effect = Exception("Tree-sitter error")
        
        mock_fallback_parser = Mock()
        mock_module = Module(
            name="test",
            path="test.py",
            functions=[],
            classes=[],
            imports=[]
        )
        mock_fallback_parser.parse_file.return_value = mock_module
        
        mock_tree_sitter.return_value = mock_tree_parser
        mock_fallback.return_value = mock_fallback_parser
        
        analyzer = ProjectAnalyzer(str(sample_project))
        
        # Test parsing
        test_file = sample_project / "main.py"
        result = analyzer._parse_file(test_file)
        
        assert result is not None
        assert result.name == "test"
        mock_tree_parser.parse_file.assert_called_once_with(test_file)
        mock_fallback_parser.parse_file.assert_called_once_with(test_file)
    
    @patch('code2logic.analyzer.TreeSitterParser')
    @patch('code2logic.analyzer.FallbackParser')
    def test_parse_file_both_fail(self, mock_fallback, mock_tree_sitter, sample_project):
        """Test when both parsers fail."""
        # Setup mocks to fail
        mock_tree_parser = Mock()
        mock_tree_parser.parse_file.side_effect = Exception("Tree-sitter error")
        
        mock_fallback_parser = Mock()
        mock_fallback_parser.parse_file.side_effect = Exception("Fallback error")
        
        mock_tree_sitter.return_value = mock_tree_parser
        mock_fallback.return_value = mock_fallback_parser
        
        analyzer = ProjectAnalyzer(str(sample_project))
        
        # Test parsing
        test_file = sample_project / "main.py"
        result = analyzer._parse_file(test_file)
        
        assert result is None
    
    def test_extract_metadata(self, sample_project):
        """Test metadata extraction."""
        # Create project files
        (sample_project / "pyproject.toml").write_text("[tool.poetry]\nname = 'test'")
        (sample_project / "requirements.txt").write_text("requests==2.25.1")
        
        analyzer = ProjectAnalyzer(str(sample_project))
        metadata = analyzer._extract_metadata()
        
        assert "pyproject.toml" in metadata
        assert "requirements.txt" in metadata
        assert "test" in metadata["pyproject.toml"]["content"]
    
    @patch('code2logic.analyzer.DependencyAnalyzer')
    @patch('code2logic.analyzer.SimilarityDetector')
    def test_analyze_complete(self, mock_similarity, mock_dependency, sample_project):
        """Test complete analysis process."""
        # Setup mocks
        mock_dep_analyzer = Mock()
        mock_dep_analyzer.analyze_dependencies.return_value = []
        
        mock_sim_detector = Mock()
        mock_sim_detector.detect_similarities.return_value = []
        
        mock_dependency.return_value = mock_dep_analyzer
        mock_similarity.return_value = mock_sim_detector
        
        analyzer = ProjectAnalyzer(str(sample_project))
        
        # Mock the parsing to avoid actual file parsing
        with patch.object(analyzer, '_parse_file') as mock_parse:
            mock_module = Module(
                name="main",
                path=str(sample_project / "main.py"),
                functions=[
                    Function(name="test_func", parameters=[], lines_of_code=5, complexity=1)
                ],
                classes=[],
                imports=["os"]
            )
            mock_parse.return_value = mock_module
            
            # Run analysis
            project = analyzer.analyze()
            
            # Assertions
            assert isinstance(project, Project)
            assert project.name == sample_project.name
            assert len(project.modules) >= 1
            assert project.modules[0].name == "main"
            
            # Verify mocks were called
            mock_dep_analyzer.analyze_dependencies.assert_called_once()
            mock_sim_detector.detect_similarities.assert_called_once()
    
    def test_analyze_without_files(self, temp_project_dir):
        """Test analysis with no source files."""
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        
        # Mock file discovery to return empty list
        with patch.object(analyzer, '_discover_source_files', return_value=[]):
            project = analyzer.analyze()
            
            assert isinstance(project, Project)
            assert len(project.modules) == 0
            assert len(project.dependencies) == 0
            assert len(project.similarities) == 0
    
    @patch('code2logic.analyzer.BaseGenerator')
    def test_generate_output(self, mock_generator_class, sample_project_model):
        """Test output generation."""
        # Setup mock generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        analyzer = ProjectAnalyzer("/test")
        analyzer.project = sample_project_model
        
        # Generate output
        analyzer.generate_output(mock_generator, "output.json")
        
        # Verify generator was called
        mock_generator.generate.assert_called_once_with(sample_project_model, "output.json")
    
    def test_generate_output_without_analysis(self, sample_project_model):
        """Test output generation without prior analysis."""
        analyzer = ProjectAnalyzer("/test")
        # analyzer.project is None
        
        with pytest.raises(ValueError, match="Project not analyzed yet"):
            analyzer.generate_output(Mock(), "output.json")
    
    def test_read_project_file_error(self, temp_project_dir):
        """Test error handling in _read_project_file."""
        analyzer = ProjectAnalyzer(str(temp_project_dir))
        
        # Test with non-existent file
        non_existent = temp_project_dir / "non_existent.json"
        result = analyzer._read_project_file(non_existent)
        
        assert result == {}
    
    def test_analysis_with_config(self, sample_project):
        """Test analysis with custom configuration."""
        config = {"debug": True, "include_tests": False}
        analyzer = ProjectAnalyzer(str(sample_project), config)
        
        # Mock the parsing to test config usage
        with patch.object(analyzer, '_parse_file') as mock_parse:
            mock_module = Module(
                name="main",
                path=str(sample_project / "main.py"),
                functions=[],
                classes=[],
                imports=[]
            )
            mock_parse.return_value = mock_module
            
            project = analyzer.analyze()
            
            assert analyzer.config == config
            assert isinstance(project, Project)
