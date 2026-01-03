"""
Tests for output generators.
"""

import json
import yaml
import csv
import pytest
from pathlib import Path
from unittest.mock import mock_open, patch

from code2logic.generators import (
    CSVGenerator,
    YAMLGenerator,
    JSONGenerator,
    CompactGenerator,
    MarkdownGenerator,
    get_generator
)
from code2logic.models import Project, Module, Function, Class, Dependency


class TestCSVGenerator:
    """Test cases for CSVGenerator."""
    
    def test_generate_creates_files(self, sample_project_model, tmp_path):
        """Test that CSV generation creates multiple files."""
        generator = CSVGenerator()
        output_path = tmp_path / "test"
        
        generator.generate(sample_project_model, str(output_path))
        
        # Check that files were created
        assert (tmp_path / "test.modules.csv").exists()
        assert (tmp_path / "test.functions.csv").exists()
        assert (tmp_path / "test.classes.csv").exists()
        assert (tmp_path / "test.dependencies.csv").exists()
    
    def test_modules_csv_content(self, sample_project_model, tmp_path):
        """Test modules CSV content."""
        generator = CSVGenerator()
        output_path = tmp_path / "test"
        
        generator.generate(sample_project_model, str(output_path))
        
        # Read and verify modules CSV
        with open(tmp_path / "test.modules.csv", 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            # Check header
            assert rows[0] == ['name', 'path', 'lines_of_code', 'functions', 'classes', 'imports']
            
            # Check data rows
            assert len(rows) >= 2  # Header + at least one module
            assert rows[1][0] in ['module1', 'module2']  # Module name
    
    def test_functions_csv_content(self, sample_project_model, tmp_path):
        """Test functions CSV content."""
        generator = CSVGenerator()
        output_path = tmp_path / "test"
        
        generator.generate(sample_project_model, str(output_path))
        
        # Read and verify functions CSV
        with open(tmp_path / "test.functions.csv", 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            # Check header
            assert rows[0] == ['module', 'name', 'lines_of_code', 'complexity', 'docstring']
            
            # Check data rows
            if len(rows) > 1:  # If there are functions
                assert rows[1][0] in ['module1', 'module2']  # Module name
    
    def test_empty_project(self, tmp_path):
        """Test CSV generation with empty project."""
        empty_project = Project(
            name="empty",
            path="/empty",
            modules=[],
            dependencies=[],
            similarities=[]
        )
        
        generator = CSVGenerator()
        output_path = tmp_path / "test"
        
        generator.generate(empty_project, str(output_path))
        
        # Files should still be created with headers only
        assert (tmp_path / "test.modules.csv").exists()
        assert (tmp_path / "test.functions.csv").exists()


class TestYAMLGenerator:
    """Test cases for YAMLGenerator."""
    
    def test_generate_yaml(self, sample_project_model, tmp_path):
        """Test YAML generation."""
        generator = YAMLGenerator()
        output_path = tmp_path / "test.yaml"
        
        generator.generate(sample_project_model, str(output_path))
        
        assert output_path.exists()
        
        # Load and verify YAML content
        with open(output_path, 'r') as f:
            data = yaml.safe_load(f)
        
        assert 'project' in data
        assert data['project']['name'] == 'test_project'
        assert 'modules' in data['project']
        assert 'dependencies' in data['project']
    
    def test_yaml_structure(self, sample_project_model, tmp_path):
        """Test YAML structure."""
        generator = YAMLGenerator()
        output_path = tmp_path / "test.yaml"
        
        generator.generate(sample_project_model, str(output_path))
        
        with open(output_path, 'r') as f:
            data = yaml.safe_load(f)
        
        project_data = data['project']
        
        # Check required fields
        required_fields = ['name', 'path', 'modules', 'dependencies', 'similarities']
        for field in required_fields:
            assert field in project_data
        
        # Check module structure
        if project_data['modules']:
            module = project_data['modules'][0]
            assert 'name' in module
            assert 'functions' in module
            assert 'classes' in module


class TestJSONGenerator:
    """Test cases for JSONGenerator."""
    
    def test_generate_json(self, sample_project_model, tmp_path):
        """Test JSON generation."""
        generator = JSONGenerator()
        output_path = tmp_path / "test.json"
        
        generator.generate(sample_project_model, str(output_path))
        
        assert output_path.exists()
        
        # Load and verify JSON content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert 'project' in data
        assert data['project']['name'] == 'test_project'
    
    def test_json_statistics(self, sample_project_model, tmp_path):
        """Test JSON statistics generation."""
        generator = JSONGenerator()
        output_path = tmp_path / "test.json"
        
        generator.generate(sample_project_model, str(output_path))
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        project_data = data['project']
        
        # Check statistics
        assert 'statistics' in project_data
        stats = project_data['statistics']
        
        assert 'total_modules' in stats
        assert 'total_functions' in stats
        assert 'total_classes' in stats
        assert 'total_dependencies' in stats
        assert 'total_lines_of_code' in stats
    
    def test_json_module_structure(self, sample_project_model, tmp_path):
        """Test JSON module structure."""
        generator = JSONGenerator()
        output_path = tmp_path / "test.json"
        
        generator.generate(sample_project_model, str(output_path))
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        modules = data['project']['modules']
        
        if modules:
            module = modules[0]
            
            # Check module fields
            assert 'name' in module
            assert 'path' in module
            assert 'lines_of_code' in module
            assert 'functions' in module
            assert 'classes' in module
            
            # Check function structure
            if module['functions']:
                func = module['functions'][0]
                assert 'name' in func
                assert 'lines_of_code' in func
                assert 'complexity' in func
                assert 'has_docstring' in func


class TestCompactGenerator:
    """Test cases for CompactGenerator."""
    
    def test_generate_compact(self, sample_project_model, tmp_path):
        """Test compact text generation."""
        generator = CompactGenerator()
        output_path = tmp_path / "test.txt"
        
        generator.generate(sample_project_model, str(output_path))
        
        assert output_path.exists()
        
        # Read and verify content
        with open(output_path, 'r') as f:
            content = f.read()
        
        assert 'test_project' in content
        assert 'Modules:' in content
        assert 'Functions:' in content
        assert 'Classes:' in content
        assert 'Dependencies:' in content
        assert 'LOC:' in content
    
    def test_compact_format(self, sample_project_model, tmp_path):
        """Test compact format structure."""
        generator = CompactGenerator()
        output_path = tmp_path / "test.txt"
        
        generator.generate(sample_project_model, str(output_path))
        
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        # Should start with project summary
        assert lines[0].startswith("Project: test_project")
        
        # Should have module sections
        content = ''.join(lines)
        assert 'module1' in content or 'module2' in content
    
    def test_empty_project_compact(self, tmp_path):
        """Test compact generation with empty project."""
        empty_project = Project(
            name="empty",
            path="/empty",
            modules=[],
            dependencies=[],
            similarities=[]
        )
        
        generator = CompactGenerator()
        output_path = tmp_path / "test.txt"
        
        generator.generate(empty_project, str(output_path))
        
        with open(output_path, 'r') as f:
            content = f.read()
        
        assert 'Project: empty' in content
        assert 'Modules: 0' in content


class TestMarkdownGenerator:
    """Test cases for MarkdownGenerator."""
    
    def test_generate_markdown(self, sample_project_model, tmp_path):
        """Test Markdown generation."""
        generator = MarkdownGenerator()
        output_path = tmp_path / "test.md"
        
        generator.generate(sample_project_model, str(output_path))
        
        assert output_path.exists()
        
        # Read and verify content
        with open(output_path, 'r') as f:
            content = f.read()
        
        assert '# test_project' in content
        assert '## Statistics' in content
        assert '## Modules' in content
        assert '| Metric | Value |' in content
    
    def test_markdown_statistics_table(self, sample_project_model, tmp_path):
        """Test Markdown statistics table."""
        generator = MarkdownGenerator()
        output_path = tmp_path / "test.md"
        
        generator.generate(sample_project_model, str(output_path))
        
        with open(output_path, 'r') as f:
            content = f.read()
        
        # Check table structure
        assert '| Modules |' in content
        assert '| Functions |' in content
        assert '| Classes |' in content
        assert '| Dependencies |' in content
        assert '| Lines of Code |' in content
    
    def test_markdown_modules_section(self, sample_project_model, tmp_path):
        """Test Markdown modules section."""
        generator = MarkdownGenerator()
        output_path = tmp_path / "test.md"
        
        generator.generate(sample_project_model, str(output_path))
        
        with open(output_path, 'r') as f:
            content = f.read()
        
        # Check module sections
        assert '### module1' in content or '### module2' in content
        assert '**Path:**' in content
        assert '**Lines of Code:**' in content
    
    def test_markdown_dependencies_section(self, tmp_path):
        """Test Markdown dependencies section."""
        # Create project with dependencies
        dependency = Dependency(
            source="module1",
            target="module2",
            type="import",
            strength=0.8
        )
        
        project = Project(
            name="test",
            path="/test",
            modules=[],
            dependencies=[dependency],
            similarities=[]
        )
        
        generator = MarkdownGenerator()
        output_path = tmp_path / "test.md"
        
        generator.generate(project, str(output_path))
        
        with open(output_path, 'r') as f:
            content = f.read()
        
        assert '## Dependencies' in content
        assert '| Source | Target | Type | Strength |' in content
        assert '| module1 | module2 | import | 0.80 |' in content


class TestGeneratorRegistry:
    """Test cases for generator registry."""
    
    def test_get_generator_csv(self):
        """Test getting CSV generator."""
        generator = get_generator('csv')
        assert isinstance(generator, CSVGenerator)
    
    def test_get_generator_yaml(self):
        """Test getting YAML generator."""
        generator = get_generator('yaml')
        assert isinstance(generator, YAMLGenerator)
    
    def test_get_generator_json(self):
        """Test getting JSON generator."""
        generator = get_generator('json')
        assert isinstance(generator, JSONGenerator)
    
    def test_get_generator_compact(self):
        """Test getting compact generator."""
        generator = get_generator('compact')
        assert isinstance(generator, CompactGenerator)
    
    def test_get_generator_markdown(self):
        """Test getting Markdown generator."""
        generator = get_generator('markdown')
        assert isinstance(generator, MarkdownGenerator)
    
    def test_get_generator_invalid(self):
        """Test getting invalid generator."""
        with pytest.raises(ValueError, match="Unsupported format"):
            get_generator('invalid_format')
    
    def test_all_generators_available(self):
        """Test that all expected generators are available."""
        expected_formats = ['csv', 'yaml', 'json', 'compact', 'markdown']
        
        for fmt in expected_formats:
            generator = get_generator(fmt)
            assert generator is not None


class TestBaseGenerator:
    """Test cases for BaseGenerator functionality."""
    
    def test_ensure_output_dir_creates_directory(self, tmp_path):
        """Test that output directory is created."""
        generator = CSVGenerator()
        nested_path = tmp_path / "subdir" / "subsubdir" / "test.csv"
        
        result_path = generator._ensure_output_dir(str(nested_path))
        
        assert result_path.parent.exists()
        assert result_path.parent.is_dir()
    
    def test_ensure_output_dir_existing_directory(self, tmp_path):
        """Test ensure_output_dir with existing directory."""
        generator = CSVGenerator()
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        
        test_path = existing_dir / "test.csv"
        result_path = generator._ensure_output_dir(str(test_path))
        
        assert result_path.parent.exists()
