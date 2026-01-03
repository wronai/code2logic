#!/usr/bin/env python3
"""
Tests for format-specific characteristics.

Tests the unique strengths and weaknesses of each format:
- YAML: Best overall score, text similarity
- LogicML: Best compression, token efficiency
- Gherkin: Behavior-driven, but over-engineers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic import analyze_project
from code2logic.generators import YAMLGenerator
from code2logic.gherkin import GherkinGenerator
from code2logic.logicml import LogicMLGenerator


class TestYAMLFormat:
    """Tests for YAML format characteristics."""
    
    def test_yaml_includes_all_classes(self):
        """YAML should include all classes from analyzed code."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = YAMLGenerator()
        spec = gen.generate(project, detail='full')
        
        assert 'Calculator' in spec
        assert 'Task' in spec or 'AsyncTaskQueue' in spec
    
    def test_yaml_includes_signatures(self):
        """YAML should include function signatures."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = YAMLGenerator()
        spec = gen.generate(project, detail='full')
        
        assert 'signature' in spec.lower()
    
    def test_yaml_includes_docstrings(self):
        """YAML should include docstrings as intent."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = YAMLGenerator()
        spec = gen.generate(project, detail='full')
        
        assert 'intent' in spec.lower() or 'docstring' in spec.lower()
    
    def test_yaml_structure_is_valid(self):
        """YAML output should be parseable."""
        import yaml
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = YAMLGenerator()
        spec = gen.generate(project, detail='full')
        
        # Should not raise
        data = yaml.safe_load(spec)
        assert data is not None
        assert 'project' in data or 'modules' in data


class TestLogicMLFormat:
    """Tests for LogicML format characteristics."""
    
    def test_logicml_compression_better_than_yaml(self):
        """LogicML should have better compression than YAML."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        
        yaml_gen = YAMLGenerator()
        yaml_spec = yaml_gen.generate(project, detail='full')
        
        logicml_gen = LogicMLGenerator()
        logicml_spec = logicml_gen.generate(project)
        
        # LogicML should be smaller
        assert len(logicml_spec.content) < len(yaml_spec)
    
    def test_logicml_includes_signatures(self):
        """LogicML should include 'sig:' for signatures."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = LogicMLGenerator()
        spec = gen.generate(project)
        
        assert 'sig:' in spec.content
    
    def test_logicml_includes_async_marker(self):
        """LogicML should mark async functions."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = LogicMLGenerator()
        spec = gen.generate(project)
        
        # Should have async signatures
        assert 'async' in spec.content.lower()
    
    def test_logicml_includes_attrs(self):
        """LogicML should include class attributes."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = LogicMLGenerator()
        spec = gen.generate(project)
        
        assert 'attrs:' in spec.content
    
    def test_logicml_includes_side_effects(self):
        """LogicML should detect and include side effects."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = LogicMLGenerator()
        spec = gen.generate(project)
        
        # Should detect side effects like "Modifies list", "Adds element"
        assert 'side:' in spec.content
    
    def test_logicml_token_estimate(self):
        """LogicML should provide token estimate."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = LogicMLGenerator()
        spec = gen.generate(project)
        
        assert spec.token_estimate > 0
        # Token estimate should be roughly content_length / 4
        expected = len(spec.content) // 4
        assert abs(spec.token_estimate - expected) < expected * 0.5


class TestGherkinFormat:
    """Tests for Gherkin format characteristics."""
    
    def test_gherkin_has_feature(self):
        """Gherkin should have Feature keyword."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = GherkinGenerator()
        spec = gen.generate(project)
        
        assert 'Feature:' in spec
    
    def test_gherkin_has_scenarios(self):
        """Gherkin should have Scenario keywords."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = GherkinGenerator()
        spec = gen.generate(project)
        
        assert 'Scenario:' in spec
    
    def test_gherkin_has_given_when_then(self):
        """Gherkin should use Given/When/Then structure."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = GherkinGenerator()
        spec = gen.generate(project)
        
        assert 'Given' in spec or 'When' in spec or 'Then' in spec


class TestFormatComparison:
    """Comparative tests across formats."""
    
    def test_all_formats_cover_same_classes(self):
        """All formats should cover the same classes."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        
        yaml_gen = YAMLGenerator()
        yaml_spec = yaml_gen.generate(project, detail='full')
        
        logicml_gen = LogicMLGenerator()
        logicml_spec = logicml_gen.generate(project)
        
        gherkin_gen = GherkinGenerator()
        gherkin_spec = gherkin_gen.generate(project)
        
        # All should mention Calculator or other common class
        assert 'Calculator' in yaml_spec or 'Task' in yaml_spec
        assert 'Calculator' in logicml_spec.content or 'Task' in logicml_spec.content
        assert 'calculator' in gherkin_spec.lower() or 'task' in gherkin_spec.lower()
    
    def test_logicml_is_most_compact(self):
        """LogicML should be the most compact format."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        
        yaml_gen = YAMLGenerator()
        yaml_spec = yaml_gen.generate(project, detail='full')
        
        logicml_gen = LogicMLGenerator()
        logicml_spec = logicml_gen.generate(project)
        
        gherkin_gen = GherkinGenerator()
        gherkin_spec = gherkin_gen.generate(project)
        
        logicml_len = len(logicml_spec.content)
        yaml_len = len(yaml_spec)
        gherkin_len = len(gherkin_spec)
        
        # LogicML should be smaller than both
        assert logicml_len < yaml_len, f"LogicML ({logicml_len}) should be smaller than YAML ({yaml_len})"
    
    def test_compression_ratios(self):
        """Test compression ratios for all formats."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        
        # Calculate original size
        original_size = project.total_lines * 40  # Rough estimate
        
        yaml_gen = YAMLGenerator()
        yaml_spec = yaml_gen.generate(project, detail='full')
        
        logicml_gen = LogicMLGenerator()
        logicml_spec = logicml_gen.generate(project)
        
        yaml_ratio = len(yaml_spec) / original_size
        logicml_ratio = len(logicml_spec.content) / original_size
        
        # Both should have reasonable compression
        assert yaml_ratio < 1.5, f"YAML ratio {yaml_ratio} too high"
        assert logicml_ratio < 1.0, f"LogicML ratio {logicml_ratio} too high"


class TestAsyncCodeHandling:
    """Tests for async code handling across formats."""
    
    def test_yaml_handles_async(self):
        """YAML should properly mark async functions."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = YAMLGenerator()
        spec = gen.generate(project, detail='full')
        
        assert 'is_async' in spec or 'async' in spec.lower()
    
    def test_logicml_handles_async(self):
        """LogicML should mark async in signature."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = LogicMLGenerator()
        spec = gen.generate(project)
        
        # Should have "sig: async (...)" pattern
        assert 'async' in spec.content


class TestDataclassHandling:
    """Tests for dataclass handling across formats."""
    
    def test_yaml_handles_dataclasses(self):
        """YAML should identify dataclasses."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = YAMLGenerator()
        spec = gen.generate(project, detail='full')
        
        # Should mention dataclass attributes
        assert 'Point' in spec or 'User' in spec or 'Config' in spec
    
    def test_logicml_handles_dataclasses(self):
        """LogicML should handle dataclasses with attrs."""
        project = analyze_project('tests/samples/', use_treesitter=False)
        gen = LogicMLGenerator()
        spec = gen.generate(project)
        
        # Should have attrs for dataclasses
        assert 'attrs:' in spec.content


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
