"""
Format Comparison Tests - TOON vs YAML vs LogicML vs JSON vs Markdown.

Tests all output formats for:
- Correctness of generated content
- Token efficiency comparison
- Round-trip capability
- LLM-friendliness metrics
"""

import pytest
import tempfile
from pathlib import Path

from code2logic import (
    analyze_project,
    YAMLGenerator,
    JSONGenerator,
    LogicMLGenerator,
    MarkdownGenerator,
    TOONGenerator,
    generate_toon,
    parse_toon,
    validate_yaml,
    validate_json,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_project(tmp_path):
    """Create a sample project for testing."""
    # Main module
    (tmp_path / "main.py").write_text('''
"""Main application module."""

from utils import helper

class Application:
    """Main application class."""
    
    def __init__(self, name: str, debug: bool = False):
        """Initialize application."""
        self.name = name
        self.debug = debug
    
    def run(self) -> int:
        """Run the application."""
        return 0
    
    async def start(self, port: int = 8080) -> None:
        """Start the server."""
        pass

def main():
    """Entry point."""
    app = Application("test")
    return app.run()
''')
    
    # Utils module
    (tmp_path / "utils.py").write_text('''
"""Utility functions."""

def helper(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

def format_string(s: str) -> str:
    """Format a string."""
    return s.strip().lower()

class Config:
    """Configuration holder."""
    
    def __init__(self):
        self.settings = {}
    
    def get(self, key: str) -> str:
        """Get a setting."""
        return self.settings.get(key, "")
''')
    
    return tmp_path


@pytest.fixture
def analyzed_project(sample_project):
    """Analyze the sample project."""
    return analyze_project(str(sample_project), use_treesitter=False)


# =============================================================================
# FORMAT GENERATION TESTS
# =============================================================================

class TestFormatGeneration:
    """Test that all formats generate valid output."""
    
    def test_yaml_generation(self, analyzed_project):
        """Test YAML format generation."""
        gen = YAMLGenerator()
        output = gen.generate(analyzed_project)
        
        assert len(output) > 0
        assert "project:" in output or "modules:" in output
        
        # Validate YAML
        is_valid, errors = validate_yaml(output)
        assert is_valid, f"YAML validation failed: {errors}"
    
    def test_json_generation(self, analyzed_project):
        """Test JSON format generation."""
        gen = JSONGenerator()
        output = gen.generate(analyzed_project)
        
        assert len(output) > 0
        assert '"' in output  # JSON uses quotes
        
        # Validate JSON
        is_valid, errors = validate_json(output)
        assert is_valid, f"JSON validation failed: {errors}"
    
    def test_logicml_generation(self, analyzed_project):
        """Test LogicML format generation."""
        gen = LogicMLGenerator()
        output = gen.generate(analyzed_project)
        
        assert len(output) > 0
        assert "<project" in output or "<module" in output
    
    def test_markdown_generation(self, analyzed_project):
        """Test Markdown format generation."""
        gen = MarkdownGenerator()
        output = gen.generate(analyzed_project)
        
        assert len(output) > 0
        assert "#" in output  # Markdown headers
    
    def test_toon_generation(self, analyzed_project):
        """Test TOON format generation."""
        gen = TOONGenerator()
        output = gen.generate(analyzed_project)
        
        assert len(output) > 0
        assert "project:" in output
        assert "modules[" in output  # TOON array syntax
    
    def test_toon_with_tabs(self, analyzed_project):
        """Test TOON format with tab delimiters."""
        gen = TOONGenerator(use_tabs=True)
        output = gen.generate(analyzed_project)
        
        assert len(output) > 0
        assert '\t' in output  # Should use tabs


# =============================================================================
# TOKEN EFFICIENCY COMPARISON
# =============================================================================

class TestTokenEfficiency:
    """Compare token efficiency across formats."""
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Simple estimation: ~4 chars per token on average
        return len(text) // 4
    
    def _count_chars(self, text: str) -> int:
        """Count non-whitespace characters."""
        return len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
    
    def test_format_sizes(self, analyzed_project):
        """Compare output sizes across formats."""
        yaml_out = YAMLGenerator().generate(analyzed_project)
        json_out = JSONGenerator().generate(analyzed_project)
        logicml_out = LogicMLGenerator().generate(analyzed_project)
        markdown_out = MarkdownGenerator().generate(analyzed_project)
        toon_out = TOONGenerator().generate(analyzed_project)
        toon_tabs = TOONGenerator(use_tabs=True).generate(analyzed_project)
        
        sizes = {
            'YAML': len(yaml_out),
            'JSON': len(json_out),
            'LogicML': len(logicml_out),
            'Markdown': len(markdown_out),
            'TOON': len(toon_out),
            'TOON(tabs)': len(toon_tabs),
        }
        
        print("\n=== Format Size Comparison ===")
        for fmt, size in sorted(sizes.items(), key=lambda x: x[1]):
            tokens = self._estimate_tokens(size)
            print(f"  {fmt}: {size} chars, ~{tokens} tokens")
        
        # TOON should be smaller than JSON
        assert sizes['TOON'] < sizes['JSON'], "TOON should be more compact than JSON"
    
    def test_token_estimates(self, analyzed_project):
        """Estimate token counts for each format."""
        formats = {
            'YAML': YAMLGenerator().generate(analyzed_project),
            'JSON': JSONGenerator().generate(analyzed_project),
            'LogicML': LogicMLGenerator().generate(analyzed_project),
            'Markdown': MarkdownGenerator().generate(analyzed_project),
            'TOON': TOONGenerator().generate(analyzed_project),
            'TOON(tabs)': TOONGenerator(use_tabs=True).generate(analyzed_project),
        }
        
        tokens = {k: self._estimate_tokens(v) for k, v in formats.items()}
        
        print("\n=== Estimated Token Counts ===")
        for fmt, count in sorted(tokens.items(), key=lambda x: x[1]):
            print(f"  {fmt}: ~{count} tokens")
        
        # All formats should produce output
        assert all(t > 0 for t in tokens.values())


# =============================================================================
# CONTENT EQUIVALENCE TESTS
# =============================================================================

class TestContentEquivalence:
    """Test that formats contain equivalent information."""
    
    def test_project_name_present(self, analyzed_project):
        """All formats should include project name."""
        yaml_out = YAMLGenerator().generate(analyzed_project)
        json_out = JSONGenerator().generate(analyzed_project)
        toon_out = TOONGenerator().generate(analyzed_project)
        
        project_name = analyzed_project.name
        
        assert project_name in yaml_out or "project" in yaml_out
        assert project_name in json_out or '"name"' in json_out
        assert project_name in toon_out or "project:" in toon_out
    
    def test_class_names_present(self, analyzed_project):
        """All formats should include class names."""
        yaml_out = YAMLGenerator().generate(analyzed_project, detail='full')
        json_out = JSONGenerator().generate(analyzed_project)
        toon_out = TOONGenerator().generate(analyzed_project, detail='full')
        
        # Check for Application and Config classes
        assert "Application" in yaml_out
        assert "Application" in json_out
        assert "Application" in toon_out
    
    def test_function_names_present(self, analyzed_project):
        """All formats should include function names."""
        yaml_out = YAMLGenerator().generate(analyzed_project, detail='full')
        json_out = JSONGenerator().generate(analyzed_project)
        toon_out = TOONGenerator().generate(analyzed_project, detail='full')
        
        # Check for main and helper functions
        assert "main" in yaml_out or "helper" in yaml_out
        assert "main" in json_out or "helper" in json_out
        assert "main" in toon_out or "helper" in toon_out


# =============================================================================
# TOON SPECIFIC TESTS
# =============================================================================

class TestTOONFormat:
    """TOON-specific format tests."""
    
    def test_toon_array_syntax(self, analyzed_project):
        """Test TOON array header syntax."""
        toon_out = TOONGenerator().generate(analyzed_project)
        
        # Should have array headers with length
        import re
        array_pattern = r'\w+\[\d+\]'
        matches = re.findall(array_pattern, toon_out)
        
        assert len(matches) > 0, "TOON should have array headers"
    
    def test_toon_tabular_syntax(self, analyzed_project):
        """Test TOON tabular array syntax."""
        toon_out = TOONGenerator().generate(analyzed_project, detail='full')
        
        # Should have tabular headers with fields
        import re
        tabular_pattern = r'\w+\[\d+\]\{[^}]+\}'
        matches = re.findall(tabular_pattern, toon_out)
        
        assert len(matches) > 0, "TOON should have tabular array headers"
    
    def test_toon_no_unnecessary_quotes(self, analyzed_project):
        """Test that TOON minimizes quoting."""
        toon_out = TOONGenerator().generate(analyzed_project)
        
        # Count quotes
        quote_count = toon_out.count('"')
        char_count = len(toon_out)
        
        # Quote ratio should be low (< 10% of chars)
        quote_ratio = quote_count / char_count if char_count > 0 else 0
        
        assert quote_ratio < 0.1, f"Too many quotes in TOON: {quote_ratio:.2%}"
    
    def test_toon_parse_roundtrip(self, analyzed_project):
        """Test basic TOON parse capability."""
        toon_out = TOONGenerator().generate(analyzed_project, detail='compact')
        
        # Parse back
        parsed = parse_toon(toon_out)
        
        # Should have project key
        assert 'project' in parsed
        assert parsed['project'] == analyzed_project.name


# =============================================================================
# DETAIL LEVEL TESTS
# =============================================================================

class TestDetailLevels:
    """Test different detail levels across formats."""
    
    def test_compact_is_smallest(self, analyzed_project):
        """Compact output should be smallest."""
        yaml_compact = YAMLGenerator().generate(analyzed_project, detail='compact')
        yaml_standard = YAMLGenerator().generate(analyzed_project, detail='standard')
        yaml_full = YAMLGenerator().generate(analyzed_project, detail='full')
        
        assert len(yaml_compact) <= len(yaml_standard)
        assert len(yaml_standard) <= len(yaml_full)
    
    def test_full_has_more_info(self, analyzed_project):
        """Full output should have more detail."""
        toon_compact = TOONGenerator().generate(analyzed_project, detail='compact')
        toon_full = TOONGenerator().generate(analyzed_project, detail='full')
        
        # Full should be larger
        assert len(toon_full) > len(toon_compact)


# =============================================================================
# BENCHMARK SUMMARY
# =============================================================================

class TestBenchmarkSummary:
    """Generate benchmark summary for all formats."""
    
    def test_print_benchmark(self, analyzed_project):
        """Print comprehensive benchmark."""
        formats = {
            'YAML': YAMLGenerator().generate(analyzed_project, detail='full'),
            'JSON': JSONGenerator().generate(analyzed_project),
            'LogicML': LogicMLGenerator().generate(analyzed_project),
            'Markdown': MarkdownGenerator().generate(analyzed_project),
            'TOON': TOONGenerator().generate(analyzed_project, detail='full'),
            'TOON(tabs)': TOONGenerator(use_tabs=True).generate(analyzed_project, detail='full'),
        }
        
        print("\n" + "=" * 60)
        print("FORMAT COMPARISON BENCHMARK")
        print("=" * 60)
        
        # Size comparison
        print("\nSize (chars):")
        for fmt, content in sorted(formats.items(), key=lambda x: len(x[1])):
            print(f"  {fmt:12s}: {len(content):6d} chars")
        
        # Token estimate
        print("\nEstimated tokens (~4 chars/token):")
        for fmt, content in sorted(formats.items(), key=lambda x: len(x[1])):
            tokens = len(content) // 4
            print(f"  {fmt:12s}: ~{tokens:5d} tokens")
        
        # Lines comparison
        print("\nLine count:")
        for fmt, content in sorted(formats.items(), key=lambda x: x[1].count('\n')):
            lines = content.count('\n') + 1
            print(f"  {fmt:12s}: {lines:6d} lines")
        
        # Relative efficiency (vs JSON baseline)
        json_size = len(formats['JSON'])
        print(f"\nRelative to JSON ({json_size} chars = 100%):")
        for fmt, content in sorted(formats.items(), key=lambda x: len(x[1])):
            pct = (len(content) / json_size) * 100
            print(f"  {fmt:12s}: {pct:6.1f}%")
        
        print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
