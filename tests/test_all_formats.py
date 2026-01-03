"""
Comprehensive tests for all Code2Logic formats.

Tests:
- YAML format generation and validation
- LogicML format generation and validation
- JSON format generation and validation
- Gherkin format generation
- Markdown format generation and validation
- All sample file types
"""

import pytest
from pathlib import Path

from code2logic import (
    analyze_project,
    YAMLGenerator,
    JSONGenerator,
    GherkinGenerator,
    MarkdownGenerator,
    LogicMLGenerator,
    validate_yaml,
    validate_logicml,
    validate_markdown,
    validate_json,
)
from code2logic.analyzer import UniversalParser
from pathlib import Path


def analyze_file(path: str):
    """Analyze a single file and return ModuleInfo."""
    parser = UniversalParser()
    code = Path(path).read_text()
    # Detect language from extension
    ext = Path(path).suffix.lower()
    lang_map = {'.py': 'python', '.js': 'javascript', '.ts': 'typescript'}
    lang = lang_map.get(ext, 'python')
    return parser.parse(code, path, lang)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def samples_dir():
    """Get samples directory path."""
    return Path(__file__).parent / "samples"


@pytest.fixture
def sample_class_code():
    """Sample Python class code."""
    return '''
class Calculator:
    """A simple calculator class."""
    
    def __init__(self, precision: int = 2):
        self.precision = precision
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = round(a + b, self.precision)
        self.history.append(('add', a, b, result))
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        result = round(a - b, self.precision)
        self.history.append(('subtract', a, b, result))
        return result
'''


@pytest.fixture
def sample_async_code():
    """Sample async Python code."""
    return '''
import asyncio
from typing import List, Optional

async def fetch_data(url: str, timeout: int = 30) -> dict:
    """Fetch data from URL asynchronously."""
    await asyncio.sleep(0.1)
    return {"url": url, "status": "ok"}

async def process_items(items: List[str]) -> List[str]:
    """Process items concurrently."""
    results = []
    for item in items:
        results.append(item.upper())
    return results

class AsyncProcessor:
    """Async processor class."""
    
    def __init__(self):
        self.results = []
    
    async def run(self, data: dict) -> Optional[dict]:
        """Run async processing."""
        await asyncio.sleep(0.01)
        self.results.append(data)
        return data
'''


@pytest.fixture
def sample_dataclass_code():
    """Sample dataclass code."""
    return '''
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class User:
    """User model."""
    id: int
    name: str
    email: str
    tags: List[str] = field(default_factory=list)
    active: bool = True

@dataclass
class Order:
    """Order model."""
    order_id: str
    user_id: int
    items: List[str]
    total: float = 0.0
    status: str = "pending"
'''


@pytest.fixture
def sample_pydantic_code():
    """Sample Pydantic model code."""
    return '''
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Task(BaseModel):
    """Task model."""
    id: str
    name: str
    status: str = "pending"
    created_at: Optional[datetime] = Field(default=None, alias="createdAt")
    
    model_config = {"populate_by_name": True}

class TaskQueue(BaseModel):
    """Task queue container."""
    tasks: List[Task] = Field(default_factory=list)
    
    def add(self, task: Task) -> None:
        """Add task to queue."""
        self.tasks.append(task)
'''


@pytest.fixture
def sample_enum_code():
    """Sample enum code."""
    return '''
from enum import Enum, auto

class Status(Enum):
    """Status enumeration."""
    PENDING = auto()
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Priority(Enum):
    """Priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
'''


# =============================================================================
# YAML FORMAT TESTS
# =============================================================================

class TestYAMLFormat:
    """Tests for YAML format generation and validation."""
    
    def test_yaml_generation_basic(self, sample_class_code, tmp_path):
        """Test basic YAML generation."""
        # Write sample to temp file
        sample_file = tmp_path / "calculator.py"
        sample_file.write_text(sample_class_code)
        
        # Analyze and generate
        module = analyze_file(str(sample_file))
        gen = YAMLGenerator()
        spec = gen.generate_from_module(module)
        
        assert spec is not None
        assert len(spec) > 0
        assert "Calculator" in spec
        assert "add" in spec
        assert "subtract" in spec
    
    def test_yaml_validation_valid(self):
        """Test YAML validation with valid spec."""
        spec = """
project: test
modules:
  - path: main.py
    classes:
      - name: Calculator
        methods:
          - name: add
            signature: (self, a, b)
"""
        is_valid, errors = validate_yaml(spec)
        assert is_valid
        assert len(errors) == 0
    
    def test_yaml_validation_invalid(self):
        """Test YAML validation with invalid spec."""
        spec = "not: valid: yaml: {{{"
        is_valid, errors = validate_yaml(spec)
        assert not is_valid
        assert len(errors) > 0
    
    def test_yaml_async_code(self, sample_async_code, tmp_path):
        """Test YAML with async code."""
        sample_file = tmp_path / "async_code.py"
        sample_file.write_text(sample_async_code)
        
        module = analyze_file(str(sample_file))
        gen = YAMLGenerator()
        spec = gen.generate_from_module(module)
        
        assert "fetch_data" in spec
        assert "async" in spec.lower() or "AsyncProcessor" in spec
    
    def test_yaml_dataclass(self, sample_dataclass_code, tmp_path):
        """Test YAML with dataclasses."""
        sample_file = tmp_path / "models.py"
        sample_file.write_text(sample_dataclass_code)
        
        module = analyze_file(str(sample_file))
        gen = YAMLGenerator()
        spec = gen.generate_from_module(module)
        
        assert "User" in spec
        assert "Order" in spec


# =============================================================================
# LOGICML FORMAT TESTS
# =============================================================================

class TestLogicMLFormat:
    """Tests for LogicML format generation and validation."""
    
    def test_logicml_generation_basic(self, sample_class_code, tmp_path):
        """Test basic LogicML generation."""
        sample_file = tmp_path / "calculator.py"
        sample_file.write_text(sample_class_code)
        
        module = analyze_file(str(sample_file))
        project = analyze_project(str(tmp_path))
        
        gen = LogicMLGenerator()
        spec = gen.generate(project)
        
        assert spec.content is not None
        assert len(spec.content) > 0
        assert "Calculator" in spec.content
    
    def test_logicml_validation_valid(self):
        """Test LogicML validation with valid spec."""
        spec = """
# main.py | Calculator | 50 lines

Calculator:
  doc: "Simple calculator"
  methods:
    add:
      sig: (self, a: float, b: float) -> float
      does: "Add two numbers"
"""
        is_valid, errors = validate_logicml(spec)
        assert is_valid
        assert len(errors) == 0
    
    def test_logicml_validation_invalid_signature(self):
        """Test LogicML validation with unbalanced signature."""
        spec = """
# main.py | Test | 10 lines

Test:
  methods:
    broken:
      sig: (self, a: int -> int
      does: "Broken signature"
"""
        is_valid, errors = validate_logicml(spec)
        assert not is_valid
        assert any("parentheses" in e.lower() or "return" in e.lower() for e in errors)
    
    def test_logicml_async_support(self, sample_async_code, tmp_path):
        """Test LogicML async support."""
        sample_file = tmp_path / "async_code.py"
        sample_file.write_text(sample_async_code)
        
        project = analyze_project(str(tmp_path))
        gen = LogicMLGenerator()
        spec = gen.generate(project)
        
        assert "async" in spec.content
    
    def test_logicml_pydantic_support(self, sample_pydantic_code, tmp_path):
        """Test LogicML Pydantic support."""
        sample_file = tmp_path / "models.py"
        sample_file.write_text(sample_pydantic_code)
        
        project = analyze_project(str(tmp_path))
        gen = LogicMLGenerator()
        spec = gen.generate(project)
        
        assert "Task" in spec.content
        assert "BaseModel" in spec.content or "Pydantic" in spec.content
    
    def test_logicml_reexport_module(self, tmp_path):
        """Test LogicML re-export module handling."""
        init_file = tmp_path / "__init__.py"
        init_file.write_text("""
from .models import User, Order
from .utils import process

__all__ = ['User', 'Order', 'process']
""")
        
        project = analyze_project(str(tmp_path))
        gen = LogicMLGenerator()
        spec = gen.generate(project)
        
        assert "re-export" in spec.content.lower() or "exports" in spec.content.lower()


# =============================================================================
# JSON FORMAT TESTS
# =============================================================================

class TestJSONFormat:
    """Tests for JSON format generation and validation."""
    
    def test_json_generation_basic(self, sample_class_code, tmp_path):
        """Test basic JSON generation."""
        sample_file = tmp_path / "calculator.py"
        sample_file.write_text(sample_class_code)
        
        module = analyze_file(str(sample_file))
        gen = JSONGenerator()
        spec = gen.generate_from_module(module)
        
        assert spec is not None
        assert len(spec) > 0
        assert "Calculator" in spec
    
    def test_json_validation_valid(self):
        """Test JSON validation with valid spec."""
        spec = '''
{
  "project": "test",
  "modules": [
    {
      "path": "main.py",
      "classes": [
        {
          "name": "Calculator",
          "methods": [
            {"name": "add", "signature": "(self, a, b)"}
          ]
        }
      ]
    }
  ]
}
'''
        is_valid, errors = validate_json(spec)
        assert is_valid
        assert len(errors) == 0
    
    def test_json_validation_invalid(self):
        """Test JSON validation with invalid spec."""
        spec = '{"invalid": json without quotes}'
        is_valid, errors = validate_json(spec)
        assert not is_valid
        assert len(errors) > 0
    
    def test_json_validation_missing_fields(self):
        """Test JSON validation with missing fields."""
        spec = '{"modules": [{"classes": [{}]}]}'
        is_valid, errors = validate_json(spec)
        assert not is_valid
        assert any("path" in e.lower() or "name" in e.lower() for e in errors)


# =============================================================================
# GHERKIN FORMAT TESTS
# =============================================================================

class TestGherkinFormat:
    """Tests for Gherkin format generation."""
    
    def test_gherkin_generation_basic(self, sample_class_code, tmp_path):
        """Test basic Gherkin generation."""
        sample_file = tmp_path / "calculator.py"
        sample_file.write_text(sample_class_code)
        
        project = analyze_project(str(tmp_path))
        gen = GherkinGenerator()
        spec = gen.generate(project)
        
        assert spec is not None
        assert len(spec) > 0
        # Gherkin keywords
        assert "Feature" in spec or "Scenario" in spec or "Given" in spec
    
    def test_gherkin_async_code(self, sample_async_code, tmp_path):
        """Test Gherkin with async code."""
        sample_file = tmp_path / "async_code.py"
        sample_file.write_text(sample_async_code)
        
        project = analyze_project(str(tmp_path))
        gen = GherkinGenerator()
        spec = gen.generate(project)
        
        assert len(spec) > 0


# =============================================================================
# MARKDOWN FORMAT TESTS
# =============================================================================

class TestMarkdownFormat:
    """Tests for Markdown format generation and validation."""
    
    def test_markdown_generation_basic(self, sample_class_code, tmp_path):
        """Test basic Markdown generation."""
        sample_file = tmp_path / "calculator.py"
        sample_file.write_text(sample_class_code)
        
        project = analyze_project(str(tmp_path))
        gen = MarkdownGenerator()
        spec = gen.generate(project)
        
        assert spec is not None
        assert len(spec) > 0
        assert "#" in spec  # Markdown headers
    
    def test_markdown_validation_valid(self):
        """Test Markdown validation with valid spec."""
        spec = """
# Module: main.py

## Classes

### Calculator

**Methods:**
#### add
```yaml
signature: (self, a, b)
```
"""
        is_valid, errors = validate_markdown(spec)
        assert is_valid
        assert len(errors) == 0
    
    def test_markdown_validation_unclosed_code_block(self):
        """Test Markdown validation with unclosed code block."""
        spec = """
# Module: main.py

```yaml
signature: (self)
"""
        is_valid, errors = validate_markdown(spec)
        assert not is_valid
        assert any("unclosed" in e.lower() or "block" in e.lower() for e in errors)


# =============================================================================
# CROSS-FORMAT TESTS
# =============================================================================

class TestCrossFormat:
    """Tests comparing all formats."""
    
    def test_all_formats_generate(self, sample_class_code, tmp_path):
        """Test that all formats generate output."""
        sample_file = tmp_path / "calculator.py"
        sample_file.write_text(sample_class_code)
        
        project = analyze_project(str(tmp_path))
        module = analyze_file(str(sample_file))
        
        # Test all generators
        yaml_gen = YAMLGenerator()
        yaml_spec = yaml_gen.generate_from_module(module)
        assert len(yaml_spec) > 0
        
        json_gen = JSONGenerator()
        json_spec = json_gen.generate_from_module(module)
        assert len(json_spec) > 0
        
        logicml_gen = LogicMLGenerator()
        logicml_spec = logicml_gen.generate(project)
        assert len(logicml_spec.content) > 0
        
        gherkin_gen = GherkinGenerator()
        gherkin_spec = gherkin_gen.generate(project)
        assert len(gherkin_spec) > 0
        
        md_gen = MarkdownGenerator()
        md_spec = md_gen.generate(project)
        assert len(md_spec) > 0
    
    def test_logicml_most_compact(self, sample_class_code, tmp_path):
        """Test that LogicML is more compact than YAML."""
        sample_file = tmp_path / "calculator.py"
        sample_file.write_text(sample_class_code)
        
        project = analyze_project(str(tmp_path))
        module = analyze_file(str(sample_file))
        
        yaml_gen = YAMLGenerator()
        yaml_spec = yaml_gen.generate_from_module(module)
        
        logicml_gen = LogicMLGenerator()
        logicml_spec = logicml_gen.generate(project)
        
        # LogicML should be more compact (fewer tokens)
        yaml_tokens = len(yaml_spec) // 4
        logicml_tokens = logicml_spec.token_estimate
        
        # LogicML should use fewer or similar tokens
        assert logicml_tokens <= yaml_tokens * 1.2  # Allow 20% margin


# =============================================================================
# FILE TYPE TESTS
# =============================================================================

class TestFileTypes:
    """Tests for different file types."""
    
    def test_enum_file(self, sample_enum_code, tmp_path):
        """Test enum file handling."""
        sample_file = tmp_path / "enums.py"
        sample_file.write_text(sample_enum_code)
        
        module = analyze_file(str(sample_file))
        assert len(module.classes) >= 2
        assert any("Status" in c.name for c in module.classes)
    
    def test_pydantic_file(self, sample_pydantic_code, tmp_path):
        """Test Pydantic file handling."""
        sample_file = tmp_path / "models.py"
        sample_file.write_text(sample_pydantic_code)
        
        module = analyze_file(str(sample_file))
        assert len(module.classes) >= 2
        assert any("Task" in c.name for c in module.classes)
    
    def test_empty_file(self, tmp_path):
        """Test empty file handling."""
        sample_file = tmp_path / "empty.py"
        sample_file.write_text("")
        
        module = analyze_file(str(sample_file))
        assert module is not None
        assert len(module.classes) == 0
        assert len(module.functions) == 0
    
    def test_imports_only_file(self, tmp_path):
        """Test file with only imports."""
        sample_file = tmp_path / "imports_only.py"
        sample_file.write_text("""
from typing import List, Dict
from pathlib import Path
import os
import sys
""")
        
        module = analyze_file(str(sample_file))
        assert module is not None
        assert len(module.imports) >= 2


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation:
    """Tests for schema validation."""
    
    def test_all_validators_exist(self):
        """Test that all validators are importable."""
        from code2logic import (
            validate_yaml,
            validate_logicml,
            validate_markdown,
            validate_json,
        )
        
        assert callable(validate_yaml)
        assert callable(validate_logicml)
        assert callable(validate_markdown)
        assert callable(validate_json)
    
    def test_all_schemas_exist(self):
        """Test that all schemas are importable."""
        from code2logic import (
            YAMLSchema,
            LogicMLSchema,
            MarkdownSchema,
            JSONSchema,
        )
        
        assert YAMLSchema is not None
        assert LogicMLSchema is not None
        assert MarkdownSchema is not None
        assert JSONSchema is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
