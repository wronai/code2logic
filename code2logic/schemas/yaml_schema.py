"""
YAML Format Schema for Code2Logic.

Defines the structure and validation for YAML specifications.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import yaml


@dataclass
class MethodSchema:
    """Schema for method definition."""
    name: str
    signature: str
    intent: str = ""
    lines: int = 0
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    raises: List[str] = field(default_factory=list)


@dataclass
class ClassSchema:
    """Schema for class definition."""
    name: str
    bases: List[str] = field(default_factory=list)
    docstring: str = ""
    methods: List[MethodSchema] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    is_abstract: bool = False
    is_dataclass: bool = False


@dataclass
class FunctionSchema:
    """Schema for function definition."""
    name: str
    signature: str
    intent: str = ""
    lines: int = 0
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)


@dataclass
class ModuleSchema:
    """Schema for module definition."""
    path: str
    language: str = "python"
    lines: int = 0
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    classes: List[ClassSchema] = field(default_factory=list)
    functions: List[FunctionSchema] = field(default_factory=list)


@dataclass
class YAMLSchema:
    """
    Complete YAML specification schema.
    
    Structure:
    ```yaml
    project: <name>
    statistics:
      files: <int>
      lines: <int>
      languages: {<lang>: <count>}
    modules:
      - path: <path>
        language: <lang>
        imports: [<import>]
        classes:
          - name: <name>
            bases: [<base>]
            docstring: <doc>
            methods:
              - name: <name>
                signature: <sig>
                intent: <intent>
        functions:
          - name: <name>
            signature: <sig>
            intent: <intent>
    ```
    """
    project: str
    statistics: Dict[str, Any] = field(default_factory=dict)
    modules: List[ModuleSchema] = field(default_factory=list)


def validate_yaml(spec: str) -> Tuple[bool, List[str]]:
    """
    Validate YAML specification.
    
    Args:
        spec: YAML specification string
        
    Returns:
        Tuple of (is_valid, errors)
    """
    errors: List[str] = []
    
    try:
        data = yaml.safe_load(spec)
    except yaml.YAMLError as e:
        return False, [f"YAML parse error: {e}"]
    
    if not isinstance(data, dict):
        return False, ["Root must be a dictionary"]
    
    # Check required fields
    if 'project' not in data and 'modules' not in data:
        errors.append("Missing 'project' or 'modules' field")
    
    # Validate modules
    if 'modules' in data:
        if not isinstance(data['modules'], list):
            errors.append("'modules' must be a list")
        else:
            for i, module in enumerate(data['modules']):
                module_errors = _validate_module(module, i)
                errors.extend(module_errors)
    
    return len(errors) == 0, errors


def _validate_module(module: Dict, index: int) -> List[str]:
    """Validate a module definition."""
    errors: List[str] = []
    prefix = f"modules[{index}]"
    
    if not isinstance(module, dict):
        return [f"{prefix}: must be a dictionary"]
    
    if 'path' not in module:
        errors.append(f"{prefix}: missing 'path'")
    
    # Validate classes
    if 'classes' in module:
        if not isinstance(module['classes'], list):
            errors.append(f"{prefix}.classes: must be a list")
        else:
            for j, cls in enumerate(module['classes']):
                cls_errors = _validate_class(cls, f"{prefix}.classes[{j}]")
                errors.extend(cls_errors)
    
    # Validate functions
    if 'functions' in module:
        if not isinstance(module['functions'], list):
            errors.append(f"{prefix}.functions: must be a list")
    
    return errors


def _validate_class(cls: Dict, prefix: str) -> List[str]:
    """Validate a class definition."""
    errors: List[str] = []
    
    if not isinstance(cls, dict):
        return [f"{prefix}: must be a dictionary"]
    
    if 'name' not in cls:
        errors.append(f"{prefix}: missing 'name'")
    
    # Validate methods
    if 'methods' in cls:
        if not isinstance(cls['methods'], list):
            errors.append(f"{prefix}.methods: must be a list")
        else:
            for i, method in enumerate(cls['methods']):
                if not isinstance(method, dict):
                    errors.append(f"{prefix}.methods[{i}]: must be a dictionary")
                elif 'name' not in method:
                    errors.append(f"{prefix}.methods[{i}]: missing 'name'")
    
    return errors
