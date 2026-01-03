"""
Pytest configuration and fixtures for code2logic tests.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from code2logic.models import Project, Module, Function, Class


@pytest.fixture
def sample_python_code() -> str:
    """Sample Python code for testing."""
    return '''
import os
import sys
from typing import List, Dict

def calculate_sum(numbers: List[int]) -> int:
    """Calculate the sum of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total

def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract two numbers."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def get_history(self) -> List[str]:
        """Get calculation history."""
        return self.history.copy()
'''


@pytest.fixture
def sample_javascript_code() -> str:
    """Sample JavaScript code for testing."""
    return '''
const os = require('os');
const fs = require('fs');

function calculateSum(numbers) {
    let total = 0;
    for (const num of numbers) {
        total += num;
    }
    return total;
}

function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }
    
    subtract(a, b) {
        const result = a - b;
        this.history.push(`${a} - ${b} = ${result}`);
        return result;
    }
    
    getHistory() {
        return [...this.history];
    }
}

module.exports = { calculateSum, fibonacci, Calculator };
'''


@pytest.fixture
def sample_java_code() -> str:
    """Sample Java code for testing."""
    return '''
import java.util.ArrayList;
import java.util.List;

public class Calculator {
    private List<String> history;
    
    public Calculator() {
        this.history = new ArrayList<>();
    }
    
    public double add(double a, double b) {
        double result = a + b;
        history.add(a + " + " + b + " = " + result);
        return result;
    }
    
    public double subtract(double a, double b) {
        double result = a - b;
        history.add(a + " - " + b + " = " + result);
        return result;
    }
    
    public List<String> getHistory() {
        return new ArrayList<>(history);
    }
    
    public static int fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}
'''


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_project(temp_project_dir, sample_python_code):
    """Create a sample project with Python files."""
    # Create main module
    main_file = temp_project_dir / "main.py"
    main_file.write_text(sample_python_code)
    
    # Create utils module
    utils_file = temp_project_dir / "utils.py"
    utils_file.write_text('''
def helper_function():
    """A helper function."""
    return "helper"

class HelperClass:
    """A helper class."""
    
    def method(self):
        return "method result"
''')
    
    # Create __init__.py
    init_file = temp_project_dir / "__init__.py"
    init_file.write_text("")
    
    return temp_project_dir


@pytest.fixture
def sample_module():
    """Create a sample module for testing."""
    functions = [
        Function(
            name="test_func",
            parameters=["arg1", "arg2"],
            lines_of_code=10,
            complexity=2,
            docstring="Test function",
            code="def test_func(arg1, arg2):\n    return arg1 + arg2"
        )
    ]
    
    classes = [
        Class(
            name="TestClass",
            methods=[
                Function(
                    name="method1",
                    parameters=[],
                    lines_of_code=5,
                    complexity=1,
                    docstring="Method 1"
                )
            ],
            lines_of_code=20,
            docstring="Test class"
        )
    ]
    
    return Module(
        name="test_module",
        path="/test/test_module.py",
        functions=functions,
        classes=classes,
        imports=["os", "sys"],
        lines_of_code=50
    )


@pytest.fixture
def sample_project_model():
    """Create a sample project model for testing."""
    module1 = Module(
        name="module1",
        path="/test/module1.py",
        functions=[
            Function(name="func1", parameters=[], lines_of_code=5, complexity=1)
        ],
        classes=[],
        imports=["os"],
        lines_of_code=10
    )
    
    module2 = Module(
        name="module2", 
        path="/test/module2.py",
        functions=[
            Function(name="func2", parameters=["arg"], lines_of_code=8, complexity=2)
        ],
        classes=[],
        imports=["sys", "module1"],
        lines_of_code=15
    )
    
    return Project(
        name="test_project",
        path="/test",
        modules=[module1, module2],
        dependencies=[],
        similarities=[]
    )


@pytest.fixture
def mock_llm_config():
    """Mock LLM configuration for testing."""
    return {
        "provider": "mock",
        "model": "test-model",
        "api_key": "test-key",
        "temperature": 0.7,
        "max_tokens": 1000
    }


@pytest.fixture
def sample_analysis_result():
    """Sample analysis result for testing."""
    project = Project(
        name="test_project",
        path="/test",
        modules=[],
        dependencies=[],
        similarities=[]
    )
    
    from code2logic.models import AnalysisResult
    
    return AnalysisResult(
        project=project,
        analysis_time=1.5,
        parser_used="tree_sitter",
        errors=[],
        warnings=["Minor warning"]
    )
