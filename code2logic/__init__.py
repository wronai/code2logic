"""
code2logic - Convert codebase structure to logical representations.

A Python package for analyzing code projects and generating various
output formats including dependency graphs, logical models, and
AI-powered insights.
"""

from .analyzer import ProjectAnalyzer
from .models import Project, Module, Function, Class, Dependency
from .generators import (
    CSVGenerator,
    YAMLGenerator,
    JSONGenerator,
    CompactGenerator,
    MarkdownGenerator,
)
from .llm import LLMInterface
from .intent import IntentAnalyzer

__version__ = "1.0.0"
__author__ = "code2logic team"

__all__ = [
    "ProjectAnalyzer",
    "Project",
    "Module", 
    "Function",
    "Class",
    "Dependency",
    "CSVGenerator",
    "YAMLGenerator", 
    "JSONGenerator",
    "CompactGenerator",
    "MarkdownGenerator",
    "LLMInterface",
    "IntentAnalyzer",
]
