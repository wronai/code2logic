"""Core analysis components.

Re-exports from parent package for backward compatibility.
"""
from ..models import (
    FunctionInfo, ClassInfo, TypeInfo, ModuleInfo,
    DependencyNode, ProjectInfo
)
from ..analyzer import ProjectAnalyzer, analyze_project
from ..dependency import DependencyAnalyzer
from ..errors import (
    ErrorSeverity, ErrorType, AnalysisError, AnalysisResult,
    ErrorHandler, create_error_handler
)

__all__ = [
    'FunctionInfo', 'ClassInfo', 'TypeInfo', 'ModuleInfo',
    'DependencyNode', 'ProjectInfo',
    'ProjectAnalyzer', 'analyze_project', 'DependencyAnalyzer',
    'ErrorSeverity', 'ErrorType', 'AnalysisError', 'AnalysisResult',
    'ErrorHandler', 'create_error_handler',
]
