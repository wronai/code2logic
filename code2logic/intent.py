"""
NLP-powered intent analysis for code2logic.

This module provides functionality to analyze user intent and
provide intelligent suggestions for code refactoring and analysis.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .models import Project, Module, Function, Class


class IntentType(Enum):
    """Types of user intents."""
    REFACTOR = "refactor"
    ANALYZE = "analyze"
    OPTIMIZE = "optimize"
    DOCUMENT = "document"
    TEST = "test"
    DEBUG = "debug"
    MIGRATE = "migrate"
    SIMPLIFY = "simplify"


@dataclass
class Intent:
    """Represents a detected user intent."""
    type: IntentType
    confidence: float
    target: str  # Module, class, or function name
    description: str
    suggestions: List[str]


class IntentAnalyzer:
    """Analyzes user intent using NLP techniques."""
    
    def __init__(self):
        """Initialize the intent analyzer."""
        self.intent_patterns = self._initialize_patterns()
        self.code_smell_patterns = self._initialize_code_smell_patterns()
    
    def analyze_intent(
        self, 
        query: str, 
        project: Project
    ) -> List[Intent]:
        """
        Analyze user query and detect intents.
        
        Args:
            query: User query string
            project: Project context
            
        Returns:
            List of detected intents
        """
        intents = []
        
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        # Match against intent patterns
        for intent_type, patterns in self.intent_patterns.items():
            confidence = self._calculate_intent_confidence(keywords, patterns)
            if confidence > 0.3:
                target = self._identify_target(query, project)
                description = self._generate_description(intent_type, target)
                suggestions = self._generate_suggestions(intent_type, target, project)
                
                intents.append(Intent(
                    type=intent_type,
                    confidence=confidence,
                    target=target,
                    description=description,
                    suggestions=suggestions
                ))
        
        # Sort by confidence
        intents.sort(key=lambda x: x.confidence, reverse=True)
        
        return intents
    
    def detect_code_smells(self, project: Project) -> List[Dict[str, Any]]:
        """
        Detect code smells in the project.
        
        Args:
            project: Project to analyze
            
        Returns:
            List of detected code smells
        """
        smells = []
        
        for module in project.modules:
            # Long module
            if module.lines_of_code > 500:
                smells.append({
                    'type': 'long_module',
                    'severity': 'medium',
                    'target': module.name,
                    'description': f"Module {module.name} is too long ({module.lines_of_code} LOC)",
                    'suggestion': 'Consider splitting into smaller modules'
                })
            
            # Too many imports
            if len(module.imports) > 20:
                smells.append({
                    'type': 'too_many_imports',
                    'severity': 'low',
                    'target': module.name,
                    'description': f"Module {module.name} has too many imports ({len(module.imports)})",
                    'suggestion': 'Consider consolidating related imports or using facades'
                })
            
            # Complex functions
            for func in module.functions:
                if func.complexity > 10:
                    smells.append({
                        'type': 'complex_function',
                        'severity': 'high',
                        'target': f"{module.name}.{func.name}",
                        'description': f"Function {func.name} is too complex (complexity: {func.complexity})",
                        'suggestion': 'Consider breaking down into smaller functions'
                    })
                
                if func.lines_of_code > 50:
                    smells.append({
                        'type': 'long_function',
                        'severity': 'medium',
                        'target': f"{module.name}.{func.name}",
                        'description': f"Function {func.name} is too long ({func.lines_of_code} LOC)",
                        'suggestion': 'Consider breaking down into smaller functions'
                    })
            
            # Large classes
            for cls in module.classes:
                if len(cls.methods) > 15:
                    smells.append({
                        'type': 'large_class',
                        'severity': 'medium',
                        'target': f"{module.name}.{cls.name}",
                        'description': f"Class {cls.name} has too many methods ({len(cls.methods)})",
                        'suggestion': 'Consider splitting into smaller classes or using composition'
                    })
                
                if cls.lines_of_code > 300:
                    smells.append({
                        'type': 'large_class',
                        'severity': 'high',
                        'target': f"{module.name}.{cls.name}",
                        'description': f"Class {cls.name} is too large ({cls.lines_of_code} LOC)",
                        'suggestion': 'Consider splitting into smaller classes'
                    })
        
        return smells
    
    def suggest_refactoring(
        self, 
        target: str, 
        project: Project
    ) -> List[str]:
        """
        Suggest refactoring options for a target.
        
        Args:
            target: Target module/class/function
            project: Project context
            
        Returns:
            List of refactoring suggestions
        """
        suggestions = []
        
        # Find the target
        target_obj = self._find_target_object(target, project)
        if not target_obj:
            return suggestions
        
        # Generate suggestions based on target type
        if isinstance(target_obj, Module):
            suggestions.extend(self._suggest_module_refactoring(target_obj))
        elif isinstance(target_obj, Class):
            suggestions.extend(self._suggest_class_refactoring(target_obj))
        elif isinstance(target_obj, Function):
            suggestions.extend(self._suggest_function_refactoring(target_obj))
        
        return suggestions
    
    def _initialize_patterns(self) -> Dict[IntentType, List[str]]:
        """Initialize intent patterns."""
        return {
            IntentType.REFACTOR: [
                'refactor', 'restructure', 'reorganize', 'improve', 'clean',
                'simplify', 'rework', 'redesign'
            ],
            IntentType.ANALYZE: [
                'analyze', 'understand', 'explain', 'review', 'examine',
                'investigate', 'explore', 'study'
            ],
            IntentType.OPTIMIZE: [
                'optimize', 'performance', 'speed', 'fast', 'efficient',
                'improve performance', 'faster', 'optimize'
            ],
            IntentType.DOCUMENT: [
                'document', 'docs', 'documentation', 'readme', 'explain',
                'describe', 'comment', 'annotate'
            ],
            IntentType.TEST: [
                'test', 'testing', 'tests', 'unit test', 'coverage',
                'test case', 'spec', 'specification'
            ],
            IntentType.DEBUG: [
                'debug', 'fix', 'error', 'bug', 'issue', 'problem',
                'broken', 'not working', 'fail'
            ],
            IntentType.MIGRATE: [
                'migrate', 'convert', 'upgrade', 'update', 'port',
                'transition', 'move', 'change'
            ],
            IntentType.SIMPLIFY: [
                'simplify', 'simple', 'easier', 'reduce', 'minimal',
                'streamline', 'clean up'
            ]
        }
    
    def _initialize_code_smell_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize code smell patterns."""
        return {
            'long_parameter_list': {
                'threshold': 5,
                'severity': 'medium',
                'suggestion': 'Consider using parameter objects or configuration dictionaries'
            },
            'duplicate_code': {
                'threshold': 0.8,
                'severity': 'high',
                'suggestion': 'Extract common code into shared functions or classes'
            },
            'magic_numbers': {
                'pattern': r'\b\d{2,}\b',
                'severity': 'low',
                'suggestion': 'Replace magic numbers with named constants'
            },
            'deep_nesting': {
                'threshold': 4,
                'severity': 'medium',
                'suggestion': 'Consider using early returns or guard clauses'
            }
        }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from user query."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if len(word) > 2]
    
    def _calculate_intent_confidence(
        self, 
        keywords: List[str], 
        patterns: List[str]
    ) -> float:
        """Calculate confidence score for intent."""
        matches = sum(1 for pattern in patterns if pattern in keywords)
        return matches / len(patterns) if patterns else 0.0
    
    def _identify_target(self, query: str, project: Project) -> str:
        """Identify the target of the intent."""
        # Look for module, class, or function names in the query
        for module in project.modules:
            if module.name.lower() in query.lower():
                return module.name
            
            for cls in module.classes:
                if cls.name.lower() in query.lower():
                    return f"{module.name}.{cls.name}"
            
            for func in module.functions:
                if func.name.lower() in query.lower():
                    return f"{module.name}.{func.name}"
        
        return "project"
    
    def _generate_description(self, intent_type: IntentType, target: str) -> str:
        """Generate description for intent."""
        descriptions = {
            IntentType.REFACTOR: f"Refactoring suggestions for {target}",
            IntentType.ANALYZE: f"Analysis of {target}",
            IntentType.OPTIMIZE: f"Performance optimization for {target}",
            IntentType.DOCUMENT: f"Documentation for {target}",
            IntentType.TEST: f"Testing strategy for {target}",
            IntentType.DEBUG: f"Debugging {target}",
            IntentType.MIGRATE: f"Migration plan for {target}",
            IntentType.SIMPLIFY: f"Simplification of {target}"
        }
        return descriptions.get(intent_type, f"Analysis of {target}")
    
    def _generate_suggestions(
        self, 
        intent_type: IntentType, 
        target: str, 
        project: Project
    ) -> List[str]:
        """Generate suggestions based on intent type."""
        suggestions = []
        
        if intent_type == IntentType.REFACTOR:
            suggestions = self.suggest_refactoring(target, project)
        elif intent_type == IntentType.ANALYZE:
            suggestions = [
                "Review the dependency graph",
                "Check for code smells",
                "Analyze complexity metrics"
            ]
        elif intent_type == IntentType.OPTIMIZE:
            suggestions = [
                "Profile performance bottlenecks",
                "Review algorithmic complexity",
                "Check for memory leaks"
            ]
        elif intent_type == IntentType.DOCUMENT:
            suggestions = [
                "Add docstrings to functions",
                "Create README documentation",
                "Generate API documentation"
            ]
        elif intent_type == IntentType.TEST:
            suggestions = [
                "Write unit tests",
                "Add integration tests",
                "Improve test coverage"
            ]
        
        return suggestions
    
    def _find_target_object(
        self, 
        target: str, 
        project: Project
    ) -> Optional[Any]:
        """Find the target object in the project."""
        if target == "project":
            return project
        
        parts = target.split('.')
        if len(parts) == 1:
            # Module name
            for module in project.modules:
                if module.name == parts[0]:
                    return module
        elif len(parts) == 2:
            # Module.Class or Module.Function
            module_name, item_name = parts
            for module in project.modules:
                if module.name == module_name:
                    for cls in module.classes:
                        if cls.name == item_name:
                            return cls
                    for func in module.functions:
                        if func.name == item_name:
                            return func
        
        return None
    
    def _suggest_module_refactoring(self, module: Module) -> List[str]:
        """Suggest refactoring for a module."""
        suggestions = []
        
        if len(module.functions) > 20:
            suggestions.append("Consider splitting into multiple modules")
        
        if len(module.classes) > 10:
            suggestions.append("Consider grouping related classes")
        
        if len(module.imports) > 15:
            suggestions.append("Consider reducing import dependencies")
        
        return suggestions
    
    def _suggest_class_refactoring(self, cls: Class) -> List[str]:
        """Suggest refactoring for a class."""
        suggestions = []
        
        if len(cls.methods) > 15:
            suggestions.append("Consider splitting into smaller classes")
        
        if len(cls.base_classes) > 3:
            suggestions.append("Consider composition over inheritance")
        
        return suggestions
    
    def _suggest_function_refactoring(self, func: Function) -> List[str]:
        """Suggest refactoring for a function."""
        suggestions = []
        
        if func.complexity > 10:
            suggestions.append("Consider breaking down into smaller functions")
        
        if len(func.parameters) > 5:
            suggestions.append("Consider using parameter objects")
        
        if not func.docstring:
            suggestions.append("Add docstring documentation")
        
        return suggestions
