"""
Code parsers for code2logic using Tree-sitter and fallback methods.

This module provides parsers for different programming languages
using Tree-sitter for accurate AST parsing and fallback methods
for unsupported languages.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

from .models import Module, Function, Class, Dependency


class TreeSitterParser:
    """Parser using Tree-sitter for accurate AST parsing."""
    
    def __init__(self):
        """Initialize Tree-sitter parser."""
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter package is required")
        
        self.parsers = {}
        self._initialize_parsers()
    
    def _initialize_parsers(self) -> None:
        """Initialize parsers for different languages."""
        try:
            # Python
            py_language = Language(tree_sitter.Language.build_library(
                'build/my-languages.so',
                ['vendor/tree-sitter-py']
            ), 'python')
            self.parsers['python'] = Parser(py_language)
            
            # JavaScript
            js_language = Language(tree_sitter.Language.build_library(
                'build/my-languages.so',
                ['vendor/tree-sitter-javascript']
            ), 'javascript')
            self.parsers['javascript'] = Parser(js_language)
            
            # Java
            java_language = Language(tree_sitter.Language.build_library(
                'build/my-languages.so',
                ['vendor/tree-sitter-java']
            ), 'java')
            self.parsers['java'] = Parser(java_language)
            
        except Exception as e:
            print(f"Warning: Could not initialize Tree-sitter parsers: {e}")
    
    def parse_file(self, file_path: Path) -> Optional[Module]:
        """
        Parse a file using Tree-sitter.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            Module object or None if parsing fails
        """
        language = self._detect_language(file_path)
        
        if language not in self.parsers:
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            parser = self.parsers[language]
            tree = parser.parse(bytes(source_code, 'utf-8'))
            
            if language == 'python':
                return self._parse_python_module(file_path, source_code, tree)
            elif language == 'javascript':
                return self._parse_javascript_module(file_path, source_code, tree)
            elif language == 'java':
                return self._parse_java_module(file_path, source_code, tree)
            
        except Exception as e:
            print(f"Tree-sitter parsing failed for {file_path}: {e}")
        
        return None
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'javascript',
            '.tsx': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
        }
        
        return language_map.get(suffix, 'unknown')
    
    def _parse_python_module(
        self, 
        file_path: Path, 
        source_code: str, 
        tree
    ) -> Module:
        """Parse Python module using Tree-sitter."""
        module = Module(
            name=file_path.stem,
            path=str(file_path),
            lines_of_code=len(source_code.splitlines())
        )
        
        # Extract imports
        module.imports = self._extract_python_imports(tree)
        
        # Extract functions and classes
        self._extract_python_functions(tree, module)
        self._extract_python_classes(tree, module)
        
        return module
    
    def _parse_javascript_module(
        self, 
        file_path: Path, 
        source_code: str, 
        tree
    ) -> Module:
        """Parse JavaScript module using Tree-sitter."""
        module = Module(
            name=file_path.stem,
            path=str(file_path),
            lines_of_code=len(source_code.splitlines())
        )
        
        # Extract imports
        module.imports = self._extract_javascript_imports(tree)
        
        # Extract functions and classes
        self._extract_javascript_functions(tree, module)
        self._extract_javascript_classes(tree, module)
        
        return module
    
    def _parse_java_module(
        self, 
        file_path: Path, 
        source_code: str, 
        tree
    ) -> Module:
        """Parse Java module using Tree-sitter."""
        module = Module(
            name=file_path.stem,
            path=str(file_path),
            lines_of_code=len(source_code.splitlines())
        )
        
        # Extract imports
        module.imports = self._extract_java_imports(tree)
        
        # Extract classes
        self._extract_java_classes(tree, module)
        
        return module
    
    def _extract_python_imports(self, tree) -> List[str]:
        """Extract imports from Python AST."""
        imports = []
        
        def visit_node(node):
            if node.type == 'import_statement':
                for child in node.children:
                    if child.type == 'dotted_name':
                        imports.append('.'.join(child.text.decode().split('.')))
            elif node.type == 'import_from_statement':
                module_name = ''
                for child in node.children:
                    if child.type == 'dotted_name':
                        module_name = child.text.decode()
                        break
                if module_name:
                    imports.append(module_name)
            
            for child in node.children:
                visit_node(child)
        
        visit_node(tree.root_node)
        return imports
    
    def _extract_python_functions(self, tree, module: Module) -> None:
        """Extract functions from Python AST."""
        def visit_node(node):
            if node.type == 'function_definition':
                func_name = ''
                func_code = ''
                
                for child in node.children:
                    if child.type == 'identifier':
                        func_name = child.text.decode()
                    elif child.type == 'block':
                        func_code = child.text.decode()
                
                if func_name:
                    function = Function(
                        name=func_name,
                        code=func_code,
                        lines_of_code=len(func_code.splitlines()),
                        complexity=self._calculate_complexity(func_code)
                    )
                    module.functions.append(function)
            
            for child in node.children:
                visit_node(child)
        
        visit_node(tree.root_node)
    
    def _extract_python_classes(self, tree, module: Module) -> None:
        """Extract classes from Python AST."""
        def visit_node(node):
            if node.type == 'class_definition':
                class_name = ''
                class_code = ''
                
                for child in node.children:
                    if child.type == 'identifier':
                        class_name = child.text.decode()
                    elif child.type == 'block':
                        class_code = child.text.decode()
                
                if class_name:
                    cls = Class(
                        name=class_name,
                        lines_of_code=len(class_code.splitlines())
                    )
                    
                    # Extract methods from class
                    self._extract_class_methods(child, cls) if hasattr(child, 'children') else None
                    module.classes.append(cls)
            
            for child in node.children:
                visit_node(child)
        
        visit_node(tree.root_node)
    
    def _extract_class_methods(self, class_node, cls: Class) -> None:
        """Extract methods from a class node."""
        # This would need to be implemented based on Tree-sitter structure
        pass
    
    def _extract_javascript_imports(self, tree) -> List[str]:
        """Extract imports from JavaScript AST."""
        imports = []
        
        def visit_node(node):
            if node.type == 'import_statement':
                # Extract import name
                for child in node.children:
                    if child.type == 'string':
                        imports.append(child.text.decode().strip('"\''))
            
            for child in node.children:
                visit_node(child)
        
        visit_node(tree.root_node)
        return imports
    
    def _extract_javascript_functions(self, tree, module: Module) -> None:
        """Extract functions from JavaScript AST."""
        def visit_node(node):
            if node.type in ['function_declaration', 'function_expression']:
                func_name = 'anonymous'
                func_code = ''
                
                for child in node.children:
                    if child.type == 'identifier':
                        func_name = child.text.decode()
                    elif child.type == 'statement_block':
                        func_code = child.text.decode()
                
                function = Function(
                    name=func_name,
                    code=func_code,
                    lines_of_code=len(func_code.splitlines()),
                    complexity=self._calculate_complexity(func_code)
                )
                module.functions.append(function)
            
            for child in node.children:
                visit_node(child)
        
        visit_node(tree.root_node)
    
    def _extract_javascript_classes(self, tree, module: Module) -> None:
        """Extract classes from JavaScript AST."""
        def visit_node(node):
            if node.type == 'class_declaration':
                class_name = ''
                class_code = ''
                
                for child in node.children:
                    if child.type == 'identifier':
                        class_name = child.text.decode()
                    elif child.type == 'class_body':
                        class_code = child.text.decode()
                
                if class_name:
                    cls = Class(
                        name=class_name,
                        lines_of_code=len(class_code.splitlines())
                    )
                    module.classes.append(cls)
            
            for child in node.children:
                visit_node(child)
        
        visit_node(tree.root_node)
    
    def _extract_java_imports(self, tree) -> List[str]:
        """Extract imports from Java AST."""
        imports = []
        
        def visit_node(node):
            if node.type == 'import_declaration':
                for child in node.children:
                    if child.type == 'scoped_identifier':
                        imports.append(child.text.decode())
            
            for child in node.children:
                visit_node(child)
        
        visit_node(tree.root_node)
        return imports
    
    def _extract_java_classes(self, tree, module: Module) -> None:
        """Extract classes from Java AST."""
        def visit_node(node):
            if node.type == 'class_declaration':
                class_name = ''
                class_code = ''
                
                for child in node.children:
                    if child.type == 'identifier':
                        class_name = child.text.decode()
                    elif child.type == 'class_body':
                        class_code = child.text.decode()
                
                if class_name:
                    cls = Class(
                        name=class_name,
                        lines_of_code=len(class_code.splitlines())
                    )
                    module.classes.append(cls)
            
            for child in node.children:
                visit_node(child)
        
        visit_node(tree.root_node)
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        # Count decision points
        decision_keywords = [
            'if', 'elif', 'else', 'for', 'while', 'try', 'except',
            'finally', 'with', 'and', 'or', 'case', 'switch'
        ]
        
        for keyword in decision_keywords:
            complexity += len(re.findall(r'\b' + keyword + r'\b', code))
        
        return complexity


class FallbackParser:
    """Fallback parser using regex and simple AST parsing."""
    
    def parse_file(self, file_path: Path) -> Optional[Module]:
        """
        Parse a file using fallback methods.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            Module object or None if parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            suffix = file_path.suffix.lower()
            
            if suffix == '.py':
                return self._parse_python_fallback(file_path, source_code)
            elif suffix in ['.js', '.jsx', '.ts', '.tsx']:
                return self._parse_javascript_fallback(file_path, source_code)
            elif suffix == '.java':
                return self._parse_java_fallback(file_path, source_code)
            else:
                # Generic parsing
                return self._parse_generic_fallback(file_path, source_code)
        
        except Exception as e:
            print(f"Fallback parsing failed for {file_path}: {e}")
        
        return None
    
    def _parse_python_fallback(self, file_path: Path, source_code: str) -> Module:
        """Parse Python using AST fallback."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # If AST fails, use regex
            return self._parse_python_regex(file_path, source_code)
        
        module = Module(
            name=file_path.stem,
            path=str(file_path),
            lines_of_code=len(source_code.splitlines())
        )
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module.imports.append(node.module)
        
        # Extract functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function = Function(
                    name=node.name,
                    parameters=[arg.arg for arg in node.args.args],
                    lines_of_code=node.end_lineno - node.lineno + 1,
                    complexity=self._calculate_ast_complexity(node),
                    docstring=ast.get_docstring(node),
                    code=ast.get_source_segment(source_code, node) or ""
                )
                module.functions.append(function)
        
        # Extract classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                cls = Class(
                    name=node.name,
                    lines_of_code=node.end_lineno - node.lineno + 1,
                    docstring=ast.get_docstring(node),
                    base_classes=[base.id for base in node.bases if isinstance(base, ast.Name)]
                )
                
                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method = Function(
                            name=item.name,
                            parameters=[arg.arg for arg in item.args.args],
                            lines_of_code=item.end_lineno - item.lineno + 1,
                            complexity=self._calculate_ast_complexity(item),
                            docstring=ast.get_docstring(item)
                        )
                        cls.methods.append(method)
                
                module.classes.append(cls)
        
        return module
    
    def _parse_python_regex(self, file_path: Path, source_code: str) -> Module:
        """Parse Python using regex fallback."""
        module = Module(
            name=file_path.stem,
            path=str(file_path),
            lines_of_code=len(source_code.splitlines())
        )
        
        # Extract imports with regex
        import_patterns = [
            r'import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',
            r'from\s+([a-zA-Z_][a-zA-Z0-9_\.]*)\s+import'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, source_code)
            module.imports.extend(matches)
        
        # Extract functions with regex
        func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:'
        func_matches = re.finditer(func_pattern, source_code)
        
        for match in func_matches:
            func_name = match.group(1)
            # Simple complexity calculation
            func_code = self._extract_function_block(source_code, match.start())
            complexity = self._calculate_complexity(func_code)
            
            function = Function(
                name=func_name,
                code=func_code,
                lines_of_code=len(func_code.splitlines()),
                complexity=complexity
            )
            module.functions.append(function)
        
        # Extract classes with regex
        class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\([^)]*\))?\s*:'
        class_matches = re.finditer(class_pattern, source_code)
        
        for match in class_matches:
            class_name = match.group(1)
            class_code = self._extract_class_block(source_code, match.start())
            
            cls = Class(
                name=class_name,
                lines_of_code=len(class_code.splitlines())
            )
            module.classes.append(cls)
        
        return module
    
    def _parse_javascript_fallback(self, file_path: Path, source_code: str) -> Module:
        """Parse JavaScript using regex fallback."""
        module = Module(
            name=file_path.stem,
            path=str(file_path),
            lines_of_code=len(source_code.splitlines())
        )
        
        # Extract imports
        import_patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'require\([\'"]([^\'"]+)[\'"]\)'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, source_code)
            module.imports.extend(matches)
        
        # Extract functions
        func_patterns = [
            r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*{',
            r'const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\([^)]*\)\s*=>\s*{',
            r'let\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\([^)]*\)\s*=>\s*{',
            r'var\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*function\s*\([^)]*\)\s*{'
        ]
        
        for pattern in func_patterns:
            func_matches = re.finditer(pattern, source_code)
            for match in func_matches:
                func_name = match.group(1)
                func_code = self._extract_javascript_function_block(source_code, match.start())
                complexity = self._calculate_complexity(func_code)
                
                function = Function(
                    name=func_name,
                    code=func_code,
                    lines_of_code=len(func_code.splitlines()),
                    complexity=complexity
                )
                module.functions.append(function)
        
        return module
    
    def _parse_java_fallback(self, file_path: Path, source_code: str) -> Module:
        """Parse Java using regex fallback."""
        module = Module(
            name=file_path.stem,
            path=str(file_path),
            lines_of_code=len(source_code.splitlines())
        )
        
        # Extract imports
        import_pattern = r'import\s+([a-zA-Z_][a-zA-Z0-9_\.]*);'
        matches = re.findall(import_pattern, source_code)
        module.imports.extend(matches)
        
        # Extract classes
        class_pattern = r'(?:public\s+)?(?:private\s+)?(?:protected\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        class_matches = re.finditer(class_pattern, source_code)
        
        for match in class_matches:
            class_name = match.group(1)
            class_code = self._extract_java_class_block(source_code, match.start())
            
            cls = Class(
                name=class_name,
                lines_of_code=len(class_code.splitlines())
            )
            module.classes.append(cls)
        
        return module
    
    def _parse_generic_fallback(self, file_path: Path, source_code: str) -> Module:
        """Parse generic file using basic heuristics."""
        module = Module(
            name=file_path.stem,
            path=str(file_path),
            lines_of_code=len(source_code.splitlines())
        )
        
        # Very basic function detection
        func_pattern = r'(?:function|def|func|method)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        func_matches = re.findall(func_pattern, source_code, re.IGNORECASE)
        
        for func_name in func_matches:
            function = Function(name=func_name)
            module.functions.append(function)
        
        return module
    
    def _extract_function_block(self, source_code: str, start_pos: int) -> str:
        """Extract function code block."""
        lines = source_code[start_pos:].split('\n')
        func_lines = []
        indent_level = None
        
        for line in lines:
            if indent_level is None and line.strip():
                indent_level = len(line) - len(line.lstrip())
            
            if line.strip() and indent_level is not None:
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and line.strip() and func_lines:
                    break
            
            func_lines.append(line)
        
        return '\n'.join(func_lines)
    
    def _extract_class_block(self, source_code: str, start_pos: int) -> str:
        """Extract class code block."""
        return self._extract_function_block(source_code, start_pos)
    
    def _extract_javascript_function_block(self, source_code: str, start_pos: int) -> str:
        """Extract JavaScript function code block."""
        lines = source_code[start_pos:].split('\n')
        func_lines = []
        brace_count = 0
        
        for line in lines:
            func_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            
            if brace_count <= 0 and '{' in line:
                break
        
        return '\n'.join(func_lines)
    
    def _extract_java_class_block(self, source_code: str, start_pos: int) -> str:
        """Extract Java class code block."""
        lines = source_code[start_pos:].split('\n')
        class_lines = []
        brace_count = 0
        
        for line in lines:
            class_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            
            if brace_count <= 0 and '{' in line:
                break
        
        return '\n'.join(class_lines)
    
    def _calculate_ast_complexity(self, node) -> int:
        """Calculate complexity from AST node."""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        decision_keywords = [
            'if', 'elif', 'else', 'for', 'while', 'try', 'except',
            'finally', 'with', 'and', 'or', 'case', 'switch', 'catch'
        ]
        
        for keyword in decision_keywords:
            complexity += len(re.findall(r'\b' + keyword + r'\b', code))
        
        return complexity
