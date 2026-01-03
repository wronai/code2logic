"""
Code parsers for multiple languages.

Includes:
- TreeSitterParser: High-accuracy AST parsing using Tree-sitter
- UniversalParser: Fallback regex/AST parser for environments without Tree-sitter
"""

import ast
import re
from typing import Optional, List

from .models import FunctionInfo, ClassInfo, TypeInfo, ModuleInfo
from .intent import EnhancedIntentGenerator

# Optional Tree-sitter imports
TREE_SITTER_AVAILABLE = False
try:
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    pass


class TreeSitterParser:
    """
    Parser using Tree-sitter for high-accuracy AST parsing.
    
    Supports Python, JavaScript, and TypeScript with 99% accuracy.
    Falls back gracefully if Tree-sitter libraries are not installed.
    
    Example:
        >>> parser = TreeSitterParser()
        >>> if parser.is_available('python'):
        ...     module = parser.parse('main.py', content, 'python')
    """
    
    def __init__(self):
        """Initialize Tree-sitter parsers for available languages."""
        self.parsers: dict = {}
        self.languages: dict = {}
        self.intent_gen = EnhancedIntentGenerator()
        
        if TREE_SITTER_AVAILABLE:
            self._init_parsers()
    
    def _init_parsers(self):
        """Initialize parsers for each supported language."""
        try:
            # Python
            self.languages['python'] = Language(tspython.language())
            self.parsers['python'] = Parser(self.languages['python'])
            
            # JavaScript
            self.languages['javascript'] = Language(tsjavascript.language())
            self.parsers['javascript'] = Parser(self.languages['javascript'])
            
            # TypeScript - try dedicated parser, fall back to JS
            try:
                import tree_sitter_typescript as tstypescript
                self.languages['typescript'] = Language(tstypescript.language_typescript())
                self.parsers['typescript'] = Parser(self.languages['typescript'])
            except ImportError:
                self.languages['typescript'] = self.languages['javascript']
                self.parsers['typescript'] = self.parsers['javascript']
                
        except Exception as e:
            import sys
            print(f"Tree-sitter init warning: {e}", file=sys.stderr)
    
    def is_available(self, language: str) -> bool:
        """Check if Tree-sitter parser is available for a language."""
        return language in self.parsers
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of potentially supported languages."""
        return ['python', 'javascript', 'typescript']
    
    def parse(self, filepath: str, content: str, language: str) -> Optional[ModuleInfo]:
        """
        Parse a source file using Tree-sitter.
        
        Args:
            filepath: Relative path to the file
            content: File content as string
            language: Programming language
            
        Returns:
            ModuleInfo if parsing succeeds, None otherwise
        """
        if language not in self.parsers:
            return None
        
        parser = self.parsers[language]
        tree = parser.parse(bytes(content, 'utf8'))
        
        if language == 'python':
            return self._parse_python(filepath, content, tree)
        elif language in ('javascript', 'typescript'):
            return self._parse_js_ts(filepath, content, tree, language)
        
        return None
    
    def _parse_python(self, filepath: str, content: str, tree) -> ModuleInfo:
        """Parse Python source using Tree-sitter AST."""
        root = tree.root_node
        imports, classes, functions, constants, exports = [], [], [], [], []
        docstring = None
        
        for child in root.children:
            node_type = child.type
            
            # Module docstring
            if node_type == 'expression_statement' and not docstring:
                expr = child.children[0] if child.children else None
                if expr and expr.type == 'string':
                    docstring = self._extract_string(expr, content)
            
            # Imports
            elif node_type == 'import_statement':
                imports.extend(self._extract_py_import(child, content))
            elif node_type == 'import_from_statement':
                imports.extend(self._extract_py_from_import(child, content))
            
            # Functions
            elif node_type == 'function_definition':
                func = self._extract_py_function(child, content)
                if func:
                    functions.append(func)
                    if not func.name.startswith('_'):
                        exports.append(func.name)
            
            # Decorated functions
            elif node_type == 'decorated_definition':
                inner = self._find_child(child, 'function_definition')
                if inner:
                    func = self._extract_py_function(inner, content, child)
                    if func:
                        functions.append(func)
                        if not func.name.startswith('_'):
                            exports.append(func.name)
            
            # Classes
            elif node_type == 'class_definition':
                cls = self._extract_py_class(child, content)
                if cls:
                    classes.append(cls)
                    if not cls.name.startswith('_'):
                        exports.append(cls.name)
            
            # Constants
            elif node_type == 'expression_statement':
                const = self._extract_py_constant(child, content)
                if const:
                    constants.append(const)
        
        lines = content.split('\n')
        return ModuleInfo(
            path=filepath,
            language='python',
            imports=imports[:20],
            exports=exports,
            classes=classes,
            functions=functions,
            types=[],
            constants=constants[:10],
            docstring=docstring[:100] if docstring else None,
            lines_total=len(lines),
            lines_code=len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        )
    
    def _extract_py_function(self, node, content: str, 
                              decorated_node=None) -> Optional[FunctionInfo]:
        """Extract Python function from AST node."""
        name_node = self._find_child(node, 'identifier')
        if not name_node:
            return None
        name = self._text(name_node, content)
        
        # Parameters
        params = []
        params_node = self._find_child(node, 'parameters')
        if params_node:
            for child in params_node.children:
                if child.type == 'identifier':
                    params.append(self._text(child, content))
                elif child.type in ('typed_parameter', 'typed_default_parameter'):
                    n = self._find_child(child, 'identifier')
                    t = self._find_child(child, 'type')
                    if n:
                        p = self._text(n, content)
                        if t:
                            p += ':' + self._text(t, content)
                        params.append(p)
                elif child.type == 'default_parameter' and child.children:
                    params.append(self._text(child.children[0], content))
        
        # Return type
        return_type = None
        ret_node = self._find_child(node, 'type')
        if ret_node:
            return_type = self._text(ret_node, content)
        
        # Docstring
        docstring = None
        body = self._find_child(node, 'block')
        if body and body.children:
            first = body.children[0]
            if first.type == 'expression_statement':
                expr = first.children[0] if first.children else None
                if expr and expr.type == 'string':
                    docstring = self._extract_string(expr, content)
        
        # Decorators
        decorators = []
        if decorated_node:
            for c in decorated_node.children:
                if c.type == 'decorator':
                    decorators.append(self._text(c, content).lstrip('@').split('(')[0])
        
        is_async = node.type == 'async_function_definition'
        
        return FunctionInfo(
            name=name,
            params=params[:8],
            return_type=return_type,
            docstring=docstring[:100] if docstring else None,
            calls=[],
            raises=[],
            complexity=1,
            lines=node.end_point[0] - node.start_point[0] + 1,
            decorators=decorators,
            is_async=is_async,
            is_static='staticmethod' in decorators,
            is_private=name.startswith('_') and not name.startswith('__'),
            intent=self.intent_gen.generate(name, docstring),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1
        )
    
    def _extract_py_class(self, node, content: str) -> Optional[ClassInfo]:
        """Extract Python class from AST node."""
        name_node = self._find_child(node, 'identifier')
        if not name_node:
            return None
        name = self._text(name_node, content)
        
        # Base classes
        bases = []
        arg_list = self._find_child(node, 'argument_list')
        if arg_list:
            for c in arg_list.children:
                if c.type in ('identifier', 'attribute'):
                    bases.append(self._text(c, content))
        
        # Docstring and methods
        docstring = None
        methods = []
        body = self._find_child(node, 'block')
        if body:
            for i, child in enumerate(body.children):
                if i == 0 and child.type == 'expression_statement':
                    expr = child.children[0] if child.children else None
                    if expr and expr.type == 'string':
                        docstring = self._extract_string(expr, content)
                
                if child.type == 'function_definition':
                    m = self._extract_py_function(child, content)
                    if m:
                        methods.append(m)
                elif child.type == 'decorated_definition':
                    inner = self._find_child(child, 'function_definition')
                    if inner:
                        m = self._extract_py_function(inner, content, child)
                        if m:
                            methods.append(m)
        
        return ClassInfo(
            name=name,
            bases=bases,
            docstring=docstring[:100] if docstring else None,
            methods=methods,
            properties=[],
            is_interface=False,
            is_abstract='ABC' in bases or 'ABCMeta' in bases,
            generic_params=[]
        )
    
    def _extract_py_import(self, node, content: str) -> List[str]:
        """Extract import statement."""
        imports = []
        for c in node.children:
            if c.type == 'dotted_name':
                imports.append(self._text(c, content))
            elif c.type == 'aliased_import':
                n = self._find_child(c, 'dotted_name')
                if n:
                    imports.append(self._text(n, content))
        return imports
    
    def _extract_py_from_import(self, node, content: str) -> List[str]:
        """Extract from ... import ... statement."""
        imports = []
        module = None
        for c in node.children:
            if c.type in ('dotted_name', 'import_prefix'):
                module = self._text(c, content)
        if module:
            for c in node.children:
                if c.type == 'identifier':
                    imports.append(f"{module}.{self._text(c, content)}")
                elif c.type == 'aliased_import':
                    n = self._find_child(c, 'identifier')
                    if n:
                        imports.append(f"{module}.{self._text(n, content)}")
        return imports
    
    def _extract_py_constant(self, node, content: str) -> Optional[str]:
        """Extract constant (UPPERCASE assignment)."""
        if node.children:
            expr = node.children[0]
            if expr.type == 'assignment':
                left = expr.children[0] if expr.children else None
                if left and left.type == 'identifier':
                    name = self._text(left, content)
                    if name.isupper():
                        return name
        return None
    
    def _parse_js_ts(self, filepath: str, content: str, tree, language: str) -> ModuleInfo:
        """Parse JavaScript/TypeScript source using Tree-sitter AST."""
        root = tree.root_node
        imports, classes, functions, types, constants, exports = [], [], [], [], [], []
        docstring = None
        
        for child in root.children:
            node_type = child.type
            
            # Imports
            if node_type == 'import_statement':
                for c in child.children:
                    if c.type == 'string':
                        imports.append(self._text(c, content).strip('"\''))
            
            # Exports
            elif node_type == 'export_statement':
                for c in child.children:
                    if c.type == 'class_declaration':
                        cls = self._extract_js_class(c, content)
                        if cls:
                            classes.append(cls)
                            exports.append(cls.name)
                    elif c.type == 'function_declaration':
                        func = self._extract_js_function(c, content)
                        if func:
                            functions.append(func)
                            exports.append(func.name)
                    elif c.type == 'lexical_declaration':
                        func = self._extract_js_arrow_fn(c, content)
                        if func:
                            functions.append(func)
                            exports.append(func.name)
                    elif c.type in ('interface_declaration', 'type_alias_declaration'):
                        t = self._extract_ts_type(c, content)
                        if t:
                            types.append(t)
                            exports.append(t.name)
                    elif c.type == 'enum_declaration':
                        t = self._extract_ts_enum(c, content)
                        if t:
                            types.append(t)
                            exports.append(t.name)
            
            # Non-exported declarations
            elif node_type == 'class_declaration':
                cls = self._extract_js_class(child, content)
                if cls:
                    classes.append(cls)
                    exports.append(cls.name)
            elif node_type == 'function_declaration':
                func = self._extract_js_function(child, content)
                if func:
                    functions.append(func)
                    exports.append(func.name)
            elif node_type == 'lexical_declaration':
                func = self._extract_js_arrow_fn(child, content)
                if func:
                    functions.append(func)
                const = self._extract_js_constant(child, content)
                if const:
                    constants.append(const)
            elif node_type in ('interface_declaration', 'type_alias_declaration'):
                t = self._extract_ts_type(child, content)
                if t:
                    types.append(t)
                    exports.append(t.name)
            
            # Leading comment as docstring
            elif node_type == 'comment' and not docstring:
                docstring = self._extract_js_comment(child, content)
        
        lines = content.split('\n')
        return ModuleInfo(
            path=filepath,
            language=language,
            imports=imports[:20],
            exports=list(set(exports)),
            classes=classes,
            functions=functions,
            types=types,
            constants=constants[:10],
            docstring=docstring[:100] if docstring else None,
            lines_total=len(lines),
            lines_code=len([l for l in lines if l.strip() and not l.strip().startswith('//')])
        )
    
    def _extract_js_class(self, node, content: str) -> Optional[ClassInfo]:
        """Extract JS/TS class from AST node."""
        name_node = self._find_child(node, 'type_identifier') or self._find_child(node, 'identifier')
        if not name_node:
            return None
        name = self._text(name_node, content)
        
        # Base classes
        bases = []
        heritage = self._find_child(node, 'class_heritage')
        if heritage:
            for c in heritage.children:
                if c.type == 'identifier':
                    bases.append(self._text(c, content))
        
        # Methods
        methods = []
        body = self._find_child(node, 'class_body')
        if body:
            for c in body.children:
                if c.type == 'method_definition':
                    m = self._extract_js_method(c, content)
                    if m:
                        methods.append(m)
        
        return ClassInfo(
            name=name,
            bases=bases,
            docstring=None,
            methods=methods,
            properties=[],
            is_interface=False,
            is_abstract='abstract' in self._text(node, content)[:50],
            generic_params=[]
        )
    
    def _extract_js_method(self, node, content: str) -> Optional[FunctionInfo]:
        """Extract JS/TS method from AST node."""
        name_node = self._find_child(node, 'property_identifier')
        if not name_node:
            return None
        name = self._text(name_node, content)
        
        node_text = self._text(node, content)[:100]
        is_async = 'async' in node_text.split(name)[0] if name in node_text else False
        is_static = 'static' in node_text.split(name)[0] if name in node_text else False
        
        # Parameters
        params = []
        params_node = self._find_child(node, 'formal_parameters')
        if params_node:
            params = self._extract_js_params(params_node, content)
        
        # Return type
        return_type = None
        type_ann = self._find_child(node, 'type_annotation')
        if type_ann:
            return_type = self._text(type_ann, content).lstrip(':').strip()
        
        return FunctionInfo(
            name=name,
            params=params[:8],
            return_type=return_type,
            docstring=None,
            calls=[],
            raises=[],
            complexity=1,
            lines=node.end_point[0] - node.start_point[0] + 1,
            decorators=[],
            is_async=is_async,
            is_static=is_static,
            is_private=name.startswith('_') or name.startswith('#'),
            intent=self.intent_gen.generate(name),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1
        )
    
    def _extract_js_function(self, node, content: str) -> Optional[FunctionInfo]:
        """Extract JS/TS function from AST node."""
        name_node = self._find_child(node, 'identifier')
        if not name_node:
            return None
        name = self._text(name_node, content)
        is_async = self._text(node, content)[:50].strip().startswith('async')
        
        params = []
        params_node = self._find_child(node, 'formal_parameters')
        if params_node:
            params = self._extract_js_params(params_node, content)
        
        return_type = None
        type_ann = self._find_child(node, 'type_annotation')
        if type_ann:
            return_type = self._text(type_ann, content).lstrip(':').strip()
        
        return FunctionInfo(
            name=name,
            params=params[:8],
            return_type=return_type,
            docstring=None,
            calls=[],
            raises=[],
            complexity=1,
            lines=node.end_point[0] - node.start_point[0] + 1,
            decorators=[],
            is_async=is_async,
            is_static=False,
            is_private=name.startswith('_'),
            intent=self.intent_gen.generate(name),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1
        )
    
    def _extract_js_arrow_fn(self, node, content: str) -> Optional[FunctionInfo]:
        """Extract arrow function assigned to const."""
        for c in node.children:
            if c.type == 'variable_declarator':
                name_node = self._find_child(c, 'identifier')
                arrow = self._find_child(c, 'arrow_function')
                if name_node and arrow:
                    name = self._text(name_node, content)
                    is_async = 'async' in self._text(arrow, content)[:30]
                    params = []
                    pn = self._find_child(arrow, 'formal_parameters')
                    if pn:
                        params = self._extract_js_params(pn, content)
                    return FunctionInfo(
                        name=name,
                        params=params[:8],
                        return_type=None,
                        docstring=None,
                        calls=[],
                        raises=[],
                        complexity=1,
                        lines=node.end_point[0] - node.start_point[0] + 1,
                        decorators=[],
                        is_async=is_async,
                        is_static=False,
                        is_private=name.startswith('_'),
                        intent=self.intent_gen.generate(name),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1
                    )
        return None
    
    def _extract_js_params(self, params_node, content: str) -> List[str]:
        """Extract JS/TS function parameters."""
        params = []
        for c in params_node.children:
            if c.type == 'identifier':
                params.append(self._text(c, content))
            elif c.type == 'required_parameter':
                n = self._find_child(c, 'identifier')
                t = self._find_child(c, 'type_annotation')
                if n:
                    p = self._text(n, content)
                    if t:
                        p += self._text(t, content)
                    params.append(p)
            elif c.type == 'optional_parameter':
                n = self._find_child(c, 'identifier')
                if n:
                    params.append(self._text(n, content) + '?')
        return params
    
    def _extract_ts_type(self, node, content: str) -> Optional[TypeInfo]:
        """Extract TypeScript interface or type alias."""
        name_node = self._find_child(node, 'type_identifier') or self._find_child(node, 'identifier')
        if not name_node:
            return None
        name = self._text(name_node, content)
        kind = 'interface' if node.type == 'interface_declaration' else 'type'
        return TypeInfo(name=name, kind=kind, definition=self._text(node, content)[:100])
    
    def _extract_ts_enum(self, node, content: str) -> Optional[TypeInfo]:
        """Extract TypeScript enum."""
        name_node = self._find_child(node, 'identifier')
        if not name_node:
            return None
        name = self._text(name_node, content)
        return TypeInfo(name=name, kind='enum', definition='')
    
    def _extract_js_constant(self, node, content: str) -> Optional[str]:
        """Extract constant (UPPERCASE const)."""
        for c in node.children:
            if c.type == 'variable_declarator':
                n = self._find_child(c, 'identifier')
                if n:
                    name = self._text(n, content)
                    if name.isupper():
                        return name
        return None
    
    def _extract_js_comment(self, node, content: str) -> Optional[str]:
        """Extract JS comment content."""
        text = self._text(node, content)
        if text.startswith('/**'):
            lines = text[3:-2].split('\n')
            clean = [l.strip().lstrip('*').strip() for l in lines 
                    if l.strip().lstrip('*').strip() and not l.strip().startswith('@')]
            return ' '.join(clean)[:100] if clean else None
        elif text.startswith('//'):
            return text[2:].strip()[:100]
        return None
    
    # Helper methods
    def _find_child(self, node, type_name: str):
        """Find first child of given type."""
        for c in node.children:
            if c.type == type_name:
                return c
        return None
    
    def _text(self, node, content: str) -> str:
        """Get text content of node."""
        return content[node.start_byte:node.end_byte]
    
    def _extract_string(self, node, content: str) -> str:
        """Extract string content without quotes."""
        text = self._text(node, content)
        if text.startswith('"""') or text.startswith("'''"):
            return text[3:-3].strip()
        elif text.startswith('"') or text.startswith("'"):
            return text[1:-1].strip()
        return text


class UniversalParser:
    """
    Fallback parser using Python AST and regex.
    
    Used when Tree-sitter is not available. Provides reasonable
    accuracy for Python (using built-in AST) and basic support
    for JavaScript/TypeScript using regex patterns.
    
    Example:
        >>> parser = UniversalParser()
        >>> module = parser.parse('main.py', content, 'python')
    """
    
    def __init__(self):
        """Initialize the universal parser."""
        self.intent_gen = EnhancedIntentGenerator()
    
    def parse(self, filepath: str, content: str, language: str) -> Optional[ModuleInfo]:
        """
        Parse a source file using AST or regex.
        
        Args:
            filepath: Relative path to the file
            content: File content as string
            language: Programming language
            
        Returns:
            ModuleInfo if parsing succeeds, None otherwise
        """
        if language == 'python':
            return self._parse_python(filepath, content)
        elif language in ('javascript', 'typescript'):
            return self._parse_js_ts(filepath, content, language)
        return None
    
    def _parse_python(self, filepath: str, content: str) -> Optional[ModuleInfo]:
        """Parse Python using built-in AST."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            lines = content.split('\n')
            return ModuleInfo(
                path=filepath, language='python', imports=[], exports=[],
                classes=[], functions=[], types=[], constants=[], docstring=None,
                lines_total=len(lines), lines_code=len([l for l in lines if l.strip()])
            )
        
        imports, classes, functions, constants = [], [], [], []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                imports.extend(f"{module}.{a.name}" for a in node.names if a.name != '*')
            elif isinstance(node, ast.ClassDef):
                cls = self._extract_ast_class(node)
                if cls:
                    classes.append(cls)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = self._extract_ast_function(node)
                if func:
                    functions.append(func)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id.isupper():
                        constants.append(t.id)
        
        exports = [c.name for c in classes if not c.name.startswith('_')]
        exports += [f.name for f in functions if not f.name.startswith('_')]
        lines = content.split('\n')
        
        return ModuleInfo(
            path=filepath,
            language='python',
            imports=imports[:20],
            exports=exports,
            classes=classes,
            functions=functions,
            types=[],
            constants=constants[:10],
            docstring=ast.get_docstring(tree)[:100] if ast.get_docstring(tree) else None,
            lines_total=len(lines),
            lines_code=len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        )
    
    def _extract_ast_function(self, node) -> FunctionInfo:
        """Extract function from Python AST node."""
        is_async = isinstance(node, ast.AsyncFunctionDef)
        params = []
        for arg in node.args.args:
            p = arg.arg
            if arg.annotation:
                p += ':' + self._ann_str(arg.annotation)
            params.append(p)
        
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                decorators.append(dec.func.id)
        
        docstring = ast.get_docstring(node)
        return FunctionInfo(
            name=node.name,
            params=params[:8],
            return_type=self._ann_str(node.returns) if node.returns else None,
            docstring=docstring[:100] if docstring else None,
            calls=[],
            raises=[],
            complexity=1,
            lines=node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 1,
            decorators=decorators,
            is_async=is_async,
            is_static='staticmethod' in decorators,
            is_private=node.name.startswith('_') and not node.name.startswith('__'),
            intent=self.intent_gen.generate(node.name, docstring),
            start_line=node.lineno,
            end_line=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
        )
    
    def _extract_ast_class(self, node: ast.ClassDef) -> ClassInfo:
        """Extract class from Python AST node."""
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)
        
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_ast_function(item))
        
        return ClassInfo(
            name=node.name,
            bases=bases,
            docstring=ast.get_docstring(node)[:100] if ast.get_docstring(node) else None,
            methods=methods,
            properties=[],
            is_interface=False,
            is_abstract='ABC' in bases,
            generic_params=[]
        )
    
    def _ann_str(self, node) -> str:
        """Convert AST annotation to string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Subscript):
            base = self._ann_str(node.value)
            if isinstance(node.slice, ast.Tuple):
                args = ','.join(self._ann_str(e) for e in node.slice.elts)
            else:
                args = self._ann_str(node.slice)
            return f"{base}[{args}]"
        return "Any"
    
    def _parse_js_ts(self, filepath: str, content: str, language: str) -> ModuleInfo:
        """Parse JS/TS using regex patterns."""
        imports, classes, functions, types, constants, exports = [], [], [], [], [], []
        
        # Import patterns
        for m in re.finditer(r"import\s+.*?from\s+['\"]([^'\"]+)['\"]", content):
            imports.append(m.group(1))
        
        # Class patterns
        for m in re.finditer(
            r'(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?',
            content
        ):
            classes.append(ClassInfo(
                name=m.group(1),
                bases=[m.group(2)] if m.group(2) else [],
                docstring=None,
                methods=[],
                properties=[],
                is_interface=False,
                is_abstract='abstract' in m.group(0),
                generic_params=[]
            ))
            exports.append(m.group(1))
        
        # Function patterns
        for m in re.finditer(
            r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?',
            content
        ):
            name = m.group(1)
            params = [p.strip() for p in (m.group(2) or '').split(',') if p.strip()][:8]
            functions.append(FunctionInfo(
                name=name,
                params=params,
                return_type=m.group(3).strip() if m.group(3) else None,
                docstring=None,
                calls=[],
                raises=[],
                complexity=1,
                lines=1,
                decorators=[],
                is_async='async' in m.group(0),
                is_static=False,
                is_private=name.startswith('_'),
                intent=self.intent_gen.generate(name)
            ))
            exports.append(name)
        
        # Arrow function patterns
        for m in re.finditer(
            r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*[^=]+)?\s*=>',
            content
        ):
            name = m.group(1)
            functions.append(FunctionInfo(
                name=name,
                params=[],
                return_type=None,
                docstring=None,
                calls=[],
                raises=[],
                complexity=1,
                lines=1,
                decorators=[],
                is_async='async' in m.group(0),
                is_static=False,
                is_private=name.startswith('_'),
                intent=self.intent_gen.generate(name)
            ))
            exports.append(name)
        
        # Interface/Type patterns
        for m in re.finditer(r'(?:export\s+)?(interface|type)\s+(\w+)', content):
            types.append(TypeInfo(name=m.group(2), kind=m.group(1), definition=''))
            exports.append(m.group(2))
        
        # Constant patterns
        for m in re.finditer(r'const\s+([A-Z][A-Z0-9_]+)\s*=', content):
            constants.append(m.group(1))
        
        lines = content.split('\n')
        return ModuleInfo(
            path=filepath,
            language=language,
            imports=imports[:20],
            exports=list(set(exports)),
            classes=classes,
            functions=functions,
            types=types,
            constants=constants[:10],
            docstring=None,
            lines_total=len(lines),
            lines_code=len([l for l in lines if l.strip() and not l.strip().startswith('//')])
        )


def is_tree_sitter_available() -> bool:
    """Check if Tree-sitter is available."""
    return TREE_SITTER_AVAILABLE
