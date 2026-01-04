"""
TOON Format Generator for Code2Logic.

TOON (Token-Oriented Object Notation) is a compact, human-readable encoding
of JSON that minimizes tokens for LLM input. It combines YAML-like indentation
with CSV-style tabular arrays.

See: https://github.com/toon-format/toon
"""

import re
from typing import List, Dict, Any, Optional
from .models import ProjectInfo, ModuleInfo, ClassInfo, FunctionInfo


class TOONGenerator:
    """
    Generates TOON format output from ProjectInfo.
    
    TOON is optimized for LLM consumption with:
    - Minimal tokens (40% fewer than JSON)
    - Tabular arrays for uniform data
    - YAML-like indentation for structure
    
    Usage:
        gen = TOONGenerator()
        toon_output = gen.generate(project_info)
    """
    
    # Characters that require quoting in TOON
    SPECIAL_CHARS = re.compile(r'[:\"\\\[\]\{\}\n\t\r,]')
    LOOKS_LIKE_LITERAL = re.compile(r'^(true|false|null|-?\d+\.?\d*([eE][+-]?\d+)?|-)$')
    
    def __init__(self, delimiter: str = ',', use_tabs: bool = False):
        """
        Initialize TOON generator.
        
        Args:
            delimiter: Field delimiter for arrays (',' or '\t' or '|')
            use_tabs: Use tab delimiter for better token efficiency
        """
        self.delimiter = '\t' if use_tabs else delimiter
        self.delim_marker = ' ' if use_tabs else ('' if delimiter == ',' else delimiter)
    
    def generate(self, project: ProjectInfo, detail: str = 'standard') -> str:
        """
        Generate TOON format from ProjectInfo.
        
        Args:
            project: Analyzed project info
            detail: Level of detail ('compact', 'standard', 'full')
        
        Returns:
            TOON formatted string
        """
        lines = []
        
        # Project metadata
        lines.append(f"project: {self._quote(project.name)}")
        lines.append(f"root: {self._quote(project.root_path)}")
        lines.append(f"generated: {project.generated_at}")
        
        # Statistics as nested object
        lines.append("stats:")
        lines.append(f"  files: {project.total_files}")
        lines.append(f"  lines: {project.total_lines}")
        
        # Languages as primitive array
        if project.languages:
            lang_items = [f"{k}:{v}" for k, v in project.languages.items()]
            lines.append(f"  languages[{len(lang_items)}]: {self.delimiter.join(lang_items)}")
        
        # Modules - tabular format for efficiency
        if project.modules:
            lines.append("")
            lines.extend(self._generate_modules(project.modules, detail))
        
        return '\n'.join(lines)
    
    def _generate_modules(self, modules: List[ModuleInfo], detail: str) -> List[str]:
        """Generate modules section."""
        lines = []
        
        # Module summary as tabular array
        lines.append(f"modules[{len(modules)}]{{path{self.delim_marker}lang{self.delim_marker}lines}}:")
        for m in modules:
            path = self._quote(m.path)
            lines.append(f"  {path}{self.delimiter}{m.language}{self.delimiter}{m.lines_code}")
        
        # Detailed module info
        if detail in ('standard', 'full'):
            lines.append("")
            lines.append("module_details:")
            
            for m in modules:
                lines.append(f"  {self._quote(m.path)}:")
                
                # ENHANCED: Add imports for better reproduction
                if m.imports:
                    imports_str = self.delimiter.join(m.imports[:15])
                    lines.append(f"    imports[{len(m.imports)}]: {imports_str}")
                
                # ENHANCED: Add exports
                if m.exports:
                    exports_str = self.delimiter.join(m.exports[:10])
                    lines.append(f"    exports[{len(m.exports)}]: {exports_str}")
                
                # Classes
                if m.classes:
                    lines.extend(self._generate_classes(m.classes, detail, indent=4))
                
                # Functions
                if m.functions:
                    lines.extend(self._generate_functions(m.functions, detail, indent=4))
        
        return lines
    
    def _generate_classes(self, classes: List[ClassInfo], detail: str, indent: int = 0) -> List[str]:
        """Generate classes in TOON format."""
        lines = []
        ind = ' ' * indent
        
        # ENHANCED: Richer tabular format with decorators
        header_fields = f"name{self.delim_marker}bases{self.delim_marker}decorators{self.delim_marker}props{self.delim_marker}methods"
        lines.append(f"{ind}classes[{len(classes)}]{{{header_fields}}}:")
        
        for c in classes:
            name = self._quote(c.name)
            bases = '|'.join(c.bases) if c.bases else '-'
            # ClassInfo may not have decorators attribute
            class_decorators = getattr(c, 'decorators', []) or []
            decorators = '|'.join(class_decorators[:3]) if class_decorators else '-'
            props = len(c.properties) if c.properties else 0
            method_count = len(c.methods)
            lines.append(f"{ind}  {name}{self.delimiter}{bases}{self.delimiter}{decorators}{self.delimiter}{props}{self.delimiter}{method_count}")
        
        # Detailed class info with methods (for standard and full)
        if detail in ('standard', 'full'):
            lines.append(f"{ind}class_details:")
            for c in classes:
                lines.append(f"{ind}  {self._quote(c.name)}:")
                
                # ENHANCED: Add docstring
                if c.docstring:
                    doc = c.docstring[:100].replace('\n', ' ').strip()
                    lines.append(f"{ind}    doc: {self._quote(doc)}")
                
                # ENHANCED: Add properties with types
                if c.properties:
                    props_str = self.delimiter.join(c.properties[:10])
                    lines.append(f"{ind}    properties[{len(c.properties)}]: {props_str}")
                
                # Methods with full details
                if c.methods:
                    lines.extend(self._generate_methods(c.methods, detail, indent + 4))
        
        return lines
    
    def _generate_methods(self, methods: List[FunctionInfo], detail: str = 'standard', indent: int = 0) -> List[str]:
        """Generate methods in tabular TOON format."""
        lines = []
        ind = ' ' * indent
        
        # ENHANCED: Richer tabular array for methods
        header = f"name{self.delim_marker}sig{self.delim_marker}decorators{self.delim_marker}async{self.delim_marker}lines"
        lines.append(f"{ind}methods[{len(methods)}]{{{header}}}:")
        
        for m in methods:
            name = self._quote(m.name)
            sig = self._quote(self._build_signature(m))
            method_decorators = getattr(m, 'decorators', []) or []
            decorators = '|'.join(method_decorators[:2]) if method_decorators else '-'
            is_async = 'true' if getattr(m, 'is_async', False) else 'false'
            lines.append(f"{ind}  {name}{self.delimiter}{sig}{self.delimiter}{decorators}{self.delimiter}{is_async}{self.delimiter}{m.lines}")
        
        # ENHANCED: Add intent/docstring for full detail
        if detail == 'full':
            lines.append(f"{ind}method_docs:")
            for m in methods:
                if m.intent or m.docstring:
                    doc = (m.intent or m.docstring or '')[:80].replace('\n', ' ').strip()
                    if doc:
                        lines.append(f"{ind}  {self._quote(m.name)}: {self._quote(doc)}")
        
        return lines
    
    def _generate_functions(self, functions: List[FunctionInfo], detail: str, indent: int = 0) -> List[str]:
        """Generate functions in tabular TOON format."""
        lines = []
        ind = ' ' * indent
        
        # ENHANCED: Richer tabular array with decorators and category
        header = f"name{self.delim_marker}sig{self.delim_marker}decorators{self.delim_marker}async{self.delim_marker}category{self.delim_marker}lines"
        lines.append(f"{ind}functions[{len(functions)}]{{{header}}}:")
        
        for f in functions:
            name = self._quote(f.name)
            sig = self._quote(self._build_signature(f))
            func_decorators = getattr(f, 'decorators', []) or []
            decorators = '|'.join(func_decorators[:2]) if func_decorators else '-'
            is_async = 'true' if getattr(f, 'is_async', False) else 'false'
            # category may not exist on FunctionInfo
            category = getattr(f, 'category', None) or '-'
            lines.append(f"{ind}  {name}{self.delimiter}{sig}{self.delimiter}{decorators}{self.delimiter}{is_async}{self.delimiter}{category}{self.delimiter}{f.lines}")
        
        # ENHANCED: Add intent/docstring for standard and full detail
        if detail in ('standard', 'full'):
            has_docs = any(f.intent or f.docstring for f in functions)
            if has_docs:
                lines.append(f"{ind}function_docs:")
                for f in functions:
                    doc = (f.intent or f.docstring or '')[:100].replace('\n', ' ').strip()
                    if doc:
                        lines.append(f"{ind}  {self._quote(f.name)}: {self._quote(doc)}")
        
        return lines
    
    def _build_signature(self, f: FunctionInfo) -> str:
        """Build compact but complete signature string."""
        params = []
        # ENHANCED: Include more params and preserve type hints
        for p in f.params[:6]:
            p_clean = p.replace('\n', ' ').replace(',', ';').strip()
            if p_clean:
                params.append(p_clean)
        
        param_str = ';'.join(params)
        if len(f.params) > 6:
            param_str += f'...+{len(f.params)-6}'
        
        # ENHANCED: Always include return type even if None
        ret = f.return_type if f.return_type else 'None'
        return f"({param_str})->{ret}"
    
    def _quote(self, value: Any) -> str:
        """Quote a value if necessary for TOON format."""
        if value is None:
            return 'null'
        
        s = str(value)
        
        # Empty string must be quoted
        if not s:
            return '""'
        
        # Check if quoting needed
        needs_quote = (
            self.SPECIAL_CHARS.search(s) is not None or
            self.LOOKS_LIKE_LITERAL.match(s) is not None or
            s.startswith(' ') or s.endswith(' ') or
            s.startswith('-')
        )
        
        if needs_quote:
            # Escape backslashes and quotes
            s = s.replace('\\', '\\\\').replace('"', '\\"')
            s = s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
            return f'"{s}"'
        
        return s
    
    def generate_compact(self, project: ProjectInfo) -> str:
        """Generate minimal TOON output."""
        return self.generate(project, detail='compact')
    
    def generate_full(self, project: ProjectInfo) -> str:
        """Generate detailed TOON output."""
        return self.generate(project, detail='full')


class TOONParser:
    """
    Parse TOON format back to Python dict.
    
    This is a simplified parser for code2logic TOON output.
    For full TOON parsing, use the official toon library.
    """
    
    def __init__(self):
        self.delimiter = ','
    
    def parse(self, content: str) -> Dict[str, Any]:
        """Parse TOON content to dict."""
        result = {}
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            
            if not line.strip() or line.strip().startswith('#'):
                continue
            
            # Parse key-value
            indent = len(line) - len(line.lstrip())
            line = line.strip()
            
            if ':' not in line:
                continue
            
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Check for array header
            array_match = re.match(r'(\w+)\[(\d+)\](\{[^}]+\})?', key)
            if array_match:
                arr_name = array_match.group(1)
                arr_len = int(array_match.group(2))
                fields = array_match.group(3)
                
                if fields:
                    # Tabular array
                    field_names = fields[1:-1].split(',')
                    items = []
                    
                    # Parse rows
                    for _ in range(arr_len):
                        if i >= len(lines):
                            break
                        row_line = lines[i].strip()
                        i += 1
                        
                        if row_line:
                            row_values = row_line.split(self.delimiter)
                            item = {}
                            for fi, fn in enumerate(field_names):
                                if fi < len(row_values):
                                    item[fn.strip()] = self._parse_value(row_values[fi].strip())
                            items.append(item)
                    
                    result[arr_name] = items
                else:
                    # Primitive array
                    if value:
                        items = [self._parse_value(v.strip()) for v in value.split(self.delimiter)]
                        result[arr_name] = items
                    else:
                        result[arr_name] = []
            else:
                # Simple key-value
                result[key] = self._parse_value(value)
        
        return result
    
    def _parse_value(self, value: str) -> Any:
        """Parse a TOON value to Python type."""
        if not value:
            return ''
        
        # Unquote if quoted
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
            value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
            value = value.replace('\\"', '"').replace('\\\\', '\\')
            return value
        
        # Check literals
        if value == 'null':
            return None
        if value == 'true':
            return True
        if value == 'false':
            return False
        
        # Try number
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        return value


def generate_toon(project: ProjectInfo, detail: str = 'standard', use_tabs: bool = False) -> str:
    """
    Convenience function to generate TOON format.
    
    Args:
        project: Analyzed project info
        detail: Level of detail ('compact', 'standard', 'full')
        use_tabs: Use tab delimiter for better token efficiency
    
    Returns:
        TOON formatted string
    """
    gen = TOONGenerator(use_tabs=use_tabs)
    return gen.generate(project, detail)


def parse_toon(content: str) -> Dict[str, Any]:
    """Convenience function to parse TOON content."""
    return TOONParser().parse(content)
