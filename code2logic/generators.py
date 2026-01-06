"""
Output generators for Code2Logic analysis results.

Includes:
- MarkdownGenerator: Human-readable Markdown format
- CompactGenerator: Ultra-compact format for token efficiency
- JSONGenerator: Machine-readable JSON format for RAG systems
"""

import json
from pathlib import Path
from typing import List
from collections import defaultdict

from .models import (
    ProjectInfo, ModuleInfo, ClassInfo, FunctionInfo,
    DependencyNode, ConstantInfo, FieldInfo
)
from .shared_utils import categorize_function, extract_domain, compute_hash, remove_self_from_params, compact_imports, deduplicate_imports


def bytes_to_kb(bytes_value: int) -> float:
    """Convert bytes to kilobytes with single decimal precision."""
    if not bytes_value:
        return 0.0
    return round(bytes_value / 1024, 1)


class MarkdownGenerator:
    """
    Generates Markdown output for project analysis.
    
    Produces human-readable documentation with:
    - Project structure tree
    - Dependency graphs
    - Module documentation with classes and functions
    - Intent descriptions for each function
    
    Example:
        >>> generator = MarkdownGenerator()
        >>> output = generator.generate(project, detail_level='standard')
        >>> print(output)
    """
    
    def generate(self, project: ProjectInfo, detail_level: str = 'standard') -> str:
        """
        Generate Markdown output.
        
        Args:
            project: ProjectInfo analysis results
            detail_level: 'compact', 'standard', or 'detailed'
            
        Returns:
            Markdown formatted string
        """
        lines = []
        
        # Header
        lines.append(f"# 📦 {project.name}")
        lines.append("")
        lines.append("```yaml")
        lines.append(f"generated: {project.generated_at}")
        lines.append(f"files: {project.total_files}")
        lines.append(f"lines: {project.total_lines}")
        lines.append(f"languages: {json.dumps(project.languages)}")
        if project.entrypoints:
            lines.append(f"entrypoints: {json.dumps(project.entrypoints[:5])}")
        lines.append("```")
        lines.append("")
        
        # Module Map
        lines.append("## 📁 Structure")
        lines.append("")
        self._gen_tree(lines, project)
        lines.append("")
        
        # Key Modules (hubs)
        if project.dependency_metrics and detail_level != 'compact':
            hubs = [p for p, n in project.dependency_metrics.items() if n.is_hub]
            if hubs:
                lines.append("## 🔗 Key Modules")
                lines.append("")
                lines.append("```yaml")
                for h in hubs[:10]:
                    n = project.dependency_metrics[h]
                    lines.append(f"{Path(h).stem}: in={n.in_degree} out={n.out_degree} pr={n.pagerank:.3f}")
                lines.append("```")
                lines.append("")
        
        # Dependencies
        deps = {k: v for k, v in project.dependency_graph.items() if v}
        if deps and detail_level != 'compact':
            lines.append("## 🔗 Dependencies")
            lines.append("")
            lines.append("```yaml")
            
            # Use short names but avoid collisions
            seen = {}
            for p in sorted(deps.keys()):
                short = Path(p).stem
                if short in seen:
                    short = f"{Path(p).parent.name}/{short}"
                seen[short] = p
            
            p2s = {v: k for k, v in seen.items()}
            
            for p, ds in sorted(deps.items())[:30]:
                s = p2s.get(p, Path(p).stem)
                sd = [p2s.get(d, Path(d).stem) for d in ds[:5]]
                if len(ds) > 5:
                    sd.append(f"+{len(ds)-5}")
                lines.append(f"{s}: [{', '.join(sd)}]")
            lines.append("```")
            lines.append("")
        
        # Similar functions
        if project.similar_functions and detail_level == 'detailed':
            lines.append("## 🔄 Similar Functions")
            lines.append("")
            lines.append("```yaml")
            for func, similar in list(project.similar_functions.items())[:10]:
                lines.append(f"{func}:")
                for s in similar[:3]:
                    lines.append(f"  - {s}")
            lines.append("```")
            lines.append("")
        
        # Modules
        lines.append("## 📄 Modules")
        lines.append("")
        
        by_dir = defaultdict(list)
        for m in project.modules:
            d = str(Path(m.path).parent)
            by_dir[d if d != '.' else '(root)'].append(m)
        
        for d in sorted(by_dir.keys()):
            lines.append(f"### 📂 {d}")
            lines.append("")
            for m in sorted(by_dir[d], key=lambda x: x.path):
                self._gen_module(lines, m, detail_level, project)
        
        return '\n'.join(lines)
    
    def _gen_tree(self, lines: List[str], project: ProjectInfo):
        """Generate project structure tree."""
        tree = {}
        for m in project.modules:
            parts = Path(m.path).parts
            curr = tree
            for p in parts[:-1]:
                if p not in curr:
                    curr[p] = {}
                curr = curr[p]
            
            exps = m.exports[:3]
            es = ', '.join(exps)
            if len(m.exports) > 3:
                es += f" +{len(m.exports)-3}"
            
            is_hub = m.path in project.dependency_metrics and \
                     project.dependency_metrics[m.path].is_hub
            hub = " ★" if is_hub else ""
            
            curr[parts[-1]] = f"[{m.language}]{hub} {es}" if es else f"[{m.language}]{hub}"
        
        lines.append("```")
        self._print_tree(lines, tree, "")
        lines.append("```")
    
    def _print_tree(self, lines: List[str], tree: dict, prefix: str, depth: int = 0):
        """Recursively print tree structure."""
        if depth >= 4:
            lines.append(f"{prefix}...")
            return
        
        items = sorted(tree.items())
        for i, (name, val) in enumerate(items):
            last = i == len(items) - 1
            conn = "└── " if last else "├── "
            
            if isinstance(val, dict):
                lines.append(f"{prefix}{conn}{name}/")
                self._print_tree(lines, val, prefix + ("    " if last else "│   "), depth + 1)
            else:
                lines.append(f"{prefix}{conn}{name}: {val}")
    
    def _gen_module(self, lines: List[str], m: ModuleInfo, 
                    detail: str, proj: ProjectInfo):
        """Generate module documentation."""
        fn = Path(m.path).name
        is_hub = m.path in proj.dependency_metrics and \
                 proj.dependency_metrics[m.path].is_hub
        
        lines.append(f"#### `{fn}`{' ★' if is_hub else ''}")
        lines.append("")
        lines.append("```yaml")
        lines.append(f"path: {m.path}")
        lines.append(f"lang: {m.language} | lines: {m.lines_code}/{m.lines_total}")
        
        if m.imports and detail != 'compact':
            imps = ', '.join(m.imports[:5])
            if len(m.imports) > 5:
                imps += f"... +{len(m.imports)-5}"
            lines.append(f"imports: [{imps}]")
        
        if m.constants:
            lines.append(f"constants: [{', '.join(m.constants[:5])}]")
        lines.append("```")
        lines.append("")
        
        if m.docstring:
            lines.append(f"> {m.docstring}")
            lines.append("")
        
        # Types
        for t in m.types[:5]:
            lines.append(f"**{t.kind} `{t.name}`**")
            lines.append("")
        
        # Classes
        for cls in m.classes:
            self._gen_class(lines, cls, detail)
        
        # Functions
        if m.functions:
            pub = [f for f in m.functions if not f.is_private]
            if pub:
                if detail == 'compact':
                    fs = ', '.join(f.name for f in pub[:8])
                    if len(pub) > 8:
                        fs += f" +{len(pub)-8}"
                    lines.append(f"**Functions:** {fs}")
                else:
                    lines.append("**Functions:**")
                    lines.append("")
                    for f in pub[:15]:
                        sig = self._sig(f)
                        lines.append(f"- `{sig}` — {f.intent[:50]}")
                lines.append("")
        
        lines.append("---")
        lines.append("")
    
    def _gen_class(self, lines: List[str], cls: ClassInfo, detail: str):
        """Generate class documentation."""
        kind = "interface" if cls.is_interface else "abstract class" if cls.is_abstract else "class"
        bases = f"({', '.join(cls.bases)})" if cls.bases else ""
        
        lines.append(f"**{kind} `{cls.name}`{bases}**")
        lines.append("")
        
        if cls.docstring:
            lines.append(f"> {cls.docstring}")
            lines.append("")
        
        if cls.methods:
            pub = [m for m in cls.methods 
                   if not m.is_private or m.name in ('constructor', '__init__')]
            if pub:
                lines.append("```yaml")
                lines.append("methods:")
                for m in pub[:12]:
                    sig = self._sig(m)
                    lines.append(f"  {sig}  # {m.intent[:40]}")
                if len(pub) > 12:
                    lines.append(f"  # ... +{len(pub)-12} more")
                lines.append("```")
        lines.append("")
    
    def _sig(self, f: FunctionInfo) -> str:
        """Generate function signature."""
        pre = ""
        if f.is_static:
            pre = "static "
        if f.is_async:
            pre += "async "
 
        raw_params = [p.replace('\n', ' ').replace('  ', ' ').strip() for p in (f.params or [])]
        params_no_self = remove_self_from_params(raw_params)
        params = params_no_self[:4]
        if len(params_no_self) > 4:
            params = params + [f"...+{len(params_no_self)-4}"]
        ps = ', '.join(params)
 
        ret = f" -> {f.return_type}" if f.return_type else ""
 
        return f"{pre}{f.name}({ps}){ret}"


class CompactGenerator:
    """
    Generates ultra-compact output for token efficiency.
    
    Optimized for minimal token usage while preserving
    essential information for LLM context.
    
    Example:
        >>> generator = CompactGenerator()
        >>> output = generator.generate(project)
        >>> print(output)  # ~10-15x smaller than Markdown
    """
    
    def generate(self, project: ProjectInfo) -> str:
        """
        Generate compact output.
        
        Args:
            project: ProjectInfo analysis results
            
        Returns:
            Compact formatted string
        """
        lines = []
        
        langs = '/'.join(f"{k}:{v}" for k, v in project.languages.items())
        lines.append(f"# {project.name} | {project.total_files}f {project.total_lines}L | {langs}")
        lines.append("")
        
        if project.entrypoints:
            lines.append(f"ENTRY: {' '.join(project.entrypoints[:3])}")
        
        if project.dependency_metrics:
            hubs = [Path(p).stem for p, n in project.dependency_metrics.items() if n.is_hub]
            if hubs:
                lines.append(f"HUBS: {' '.join(hubs[:5])}")
        
        lines.append("")
        
        curr_dir = None
        for m in sorted(project.modules, key=lambda x: x.path):
            d = str(Path(m.path).parent)
            fn = Path(m.path).name
            
            if d != curr_dir:
                if d != '.':
                    lines.append(f"\n[{d}]")
                curr_dir = d
            
            cls_s = ','.join(c.name for c in m.classes[:3])
            fn_s = ','.join(f.name for f in m.functions[:4] if not f.is_private)
            
            parts = []
            if cls_s:
                parts.append(f"C:{cls_s}")
            if fn_s:
                parts.append(f"F:{fn_s}")
            
            content = ' | '.join(parts) if parts else '-'
            lines.append(f"  {fn} ({m.lines_code}L) {content}")
        
        return '\n'.join(lines)


class JSONGenerator:
    """
    Generates JSON output for machine processing.
    
    Suitable for:
    - RAG (Retrieval-Augmented Generation) systems
    - Database storage
    - Further programmatic analysis
    
    Example:
        >>> generator = JSONGenerator()
        >>> output = generator.generate(project)
        >>> data = json.loads(output)
    """
    
    def generate(self, project: ProjectInfo, flat: bool = False, 
                 detail: str = 'standard') -> str:
        """
        Generate JSON output.
        
        Args:
            project: ProjectInfo analysis results
            flat: If True, generate flat list for easier comparisons
            detail: 'minimal', 'standard', or 'full'
            
        Returns:
            JSON formatted string
        """
        if flat:
            return self._generate_flat(project, detail)
        return self._generate_nested(project, detail)

    def generate_from_module(self, module: ModuleInfo, detail: str = 'full') -> str:
        project = ProjectInfo(
            name=Path(module.path).name,
            root_path=str(Path(module.path).parent),
            languages={module.language: 1},
            modules=[module],
            dependency_graph={},
            dependency_metrics={},
            entrypoints=[],
            similar_functions={},
            total_files=1,
            total_lines=module.lines_total,
            generated_at="",
        )
        return self.generate(project, flat=False, detail=detail)
    
    def _generate_nested(self, project: ProjectInfo, detail: str) -> str:
        """Generate nested JSON structure."""
        def ser_func(f: FunctionInfo) -> dict:
            data = {
                'name': f.name,
                'signature': self._build_signature(f),
            }
            if detail in ('standard', 'full'):
                data['intent'] = f.intent
                data['is_async'] = f.is_async
            if detail == 'full':
                data['params'] = f.params
                data['return_type'] = f.return_type
                data['complexity'] = f.complexity
                data['lines'] = f.lines
                data['is_private'] = f.is_private
            return data
    
    def _field_to_dict(self, field: FieldInfo) -> dict:
        """Serialize dataclass FieldInfo to dictionary."""
        data = {'name': field.name}
        if getattr(field, 'type_annotation', None):
            data['type'] = field.type_annotation
        if getattr(field, 'default', None):
            data['default'] = field.default
        if getattr(field, 'default_factory', None):
            data['factory'] = field.default_factory
        return data
        
        def ser_class(c: ClassInfo) -> dict:
            data = {
                'name': c.name,
                'bases': c.bases,
            }
            if detail in ('standard', 'full'):
                data['docstring'] = c.docstring
            if c.methods:
                data['methods'] = [ser_func(m) for m in c.methods]
            return data
        
        def ser_module(m: ModuleInfo) -> dict:
            data = {
                'path': m.path,
                'language': m.language,
                'lines': m.lines_code,
            }
            if detail in ('standard', 'full'):
                data['imports'] = m.imports[:10]
                data['exports'] = m.exports[:10]
            if m.classes:
                data['classes'] = [ser_class(c) for c in m.classes]
            if m.functions:
                data['functions'] = [ser_func(f) for f in m.functions]
            if detail == 'full' and m.types:
                data['types'] = [{'name': t.name, 'kind': t.kind} for t in m.types]
            return data
        
        data = {
            'name': project.name,
            'statistics': {
                'files': project.total_files,
                'lines': project.total_lines,
                'languages': project.languages,
            },
            'entrypoints': project.entrypoints,
            'modules': [ser_module(m) for m in project.modules],
        }
        
        if detail == 'full':
            data['dependency_graph'] = project.dependency_graph
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _generate_flat(self, project: ProjectInfo, detail: str) -> str:
        """Generate flat JSON list for comparisons."""
        rows = []
        
        for m in project.modules:
            deps = project.dependency_graph.get(m.path, [])
            
            for t in m.types:
                row = self._build_element_row(m, 'type', t.name, t.kind, 
                                             None, deps, detail)
                rows.append(row)
            
            for c in m.classes:
                bases_sig = f"({','.join(c.bases)})" if c.bases else "()"
                row = self._build_element_row(m, 'class', c.name, bases_sig,
                                             None, deps, detail)
                rows.append(row)
                
                for method in c.methods:
                    row = self._build_element_row(
                        m, 'method', f"{c.name}.{method.name}",
                        self._build_signature(method), method, deps, detail
                    )
                    rows.append(row)
            
            for f in m.functions:
                row = self._build_element_row(m, 'function', f.name,
                                             self._build_signature(f), f, deps, detail)
                rows.append(row)
        
        return json.dumps({
            'project': project.name,
            'files': project.total_files,
            'lines': project.total_lines,
            'elements': rows
        }, indent=2, ensure_ascii=False)
    
    def _build_element_row(self, m: ModuleInfo, elem_type: str, name: str,
                          signature: str, f: FunctionInfo, deps: list, 
                          detail: str) -> dict:
        """Build a single element row for flat output."""
        row = {
            'path': m.path,
            'type': elem_type,
            'name': name,
            'signature': signature,
            'language': m.language,
        }
        
        if detail in ('standard', 'full'):
            row['intent'] = f.intent if f else ''
            row['category'] = self._categorize(name)
            row['domain'] = self._extract_domain(m.path)
            row['imports'] = m.imports[:5]
        
        if detail == 'full':
            row['calls'] = f.calls[:5] if f else []
            row['depends_on'] = deps[:5]
            row['lines'] = f.lines if f else 0
            row['complexity'] = f.complexity if f else 1
            row['is_public'] = not f.is_private if f else True
            row['is_async'] = f.is_async if f else False
            row['hash'] = self._compute_hash(name, signature)
        
        return row
    
    def _build_signature(self, f: FunctionInfo) -> str:
        """Build compact signature."""
        raw_params = [p.replace('\n', ' ').replace('  ', ' ').strip() for p in (f.params or [])]
        params_no_self = [p for p in raw_params if p]
        params_no_self = remove_self_from_params(params_no_self)
        params = ','.join(params_no_self[:4])
        if len(params_no_self) > 4:
            params += f'...+{len(params_no_self)-4}'
        ret = f"->{f.return_type}" if f.return_type else ""
        return f"({params}){ret}"
    
    def _categorize(self, name: str) -> str:
        """Categorize by name pattern."""
        return categorize_function(name)
    
    def _extract_domain(self, path: str) -> str:
        """Extract domain from path."""
        return extract_domain(path)
    
    def _compute_hash(self, name: str, signature: str) -> str:
        """Compute short hash."""
        return compute_hash(name, signature, length=8)


# ============================================================================
# YAML Generator
# ============================================================================

class YAMLGenerator:
    """
    Generates YAML output for human-readable representation.
    
    Supports both nested (hierarchical) and flat (table-like) formats.
    
    Example:
        >>> generator = YAMLGenerator()
        >>> output = generator.generate(project, flat=True, detail='standard')
    """
    
    # Key legend for compact format (for LLM transparency)
    KEY_LEGEND = {
        'p': 'path',       # file path
        'l': 'lines',      # line count  
        'i': 'imports',    # import list
        'e': 'exports',    # exported symbols
        'c': 'classes',    # class definitions
        'f': 'functions',  # standalone functions
        'n': 'name',       # symbol name
        'd': 'docstring',  # documentation string
        'b': 'bases',      # base classes
        'm': 'methods',    # class methods
        'props': 'properties',  # class properties
        'sig': 'signature',     # function signature
        'ret': 'return_type',   # return type
        'async': 'is_async',    # async function flag
        'kb': 'kilobytes',      # file size in kilobytes
    }
    
    def generate(self, project: ProjectInfo, flat: bool = False, 
                 detail: str = 'standard', compact: bool = True) -> str:
        """
        Generate YAML output.
        
        Args:
            project: ProjectInfo analysis results
            flat: If True, generate flat list instead of nested structure
            detail: 'minimal', 'standard', or 'full'
            compact: If True, use short keys for smaller output (default: True)
            
        Returns:
            YAML formatted string
        """
        try:
            import yaml
        except ImportError:
            return self._generate_simple_yaml(project, flat, detail)
        
        if flat:
            data = self._build_flat_data(project, detail)
            yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, 
                                sort_keys=False, width=120)
        elif compact:
            # Compact format with short keys and meta.legend structure
            data = self._build_compact_data(project, detail)
            yaml_str = yaml.dump(data, default_flow_style=False, 
                                 allow_unicode=True, sort_keys=False, width=120)
        else:
            data = self._build_nested_data(project, detail)
            yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, 
                                sort_keys=False, width=120)
        
        return yaml_str

    def generate_schema(self, format_type: str = 'compact') -> str:
        """
        Generate JSON Schema for the YAML format.
        
        Args:
            format_type: 'compact', 'full', or 'hybrid' - determines key format
            
        Returns:
            JSON Schema as string
        """
        if format_type == 'hybrid':
            return self._generate_hybrid_schema()
        elif format_type == 'compact':
            return self._generate_compact_schema()
        else:
            return self._generate_full_schema()
    
    def _generate_compact_schema(self) -> str:
        """Generate JSON Schema for compact YAML format with meta.legend."""
        import json
        
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Code2Logic Compact YAML Schema",
            "description": "Schema for Code2Logic compact YAML output with short keys and meta.legend",
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "properties": {
                        "legend": {
                            "type": "object",
                            "description": "Key mapping legend for LLM transparency",
                            "properties": {
                                "p": {"type": "string", "const": "path"},
                                "l": {"type": "string", "const": "lines"},
                                "i": {"type": "string", "const": "imports"},
                                "e": {"type": "string", "const": "exports"},
                                "c": {"type": "string", "const": "classes"},
                                "f": {"type": "string", "const": "functions"},
                                "n": {"type": "string", "const": "name"},
                                "d": {"type": "string", "const": "docstring"},
                                "b": {"type": "string", "const": "bases"},
                                "m": {"type": "string", "const": "methods"},
                                "props": {"type": "string", "const": "properties"},
                                "sig": {"type": "string", "const": "signature (without self)"},
                                "ret": {"type": "string", "const": "return_type"},
                                "async": {"type": "string", "const": "is_async"},
                                "lang": {"type": "string", "const": "language"}
                            },
                            "required": ["p", "l", "i", "c", "f", "n", "d"]
                        }
                    },
                    "required": ["legend"]
                },
                "defaults": {
                    "type": "object",
                    "properties": {
                        "lang": {"type": "string", "description": "Default language"}
                    }
                },
                "modules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "p": {"type": "string", "description": "Module path"},
                            "lang": {"type": "string", "description": "Language (if different from default)"},
                            "l": {"type": "integer", "description": "Lines of code"},
                            "i": {
                                "type": "array", 
                                "items": {"type": "string"},
                                "description": "Compact imports (grouped)"
                            },
                            "e": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Exports"
                            },
                            "c": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "n": {"type": "string", "description": "Class name"},
                                        "b": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Base classes"
                                        },
                                        "d": {"type": "string", "description": "Docstring"},
                                        "props": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Properties"
                                        },
                                        "m": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "n": {"type": "string", "description": "Method name"},
                                                    "sig": {"type": "string", "description": "Signature without self"},
                                                    "ret": {"type": "string", "description": "Return type"},
                                                    "d": {"type": "string", "description": "Docstring/intent"},
                                                    "l": {"type": "integer", "description": "Lines"},
                                                    "async": {"type": "boolean", "description": "Is async"}
                                                },
                                                "required": ["n", "sig"]
                                            },
                                            "description": "Methods"
                                        }
                                    },
                                    "required": ["n"]
                                },
                                "description": "Classes"
                            },
                            "f": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "n": {"type": "string", "description": "Function name"},
                                        "sig": {"type": "string", "description": "Signature"},
                                        "ret": {"type": "string", "description": "Return type"},
                                        "d": {"type": "string", "description": "Docstring/intent"},
                                        "l": {"type": "integer", "description": "Lines"},
                                        "async": {"type": "boolean", "description": "Is async"}
                                    },
                                    "required": ["n", "sig"]
                                },
                                "description": "Functions"
                            }
                        },
                        "required": ["p", "l"]
                    }
                }
            },
            "required": ["meta", "defaults", "modules"]
        }
        
        return json.dumps(schema, indent=2)
    
    def _generate_full_schema(self) -> str:
        """Generate JSON Schema for full YAML format."""
        import json
        
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Code2Logic Full YAML Schema",
            "description": "Schema for Code2Logic full YAML output with complete keys",
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Project name"},
                "statistics": {
                    "type": "object",
                    "properties": {
                        "files": {"type": "integer"},
                        "lines": {"type": "integer"},
                        "languages": {
                            "type": "object",
                            "patternProperties": {
                                ".*": {"type": "integer"}
                            }
                        }
                    }
                },
                "modules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "language": {"type": "string"},
                            "lines": {"type": "integer"},
                            "imports": {"type": "array", "items": {"type": "string"}},
                            "exports": {"type": "array", "items": {"type": "string"}},
                            "classes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "bases": {"type": "array", "items": {"type": "string"}},
                                        "docstring": {"type": "string"},
                                        "properties": {"type": "array", "items": {"type": "string"}},
                                        "methods": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "signature": {"type": "string"},
                                                    "intent": {"type": "string"},
                                                    "lines": {"type": "integer"},
                                                    "is_async": {"type": "boolean"}
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "functions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "signature": {"type": "string"},
                                        "intent": {"type": "string"},
                                        "lines": {"type": "integer"},
                                        "is_async": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return json.dumps(schema, indent=2)
    
    def _generate_hybrid_schema(self) -> str:
        """Generate JSON Schema for hybrid format."""
        import json
        
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Code2Logic Hybrid YAML Schema",
            "description": "Schema for Code2Logic hybrid format combining TOON compactness with YAML completeness",
            "type": "object",
            "properties": {
                "header": {
                    "type": "object",
                    "properties": {
                        "project": {"type": "string"},
                        "files": {"type": "integer"},
                        "lines": {"type": "integer"},
                        "languages": {"type": "object"},
                        "modules_count": {"type": "integer"}
                    },
                    "required": ["project", "files", "lines"]
                },
                "M": {
                    "type": "array",
                    "description": "Compact module overview (path:lines)",
                    "items": {"type": "string"}
                },
                "modules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "p": {"type": "string", "description": "Path"},
                            "lang": {"type": "string", "description": "Language"},
                            "l": {"type": "integer", "description": "Lines"},
                            "i": {"type": "array", "items": {"type": "string"}, "description": "Imports"},
                            "e": {"type": "array", "items": {"type": "string"}, "description": "Exports"},
                            "c": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "n": {"type": "string", "description": "Class name"},
                                        "b": {"type": "array", "items": {"type": "string"}, "description": "Bases"},
                                        "d": {"type": "string", "description": "Docstring"},
                                        "props": {"type": "array", "items": {"type": "string"}, "description": "Properties"},
                                        "m": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "n": {"type": "string", "description": "Method name"},
                                                    "sig": {"type": "string", "description": "Signature"},
                                                    "ret": {"type": "string", "description": "Return type"},
                                                    "d": {"type": "string", "description": "Intent"},
                                                    "async": {"type": "boolean", "description": "Is async"}
                                                },
                                                "required": ["n", "sig"]
                                            },
                                            "description": "Methods"
                                        }
                                    },
                                    "required": ["n"]
                                },
                                "description": "Classes"
                            },
                            "f": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "n": {"type": "string", "description": "Function name"},
                                        "sig": {"type": "string", "description": "Signature"},
                                        "ret": {"type": "string", "description": "Return type"},
                                        "d": {"type": "string", "description": "Intent"},
                                        "async": {"type": "boolean", "description": "Is async"}
                                    },
                                    "required": ["n", "sig"]
                                },
                                "description": "Functions"
                            },
                            "const": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "n": {"type": "string", "description": "Constant name"},
                                        "t": {"type": "string", "description": "Type"}
                                    }
                                },
                                "description": "Constants"
                            },
                            "dataclasses": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Dataclass names"
                            },
                            "conditional_imports": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Imports in try/except blocks"
                            }
                        },
                        "required": ["p", "l"]
                    }
                },
                "defaults": {
                    "type": "object",
                    "properties": {
                        "lang": {"type": "string", "description": "Default language"}
                    }
                }
            },
            "required": ["header", "M", "modules", "defaults"]
        }
        
        return json.dumps(schema, indent=2)

    def generate_hybrid(self, project: ProjectInfo, detail: str = 'standard') -> str:
        """
        Generate hybrid format combining TOON compactness with YAML completeness.
        
        Features:
        - Compact module overview (like TOON header)
        - Full YAML structure for detailed information
        - Includes constants, dataclasses, default values
        - ~70% of YAML size, ~90% of information
        - Best balance for LLM code generation
        """
        try:
            import yaml
        except ImportError:
            return self._generate_simple_hybrid(project)
        
        # Detect default language
        default_lang = max(project.languages.items(), key=lambda x: x[1])[0] if project.languages else 'python'
        
        total_kb = bytes_to_kb(getattr(project, 'total_bytes', 0))
        # Compact module overview
        modules_overview = []
        for m in project.modules:
            file_kb = bytes_to_kb(getattr(m, 'file_bytes', 0))
            entry = f"{m.path}:{m.lines_code}"
            if file_kb:
                entry = f"{entry}:{file_kb}kb"
            modules_overview.append(entry)
        
        # Build detailed module data with enhanced information
        detailed_modules = []
        for m in project.modules:
            file_kb = bytes_to_kb(getattr(m, 'file_bytes', 0))
            mod_data = {
                'p': m.path,  # path
            }
            
            # Only add language if different from default
            if m.language != default_lang:
                mod_data['lang'] = m.language
            
            # Add line count
            mod_data['l'] = m.lines_code  # lines
            if file_kb:
                mod_data['kb'] = file_kb
            
            # Enhanced imports with grouping
            if m.imports:
                compact_imports = self._compact_imports(m.imports[:15])
                if compact_imports:
                    mod_data['i'] = compact_imports  # imports
            
            # Exports
            if m.exports:
                mod_data['e'] = m.exports[:10]  # exports
            
            # Constants
            if hasattr(m, 'constants') and m.constants:
                const_data = []
                for const in m.constants:
                    if isinstance(const, str):
                        # Handle string constants (from UniversalParser)
                        const_data.append({'n': const})
                    else:
                        # Handle ConstantInfo objects (from TreeSitter parser)
                        const_data.append(self._constant_to_dict(const))
                if const_data:
                    mod_data['const'] = const_data
            
            # TYPE_CHECKING imports
            if hasattr(m, 'type_checking_imports') and m.type_checking_imports:
                mod_data['type_checking'] = m.type_checking_imports[:10]
            
            # Conditional imports
            if hasattr(m, 'optional_imports') and m.optional_imports:
                optional_data = []
                for opt in m.optional_imports:
                    opt_dict = {'module': opt.module}
                    if opt.flag_name:
                        opt_dict['flag'] = opt.flag_name
                    if not opt.fallback_value:
                        opt_dict['fallback'] = False
                    if opt.imports:
                        opt_dict['imports'] = opt.imports
                    optional_data.append(opt_dict)
                if optional_data:
                    mod_data['optional_imports'] = optional_data
            
            # Aliases
            if hasattr(m, 'aliases') and m.aliases:
                mod_data['aliases'] = m.aliases
            
            # Classes with enhanced information
            if m.classes:
                classes_data = []
                for c in m.classes[:8]:  # Limit to 8 classes per module
                    cls_data = {
                        'n': c.name,  # name
                    }
                    
                    # Bases
                    if c.bases:
                        cls_data['b'] = c.bases  # bases
                    
                    # Enhanced docstring
                    if c.docstring:
                        doc = c.docstring.split('\n')[0][:80].strip()
                        if doc:
                            cls_data['d'] = doc  # docstring
                    
                    # NEW: Class decorators
                    if hasattr(c, 'decorators') and c.decorators:
                        cls_data['dec'] = c.decorators
                    
                    # NEW: Dataclass fields
                    if hasattr(c, 'is_dataclass') and c.is_dataclass and hasattr(c, 'fields') and c.fields:
                        fields_data = []
                        for field in c.fields:
                            field_dict = {'n': field.name}
                            if field.type_annotation:
                                field_dict['t'] = field.type_annotation
                            if field.default:
                                field_dict['default'] = field.default
                            if field.default_factory:
                                field_dict['factory'] = field.default_factory
                            fields_data.append(field_dict)
                        if fields_data:
                            cls_data['fields'] = fields_data
                    
                    # NEW: Class attributes
                    if hasattr(c, 'attributes') and c.attributes:
                        attrs_data = []
                        for attr in c.attributes:
                            attr_dict = {'n': attr.name}
                            if attr.type_annotation:
                                attr_dict['t'] = attr.type_annotation
                            attrs_data.append(attr_dict)
                        if attrs_data:
                            cls_data['attrs'] = attrs_data
                    
                    # Methods with enhanced signatures
                    if c.methods:
                        methods_data = []
                        for method in c.methods[:10]:  # Limit to 10 methods per class
                            method_data = self._method_to_dict(method, detail)
                            if method_data:
                                methods_data.append(method_data)
                        
                        if methods_data:
                            cls_data['m'] = methods_data
                    
                    classes_data.append(cls_data)
                
                if classes_data:
                    mod_data['c'] = classes_data
            
            # Functions with enhanced information
            if m.functions:
                functions_data = []
                for f in m.functions[:12]:  # Limit to 12 functions per module
                    # Skip if not a FunctionInfo object (safety check)
                    if not hasattr(f, 'params'):
                        continue
                    
                    func_data = {
                        'n': f.name,  # name
                        'sig': self._build_enhanced_signature(f),  # enhanced signature
                    }
                    
                    # Return type
                    if f.return_type and f.return_type != 'None':
                        func_data['ret'] = f.return_type
                    
                    # NEW: Function decorators
                    if hasattr(f, 'decorators') and f.decorators:
                        func_data['dec'] = f.decorators
                    
                    # Intent
                    if f.intent:
                        intent = f.intent.replace('\n', ' ')[:60].strip()
                        if intent:
                            func_data['d'] = intent
                    
                    # Async flag
                    if f.is_async:
                        func_data['async'] = True
                    
                    functions_data.append(func_data)
                
                if functions_data:
                    mod_data['f'] = functions_data
            
            # Add constants/types information (if available)
            constants = self._extract_constants(m)
            if constants:
                mod_data.setdefault('const', []).extend(constants)
            
            # Add dataclass information
            dataclasses = self._extract_dataclasses(m)
            if dataclasses:
                mod_data['dataclasses'] = dataclasses
            
            # Add conditional imports information
            conditional_imports = self._extract_conditional_imports(m)
            if conditional_imports:
                mod_data['conditional_imports'] = conditional_imports
            
            detailed_modules.append(mod_data)
        
        # Build final hybrid structure
        hybrid_data = {
            # Compact header (like TOON)
            'header': {
                'project': project.name,
                'files': project.total_files,
                'lines': project.total_lines,
                'kb': total_kb,
                'languages': dict(project.languages),
                'modules_count': len(project.modules)
            },
            
            # Compact module overview
            'M': modules_overview,  # M for modules (like TOON)
            
            # Full YAML details
            'modules': detailed_modules,
            
            # Defaults
            'defaults': {'lang': default_lang}
        }
        
        yaml_str = yaml.dump(hybrid_data, default_flow_style=False, 
                           allow_unicode=True, sort_keys=False, width=120)
        return yaml_str

    def _build_enhanced_signature(self, f: FunctionInfo) -> str:
        """Build enhanced signature with better parameter handling."""
        if not f.params:
            return '()'
        
        # Remove self/cls parameters
        params = []
        for p in f.params:
            p_clean = p.replace('\n', ' ').strip()
            if p_clean and p_clean not in ('self', 'cls') and not p_clean.startswith('self:'):
                params.append(p_clean)
        
        if not params:
            return '()'
        
        # Limit to 5 params for compactness
        if len(params) > 5:
            params = params[:5] + [f'...+{len(params)-5}']
        
        return f"({','.join(params)})"

    def _extract_constants(self, module: ModuleInfo) -> list:
        """Extract constants and type definitions from module."""
        constants = []
        
        # Look for common constant patterns
        for name in module.exports:
            # Check if it looks like a constant (UPPER_CASE)
            if name.isupper() and '_' in name:
                constants.append({
                    'n': name,
                    't': 'constant'  # type hint
                })
        
        return constants[:5]  # Limit to 5 constants

    def _extract_dataclasses(self, module: ModuleInfo) -> list:
        """Extract dataclass information from classes."""
        dataclasses = []
        
        for cls in module.classes:
            # Check if class is marked as dataclass
            if getattr(cls, 'is_dataclass', False):
                dataclasses.append(cls.name)
        
        return dataclasses[:3]  # Limit to 3 dataclasses
    
    def _extract_conditional_imports(self, module: ModuleInfo) -> list:
        """Extract conditional imports from constants."""
        conditional_imports = []
        
        for const in module.constants:
            if isinstance(const, str) and const.startswith('conditional:'):
                # Extract the import name
                import_name = const.replace('conditional:', '')
                conditional_imports.append(import_name)
        
        return conditional_imports[:5]  # Limit to 5 conditional imports

    def generate_from_module(self, module: ModuleInfo, detail: str = 'full') -> str:
        project = ProjectInfo(
            name=Path(module.path).name,
            root_path=str(Path(module.path).parent),
            languages={module.language: 1},
            modules=[module],
            dependency_graph={},
            dependency_metrics={},
            entrypoints=[],
            similar_functions={},
            total_files=1,
            total_lines=module.lines_total,
            generated_at="",
        )
        return self.generate(project, flat=False, detail=detail)
    
    def _build_flat_data(self, project: ProjectInfo, detail: str) -> dict:
        """Build flat data structure optimized for comparisons."""
        rows = []
        
        for m in project.modules:
            # Add module-level types
            for t in m.types:
                row = self._build_row(m.path, 'type', t.name, '', m.language, detail, project)
                rows.append(row)
            
            # Add classes and their methods
            for c in m.classes:
                bases_str = ','.join(c.bases) if c.bases else ''
                row = self._build_row(m.path, 'class', c.name, f"({bases_str})", 
                                     m.language, detail, project)
                row['docstring'] = c.docstring[:50] if c.docstring else ''
                rows.append(row)
                
                for method in c.methods:
                    row = self._build_method_row(m.path, c.name, method, m.language, 
                                                detail, project, m.imports)
                    rows.append(row)
            
            # Add standalone functions
            for f in m.functions:
                row = self._build_function_row(m.path, f, m.language, detail, 
                                              project, m.imports)
                rows.append(row)
        
        return {
            'project': project.name,
            'files': project.total_files,
            'lines': project.total_lines,
            'elements': rows
        }
    
    def _build_nested_data(self, project: ProjectInfo, detail: str) -> dict:
        """Build nested hierarchical data structure."""
        modules = []
        module_overview = []
        for m in project.modules:
            module_overview.append(f"{m.path}:{m.lines_code}")
        for m in project.modules:
            module_data = {
                'path': m.path,
                'language': m.language,
                'lines': m.lines_code,
            }
            
            if detail in ('standard', 'full'):
                module_data['imports'] = m.imports[:10]
                module_data['exports'] = m.exports[:10]
            
            if m.classes:
                module_data['classes'] = []
                dataclass_summary = []
                for c in m.classes:
                    cls_data = {
                        'name': c.name,
                        'bases': c.bases,
                        'docstring': c.docstring[:80] if c.docstring else '',
                    }
                    if getattr(c, 'properties', None):
                        cls_data['properties'] = c.properties[:20]
                    # Include properties (critical for dataclass reproduction)
                    if getattr(c, 'is_dataclass', False):
                        cls_data['dataclass'] = True
                        if getattr(c, 'fields', None):
                            cls_data['fields'] = [
                                self._field_to_dict(field)
                                for field in c.fields[:15]
                            ]
                        dataclass_summary.append({
                            'name': c.name,
                            'fields': [
                                self._field_to_dict(field)
                                for field in c.fields[:15]
                            ] if getattr(c, 'fields', None) else []
                        })
                    if c.methods:
                        cls_data['methods'] = [
                            self._method_to_dict(method, detail)
                            for method in c.methods[:15]
                        ]
                    module_data['classes'].append(cls_data)
                if dataclass_summary:
                    module_data['dataclasses'] = dataclass_summary[:10]
            
            if m.functions:
                module_data['functions'] = [
                    self._function_to_dict(f, detail)
                    for f in m.functions[:20]
                ]
            
            constant_entries = self._constants_for_module_verbose(m, limit=12)
            if constant_entries:
                module_data['constants'] = constant_entries
            
            modules.append(module_data)
        
        return {
            'project': project.name,
            'statistics': {
                'files': project.total_files,
                'lines': project.total_lines,
                'languages': project.languages,
            },
            'modules': modules
        }
    
    def _build_row(self, path: str, elem_type: str, name: str, signature: str,
                   language: str, detail: str, project: ProjectInfo) -> dict:
        """Build a single row for flat output."""
        row = {
            'path': path,
            'type': elem_type,
            'name': name,
            'signature': signature,
            'language': language,
        }
        return row
    
    def _build_function_row(self, path: str, f: FunctionInfo, language: str,
                           detail: str, project: ProjectInfo, imports: list) -> dict:
        """Build row for standalone function."""
        sig = self._build_signature(f)
        row = {
            'path': path,
            'type': 'function',
            'name': f.name,
            'signature': sig,
            'language': language,
        }
        
        if detail in ('standard', 'full'):
            row['intent'] = f.intent[:60] if f.intent else ''
            row['category'] = self._categorize(f.name)
            row['domain'] = self._extract_domain(path)
            row['imports'] = ','.join(imports[:5])
        
        if detail == 'full':
            row['calls'] = ','.join(f.calls[:5])
            row['lines'] = f.lines
            row['complexity'] = f.complexity
            row['is_async'] = f.is_async
            row['is_public'] = not f.is_private
            row['hash'] = self._compute_hash(f.name, sig)
        
        return row
    
    def _build_method_row(self, path: str, class_name: str, f: FunctionInfo,
                         language: str, detail: str, project: ProjectInfo,
                         imports: list) -> dict:
        """Build row for class method."""
        sig = self._build_signature(f)
        row = {
            'path': path,
            'type': 'method',
            'name': f"{class_name}.{f.name}",
            'signature': sig,
            'language': language,
        }
        
        if detail in ('standard', 'full'):
            row['intent'] = f.intent[:60] if f.intent else ''
            row['category'] = self._categorize(f.name)
            row['domain'] = self._extract_domain(path)
            row['imports'] = ','.join(imports[:5])
        
        if detail == 'full':
            row['calls'] = ','.join(f.calls[:5])
            row['lines'] = f.lines
            row['complexity'] = f.complexity
            row['is_async'] = f.is_async
            row['is_public'] = not f.is_private
            row['hash'] = self._compute_hash(f"{class_name}.{f.name}", sig)
        
        return row
    
    def _function_to_dict(self, f: FunctionInfo, detail: str) -> dict:
        """Convert function to dict for nested output."""
        # Clean function name (remove any newlines or special chars)
        name = f.name.replace('\n', '').strip() if f.name else ''
        
        data = {
            'name': name,
            'signature': self._build_signature(f),
        }
        if detail in ('standard', 'full'):
            # Clean intent - remove newlines and limit length
            intent = f.intent.replace('\n', ' ').strip()[:100] if f.intent else ''
            data['intent'] = intent
        if detail == 'full':
            data['lines'] = f.lines
            data['is_async'] = f.is_async
        return data
    
    def _method_to_dict(self, f: FunctionInfo, detail: str) -> dict:
        """Convert method to dictionary for YAML output."""
        if not f:
            return None
            
        # Enhanced signature with defaults
        sig = self._build_enhanced_signature(f)
        
        data = {
            'n': f.name,
            'sig': sig,
        }
        
        # Return type
        if f.return_type and f.return_type != 'None':
            data['ret'] = f.return_type
        
        # NEW: Method decorators
        if hasattr(f, 'decorators') and f.decorators:
            data['dec'] = f.decorators
        
        # Intent/docstring
        if f.intent:
            intent = f.intent.replace('\n', ' ')[:60].strip()
            if intent:
                data['d'] = intent
        
        # Async flag
        if f.is_async:
            data['async'] = True
            
        # NEW: Method type flags
        if hasattr(f, 'is_static') and f.is_static:
            data['static'] = True
        if hasattr(f, 'is_classmethod') and f.is_classmethod:
            data['classmethod'] = True
        if hasattr(f, 'is_property') and f.is_property:
            data['property'] = True
        
        return data
    
    def _build_signature(self, f: FunctionInfo) -> str:
        """Build compact signature string."""
        # Clean params - remove newlines and extra spaces
        clean_params = []
        raw_params = [p.replace('\n', ' ').replace('  ', ' ').strip() for p in (f.params or [])]
        raw_params = [p for p in raw_params if p]
        params_no_self = remove_self_from_params(raw_params)
        for p in params_no_self[:6]:
            p_clean = p.replace('\n', ' ').replace('  ', ' ').strip()
            if p_clean:
                clean_params.append(p_clean)
        
        params = ','.join(clean_params)
        if len(params_no_self) > 6:
            params += f'...+{len(params_no_self)-6}'
        
        ret = f"->{f.return_type}" if f.return_type else ""
        return f"({params}){ret}"
    
    def _categorize(self, name: str) -> str:
        """Categorize function by name pattern."""
        return categorize_function(name)
    
    def _extract_domain(self, path: str) -> str:
        """Extract domain from file path."""
        return extract_domain(path)
    
    def _compute_hash(self, name: str, signature: str) -> str:
        """Compute short hash for quick comparison."""
        return compute_hash(name, signature, length=8)
    
    def _generate_simple_yaml(self, project: ProjectInfo, flat: bool, 
                              detail: str) -> str:
        """Fallback YAML generation without pyyaml."""
        lines = [f"project: {project.name}"]
        lines.append(f"files: {project.total_files}")
        lines.append(f"lines: {project.total_lines}")
        lines.append("modules:")
        
        for m in project.modules:
            lines.append(f"  - path: {m.path}")
            lines.append(f"    language: {m.language}")
            lines.append(f"    lines: {m.lines_code}")
            if m.classes:
                lines.append("    classes:")
                for c in m.classes[:10]:
                    lines.append(f"      - name: {c.name}")
                    if c.docstring:
                        doc = c.docstring.split('\n')[0][:60]
                        lines.append(f"        docstring: \"{doc}\"")
                    if c.bases:
                        lines.append(f"        bases: [{', '.join(c.bases)}]")
                    if c.properties:
                        lines.append("        properties:")
                        for prop in c.properties[:15]:
                            lines.append(f"          - {prop}")
            if m.functions:
                lines.append("    functions:")
                for f in m.functions[:15]:
                    sig = self._build_signature(f)
                    lines.append(f"      - name: {f.name}")
                    lines.append(f"        signature: {sig}")
                    if f.intent:
                        lines.append(f"        intent: {f.intent[:50]}")
                    lines.append(f"        lines: {f.lines}")
                    lines.append(f"        is_async: {str(f.is_async).lower()}")
        
        return '\n'.join(lines)
    
    def _build_compact_data(self, project: ProjectInfo, detail: str) -> dict:
        """Build compact data structure with short keys."""
        # Detect default language
        default_lang = max(project.languages.items(), key=lambda x: x[1])[0] if project.languages else 'python'
        
        total_bytes = getattr(project, 'total_bytes', 0)
        total_kb = bytes_to_kb(total_bytes)
        modules = []
        module_overview = []
        for m in project.modules:
            file_bytes = getattr(m, 'file_bytes', 0)
            file_kb = bytes_to_kb(file_bytes)
            overview_entry = f"{m.path}:{m.lines_code}"
            if file_kb:
                overview_entry = f"{overview_entry}:{file_kb}kb"
            module_overview.append(overview_entry)
            mod_data = {
                'p': m.path,  # path
            }
            
            # Only add language if different from default
            if m.language != default_lang:
                mod_data['lang'] = m.language
            
            # Add line count as comment in path
            mod_data['l'] = m.lines_code  # lines
            if file_kb:
                mod_data['kb'] = file_kb
            
            if detail in ('standard', 'full'):
                # Deduplicate and compact imports
                compact_imports = self._compact_imports(m.imports[:15])
                if compact_imports:
                    mod_data['i'] = compact_imports  # imports
                if m.exports:
                    mod_data['e'] = m.exports[:10]  # exports
            
            # Classes with compact format
            if m.classes:
                mod_data['c'] = [self._compact_class(c, detail) for c in m.classes[:10]]
            
            # Functions with compact format
            if m.functions:
                mod_data['f'] = [self._compact_function(f, detail) for f in m.functions[:15] 
                               if hasattr(f, 'params')]  # Skip non-FunctionInfo objects
            
            const_entries = self._constants_for_module(m, limit=8)
            if const_entries:
                mod_data['const'] = const_entries
            
            modules.append(mod_data)
        
        return {
            'header': {
                'project': project.name,
                'files': project.total_files,
                'lines': project.total_lines,
                'kb': total_kb,
                'languages': project.languages,
                'modules_count': len(project.modules),
            },
            'M': module_overview,
            'meta': {
                'legend': self.KEY_LEGEND.copy()
            },
            'defaults': {'lang': default_lang},
            'modules': modules
        }
    
    def _compact_imports(self, imports: list) -> list:
        """Deduplicate and compact imports (typing.Dict, typing.List -> typing.{Dict,List})."""
        if not imports:
            return []
        return compact_imports(deduplicate_imports(list(imports)), max_items=10)
    
    def _compact_class(self, cls: ClassInfo, detail: str) -> dict:
        """Generate compact class representation."""
        data = {
            'n': cls.name,  # name
        }
        
        # Only add bases if non-empty
        if cls.bases:
            data['b'] = cls.bases  # bases
        
        # Truncated docstring
        if cls.docstring:
            doc = cls.docstring.split('\n')[0][:60].strip()
            if doc:
                data['d'] = doc  # docstring
        
        # Properties (important for dataclasses)
        if cls.properties:
            data['props'] = cls.properties[:15]
        
        # Methods in compact format
        if cls.methods:
            data['m'] = [self._compact_method(m, detail) for m in cls.methods[:12]]
        
        if getattr(cls, 'decorators', None):
            data['dec'] = cls.decorators[:5]
        
        if getattr(cls, 'attributes', None):
            attrs_data = []
            for attr in cls.attributes[:10]:
                attr_dict = {'n': attr.name}
                if attr.type_annotation:
                    attr_dict['t'] = attr.type_annotation
                if hasattr(attr, 'set_in_init'):
                    attr_dict['init'] = attr.set_in_init
                attrs_data.append(attr_dict)
            if attrs_data:
                data['attrs'] = attrs_data
        
        if cls.is_dataclass:
            data['dataclass'] = True
            if getattr(cls, 'fields', None):
                fields_data = []
                for field in cls.fields[:10]:
                    field_dict = {'n': field.name}
                    if field.type_annotation:
                        field_dict['t'] = field.type_annotation
                    if field.default:
                        field_dict['default'] = field.default
                    if field.default_factory:
                        field_dict['factory'] = field.default_factory
                    fields_data.append(field_dict)
                if fields_data:
                    data['fields'] = fields_data
        
        return data
    
    def _compact_function(self, f: FunctionInfo, detail: str) -> dict:
        """Generate compact function representation."""
        data = {
            'n': f.name,  # name
            'sig': self._build_compact_signature(f),  # signature without self
        }
        
        if f.return_type:
            data['ret'] = f.return_type
        
        if detail in ('standard', 'full') and f.intent:
            intent = f.intent.replace('\n', ' ')[:50].strip()
            if intent:
                data['d'] = intent  # docstring/intent
        
        if detail == 'full':
            data['l'] = f.lines  # lines
            if f.is_async:
                data['async'] = True
            if getattr(f, 'decorators', None):
                data['dec'] = f.decorators[:3]
        
        # Surface defaults if available
        if getattr(f, 'params_with_defaults', None):
            defaults = []
            for name, val in list(f.params_with_defaults.items())[:4]:
                defaults.append(f"{name}={val}")
            if defaults:
                data['defaults'] = defaults
        
        return data
    
    def _compact_method(self, f: FunctionInfo, detail: str) -> dict:
        """Generate compact method representation (same as function)."""
        return self._compact_function(f, detail)
    
    def _build_compact_signature(self, f: FunctionInfo) -> str:
        """Build compact signature without 'self' and with clean formatting."""
        clean_params = []
        for p in f.params[:6]:
            p_clean = p.replace('\n', ' ').replace('  ', ' ').strip()
            # Skip 'self' parameter
            if p_clean and p_clean not in ('self', 'cls') and not p_clean.startswith('self:'):
                clean_params.append(p_clean)
        
        params = ', '.join(clean_params)
        if len(f.params) > 6:
            params += f', ...+{len(f.params)-6}'
        
        if params:
            return f"({params})"
        return "()"

    def _constants_for_module(self, module: ModuleInfo, limit: int = 10) -> list:
        """Convert module constants into compact dictionaries."""
        constants_attr = getattr(module, 'constants', []) or []
        constants: list = []
        for const in constants_attr:
            if isinstance(const, str):
                if const.startswith('conditional:'):
                    continue  # handled separately for optional imports
                constants.append({'n': const})
            else:
                constants.append(self._constant_to_dict(const))
            if len(constants) >= limit:
                break
        return constants

    def _constant_to_dict(self, constant: ConstantInfo) -> dict:
        """Serialize ConstantInfo into a compact dictionary."""
        data = {'n': constant.name}
        if getattr(constant, 'type_annotation', None):
            data['t'] = constant.type_annotation
        if getattr(constant, 'value_keys', None):
            data['keys'] = constant.value_keys[:10]
        elif getattr(constant, 'value', None):
            snippet = constant.value.replace('\n', ' ').strip()
            if len(snippet) > 120:
                snippet = snippet[:117] + '...'
            data['v'] = snippet
        return data


# ============================================================================
# CSV Generator  
# ============================================================================

class CSVGenerator:
    """
    Generates CSV output optimized for LLM processing.
    
    CSV is the most token-efficient format (~50% smaller than JSON).
    Each row is self-contained with full path for better LLM context.
    
    Columns by detail level:
    - minimal: path, type, name, signature, language (5 cols)
    - standard: + intent, category, domain, imports (9 cols)
    - full: + calls, depends_on, lines, complexity, is_public, is_async, hash (16 cols)
    
    Example:
        >>> generator = CSVGenerator()
        >>> output = generator.generate(project, detail='standard')
    """
    
    # Column definitions by detail level
    COLUMNS = {
        'minimal': ['path', 'type', 'name', 'signature', 'language'],
        'standard': ['path', 'type', 'name', 'signature', 'language', 
                    'intent', 'category', 'domain', 'imports'],
        'full': ['path', 'type', 'name', 'signature', 'language',
                'intent', 'category', 'domain', 'imports', 'calls',
                'depends_on', 'lines', 'complexity', 'is_public', 'is_async', 'hash']
    }
    
    def generate(self, project: ProjectInfo, detail: str = 'standard') -> str:
        """
        Generate CSV output.
        
        Args:
            project: ProjectInfo analysis results
            detail: 'minimal', 'standard', or 'full'
            
        Returns:
            CSV formatted string
        """
        import csv
        import io
        
        columns = self.COLUMNS.get(detail, self.COLUMNS['standard'])
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        
        for m in project.modules:
            # Get module dependencies
            deps = project.dependency_graph.get(m.path, [])
            deps_str = ','.join(deps[:5])
            
            # Add types
            for t in m.types:
                row = self._build_row(m, 'type', t.name, t.kind, [], deps_str, 
                                     detail, project)
                writer.writerow(row)
            
            # Add classes and methods
            for c in m.classes:
                bases_sig = f"({','.join(c.bases)})" if c.bases else "()"
                row = self._build_row(m, 'class', c.name, bases_sig, [], deps_str,
                                     detail, project)
                writer.writerow(row)
                
                for method in c.methods:
                    row = self._build_function_row(
                        m, 'method', f"{c.name}.{method.name}", method, 
                        deps_str, detail, project
                    )
                    writer.writerow(row)
            
            # Add functions
            for f in m.functions:
                row = self._build_function_row(m, 'function', f.name, f, 
                                              deps_str, detail, project)
                writer.writerow(row)
        
        return output.getvalue()
    
    def _build_row(self, m: ModuleInfo, elem_type: str, name: str, 
                   signature: str, calls: list, deps: str,
                   detail: str, project: ProjectInfo) -> dict:
        """Build a single CSV row."""
        row = {
            'path': m.path,
            'type': elem_type,
            'name': name,
            'signature': signature,
            'language': m.language,
        }
        
        if detail in ('standard', 'full'):
            row['intent'] = ''
            row['category'] = self._categorize(name)
            row['domain'] = self._extract_domain(m.path)
            row['imports'] = ','.join(m.imports[:5])
        
        if detail == 'full':
            row['calls'] = ','.join(calls[:5])
            row['depends_on'] = deps
            row['lines'] = 0
            row['complexity'] = 1
            row['is_public'] = True
            row['is_async'] = False
            row['hash'] = self._compute_hash(name, signature)
        
        return row
    
    def _build_function_row(self, m: ModuleInfo, elem_type: str, name: str,
                           f: FunctionInfo, deps: str, detail: str,
                           project: ProjectInfo) -> dict:
        """Build CSV row for function/method."""
        sig = self._build_signature(f)
        
        row = {
            'path': m.path,
            'type': elem_type,
            'name': name,
            'signature': sig,
            'language': m.language,
        }
        
        if detail in ('standard', 'full'):
            row['intent'] = self._escape_csv(f.intent[:60]) if f.intent else ''
            row['category'] = self._categorize(f.name)
            row['domain'] = self._extract_domain(m.path)
            row['imports'] = ','.join(m.imports[:5])
        
        if detail == 'full':
            row['calls'] = ','.join(f.calls[:5])
            row['depends_on'] = deps
            row['lines'] = f.lines
            row['complexity'] = f.complexity
            row['is_public'] = not f.is_private
            row['is_async'] = f.is_async
            row['hash'] = self._compute_hash(name, sig)
        
        return row
    
    def _build_signature(self, f: FunctionInfo) -> str:
        """Build compact signature."""
        raw_params = [p.replace('\n', ' ').replace('  ', ' ').strip() for p in (f.params or [])]
        raw_params = [p for p in raw_params if p]
        params_no_self = remove_self_from_params(raw_params)
        params = ','.join(
            (p.split(':')[0] if ':' in p else p)
            for p in params_no_self[:4]
        )
        if len(params_no_self) > 4:
            params += f'...+{len(params_no_self)-4}'
        ret = f"->{f.return_type}" if f.return_type else ""
        return f"({params}){ret}"
    
    def _categorize(self, name: str) -> str:
        """Categorize function by name pattern."""
        return categorize_function(name)
    
    def _extract_domain(self, path: str) -> str:
        """Extract domain from file path."""
        return extract_domain(path)
    
    def _compute_hash(self, name: str, signature: str) -> str:
        """Compute short hash for quick comparison."""
        import hashlib
        content = f"{name}:{signature}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _escape_csv(self, text: str) -> str:
        """Escape text for CSV (remove newlines, limit commas)."""
        if not text:
            return ''
        return text.replace('\n', ' ').replace('\r', '').replace('"', "'")
