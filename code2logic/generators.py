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
    DependencyNode
)


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
        
        params = f.params[:4]
        if len(f.params) > 4:
            params.append(f"...+{len(f.params)-4}")
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
        params = ','.join(f.params[:4])
        if len(f.params) > 4:
            params += f'...+{len(f.params)-4}'
        ret = f"->{f.return_type}" if f.return_type else ""
        return f"({params}){ret}"
    
    def _categorize(self, name: str) -> str:
        """Categorize by name pattern."""
        name_lower = name.lower().split('.')[-1]
        patterns = {
            'read': ('get', 'fetch', 'find', 'load', 'read'),
            'create': ('create', 'add', 'insert', 'new', 'make'),
            'update': ('update', 'set', 'modify', 'edit'),
            'delete': ('delete', 'remove', 'clear'),
            'validate': ('validate', 'check', 'verify', 'is', 'has'),
            'transform': ('convert', 'transform', 'parse', 'format'),
        }
        for cat, verbs in patterns.items():
            if any(v in name_lower for v in verbs):
                return cat
        return 'other'
    
    def _extract_domain(self, path: str) -> str:
        """Extract domain from path."""
        parts = path.lower().replace('\\', '/').split('/')
        domains = ['auth', 'user', 'order', 'payment', 'config', 'util', 
                   'api', 'service', 'model', 'validation', 'generator']
        for part in parts:
            for domain in domains:
                if domain in part:
                    return domain
        return parts[-2] if len(parts) > 1 else 'root'
    
    def _compute_hash(self, name: str, signature: str) -> str:
        """Compute short hash."""
        import hashlib
        return hashlib.md5(f"{name}:{signature}".encode()).hexdigest()[:8]


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
    
    def generate(self, project: ProjectInfo, flat: bool = False, 
                 detail: str = 'standard') -> str:
        """
        Generate YAML output.
        
        Args:
            project: ProjectInfo analysis results
            flat: If True, generate flat list instead of nested structure
            detail: 'minimal', 'standard', or 'full'
            
        Returns:
            YAML formatted string
        """
        try:
            import yaml
        except ImportError:
            # Fallback to simple YAML generation
            return self._generate_simple_yaml(project, flat, detail)
        
        if flat:
            data = self._build_flat_data(project, detail)
        else:
            data = self._build_nested_data(project, detail)
        
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, 
                        sort_keys=False, width=120)

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
                for c in m.classes:
                    cls_data = {
                        'name': c.name,
                        'bases': c.bases,
                        'docstring': c.docstring[:80] if c.docstring else '',
                    }
                    # Include properties (critical for dataclass reproduction)
                    if c.properties:
                        cls_data['properties'] = c.properties[:20]
                    if c.methods:
                        cls_data['methods'] = [
                            self._method_to_dict(method, detail)
                            for method in c.methods[:15]
                        ]
                    module_data['classes'].append(cls_data)
            
            if m.functions:
                module_data['functions'] = [
                    self._function_to_dict(f, detail)
                    for f in m.functions[:20]
                ]
            
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
        """Convert method to dict for nested output."""
        return self._function_to_dict(f, detail)
    
    def _build_signature(self, f: FunctionInfo) -> str:
        """Build compact signature string."""
        # Clean params - remove newlines and extra spaces
        clean_params = []
        for p in f.params[:6]:
            p_clean = p.replace('\n', ' ').replace('  ', ' ').strip()
            if p_clean:
                clean_params.append(p_clean)
        
        params = ','.join(clean_params)
        if len(f.params) > 6:
            params += f'...+{len(f.params)-6}'
        
        ret = f"->{f.return_type}" if f.return_type else ""
        return f"({params}){ret}"
    
    def _categorize(self, name: str) -> str:
        """Categorize function by name pattern."""
        name_lower = name.lower()
        if any(v in name_lower for v in ('get', 'fetch', 'find', 'load', 'read')):
            return 'read'
        if any(v in name_lower for v in ('create', 'add', 'insert', 'new', 'make')):
            return 'create'
        if any(v in name_lower for v in ('update', 'set', 'modify', 'edit')):
            return 'update'
        if any(v in name_lower for v in ('delete', 'remove', 'clear')):
            return 'delete'
        if any(v in name_lower for v in ('validate', 'check', 'verify', 'is', 'has')):
            return 'validate'
        if any(v in name_lower for v in ('convert', 'transform', 'parse', 'format')):
            return 'transform'
        if any(v in name_lower for v in ('init', 'setup', 'configure')):
            return 'lifecycle'
        return 'other'
    
    def _extract_domain(self, path: str) -> str:
        """Extract domain from file path."""
        parts = path.lower().replace('\\', '/').split('/')
        domains = ['auth', 'user', 'order', 'payment', 'product', 'cart', 
                   'config', 'util', 'api', 'service', 'model', 'controller']
        for part in parts:
            for domain in domains:
                if domain in part:
                    return domain
        return parts[-2] if len(parts) > 1 else 'root'
    
    def _compute_hash(self, name: str, signature: str) -> str:
        """Compute short hash for quick comparison."""
        import hashlib
        content = f"{name}:{signature}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
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
        params = ','.join(p.split(':')[0] if ':' in p else p for p in f.params[:4])
        if len(f.params) > 4:
            params += f'...+{len(f.params)-4}'
        ret = f"->{f.return_type}" if f.return_type else ""
        return f"({params}){ret}"
    
    def _categorize(self, name: str) -> str:
        """Categorize function by name pattern."""
        name_lower = name.lower().split('.')[-1]  # Get last part for methods
        if any(v in name_lower for v in ('get', 'fetch', 'find', 'load', 'read', 'query')):
            return 'read'
        if any(v in name_lower for v in ('create', 'add', 'insert', 'new', 'make', 'build')):
            return 'create'
        if any(v in name_lower for v in ('update', 'set', 'modify', 'edit', 'patch')):
            return 'update'
        if any(v in name_lower for v in ('delete', 'remove', 'clear', 'destroy')):
            return 'delete'
        if any(v in name_lower for v in ('validate', 'check', 'verify', 'is', 'has', 'can')):
            return 'validate'
        if any(v in name_lower for v in ('convert', 'transform', 'parse', 'format', 'to')):
            return 'transform'
        if any(v in name_lower for v in ('init', 'setup', 'configure', 'start', 'stop')):
            return 'lifecycle'
        if any(v in name_lower for v in ('send', 'emit', 'notify', 'publish')):
            return 'communicate'
        return 'other'
    
    def _extract_domain(self, path: str) -> str:
        """Extract domain from file path."""
        parts = path.lower().replace('\\', '/').split('/')
        domains = ['auth', 'user', 'order', 'payment', 'product', 'cart', 
                   'config', 'util', 'api', 'service', 'model', 'controller',
                   'validation', 'test', 'generator', 'parser', 'llm']
        for part in parts:
            for domain in domains:
                if domain in part:
                    return domain
        # Return parent folder name
        return parts[-2] if len(parts) > 1 else 'root'
    
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
