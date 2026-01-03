"""
Markdown Hybrid Format Generator for Code2Logic.

Generates optimized Markdown containing:
- File tree structure
- Embedded Gherkin codeblocks for behavior
- Embedded YAML codeblocks for data structures
- Compact code summaries

This hybrid format aims to combine the best of all formats
while being more token-efficient than individual formats.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from .models import ProjectInfo, ModuleInfo, FunctionInfo, ClassInfo
from .gherkin import GherkinGenerator
from .generators import YAMLGenerator


@dataclass
class MarkdownSpec:
    """Markdown specification for a project."""
    content: str
    file_count: int
    total_chars: int
    sections: Dict[str, int]  # section name -> char count


class MarkdownHybridGenerator:
    """
    Generates optimized Markdown hybrid format.
    
    Combines:
    - File tree (compact overview)
    - Gherkin (behaviors/functions)
    - YAML (data structures/classes)
    - Metadata (imports, dependencies)
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.gherkin_gen = GherkinGenerator()
        self.yaml_gen = YAMLGenerator()
    
    def generate(self, project: ProjectInfo, detail: str = 'full') -> MarkdownSpec:
        """Generate Markdown hybrid specification."""
        sections = {}
        parts = []
        
        # Header
        header = self._generate_header(project)
        parts.append(header)
        sections['header'] = len(header)
        
        # File tree
        tree = self._generate_tree(project)
        parts.append(tree)
        sections['tree'] = len(tree)
        
        # Imports summary
        imports = self._generate_imports(project)
        parts.append(imports)
        sections['imports'] = len(imports)
        
        # Classes as YAML
        classes_yaml = self._generate_classes_yaml(project)
        parts.append(classes_yaml)
        sections['classes'] = len(classes_yaml)
        
        # Functions as Gherkin
        functions_gherkin = self._generate_functions_gherkin(project)
        parts.append(functions_gherkin)
        sections['functions'] = len(functions_gherkin)
        
        # Dependencies
        deps = self._generate_dependencies(project)
        parts.append(deps)
        sections['dependencies'] = len(deps)
        
        content = '\n'.join(parts)
        
        return MarkdownSpec(
            content=content,
            file_count=len(project.modules),
            total_chars=len(content),
            sections=sections,
        )
    
    def _generate_header(self, project: ProjectInfo) -> str:
        """Generate header section."""
        return f"""# {project.name} - Logic Specification

> Auto-generated hybrid format for code reproduction
> Files: {len(project.modules)} | Classes: {sum(len(m.classes) for m in project.modules)} | Functions: {sum(len(m.functions) for m in project.modules)}

"""
    
    def _generate_tree(self, project: ProjectInfo) -> str:
        """Generate file tree section."""
        lines = ["## 📁 File Structure\n", "```"]
        
        # Group by directory
        dirs: Dict[str, List[str]] = {}
        for module in project.modules:
            path = Path(module.path)
            dir_name = str(path.parent) if path.parent != Path('.') else '.'
            if dir_name not in dirs:
                dirs[dir_name] = []
            dirs[dir_name].append(path.name)
        
        for dir_name, files in sorted(dirs.items()):
            if dir_name != '.':
                lines.append(f"{dir_name}/")
            for f in sorted(files):
                prefix = "  " if dir_name != '.' else ""
                lines.append(f"{prefix}├── {f}")
        
        lines.append("```\n")
        return '\n'.join(lines)
    
    def _generate_imports(self, project: ProjectInfo) -> str:
        """Generate imports summary."""
        lines = ["## 📦 Dependencies\n"]
        
        all_imports = set()
        for module in project.modules:
            all_imports.update(module.imports)
        
        # Categorize imports
        stdlib = []
        third_party = []
        local = []
        
        stdlib_modules = {
            'os', 'sys', 'json', 'typing', 'pathlib', 'dataclasses',
            're', 'ast', 'abc', 'collections', 'functools', 'itertools',
            'datetime', 'logging', 'argparse', 'subprocess', 'shutil',
        }
        
        for imp in sorted(all_imports):
            base = imp.split('.')[0]
            if base in stdlib_modules:
                stdlib.append(imp)
            elif base.startswith('.') or base == project.name:
                local.append(imp)
            else:
                third_party.append(imp)
        
        if stdlib:
            lines.append("**Standard Library:**")
            lines.append(f"`{', '.join(stdlib[:10])}`")
            if len(stdlib) > 10:
                lines.append(f"... and {len(stdlib)-10} more")
            lines.append("")
        
        if third_party:
            lines.append("**Third Party:**")
            lines.append(f"`{', '.join(third_party[:10])}`")
            lines.append("")
        
        if local:
            lines.append("**Local:**")
            lines.append(f"`{', '.join(local[:10])}`")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _generate_classes_yaml(self, project: ProjectInfo) -> str:
        """Generate classes as YAML codeblock."""
        lines = ["## 🏗️ Data Structures\n"]
        
        has_classes = False
        for module in project.modules:
            if module.classes:
                has_classes = True
                lines.append(f"### {Path(module.path).name}\n")
                lines.append("```yaml")
                
                for cls in module.classes:
                    lines.append(f"{cls.name}:")
                    if cls.docstring:
                        doc = cls.docstring.split('\n')[0][:60]
                        lines.append(f"  description: \"{doc}\"")
                    if cls.bases:
                        lines.append(f"  bases: [{', '.join(cls.bases)}]")
                    if cls.properties:
                        lines.append("  properties:")
                        for prop in cls.properties[:10]:
                            lines.append(f"    - {prop}")
                    if cls.methods:
                        lines.append("  methods:")
                        for method in cls.methods[:10]:
                            sig = f"{method.name}({', '.join(method.params[:3])})"
                            lines.append(f"    - {sig}")
                
                lines.append("```\n")
        
        if not has_classes:
            lines.append("*No classes defined*\n")
        
        return '\n'.join(lines)
    
    def _generate_functions_gherkin(self, project: ProjectInfo) -> str:
        """Generate functions as Gherkin codeblock."""
        lines = ["## ⚡ Functions & Behaviors\n"]
        
        has_functions = False
        for module in project.modules:
            # Include both module functions and class methods
            all_funcs = list(module.functions)
            for cls in module.classes:
                all_funcs.extend(cls.methods)
            
            if all_funcs:
                has_functions = True
                lines.append(f"### {Path(module.path).name}\n")
                lines.append("```gherkin")
                lines.append(f"Feature: {Path(module.path).stem}")
                
                for func in all_funcs[:15]:  # Limit for token efficiency
                    lines.append(f"\n  Scenario: {func.name}")
                    
                    # Parameters
                    if func.params:
                        params = ', '.join(func.params[:5])
                        lines.append(f"    Given parameters: {params}")
                    
                    # Return type
                    if func.return_type and func.return_type != 'None':
                        lines.append(f"    Then returns: {func.return_type}")
                    
                    # Brief description
                    if func.docstring:
                        doc = func.docstring.split('\n')[0][:50]
                        lines.append(f"    # {doc}")
                
                lines.append("```\n")
        
        if not has_functions:
            lines.append("*No functions defined*\n")
        
        return '\n'.join(lines)
    
    def _generate_dependencies(self, project: ProjectInfo) -> str:
        """Generate module dependencies section."""
        lines = ["## 🔗 Module Dependencies\n"]
        
        # Build dependency graph
        deps = {}
        for module in project.modules:
            name = Path(module.path).stem
            local_deps = [
                imp for imp in module.imports 
                if imp.startswith('.') or any(
                    imp.startswith(Path(m.path).stem) 
                    for m in project.modules
                )
            ]
            if local_deps:
                deps[name] = local_deps[:5]
        
        if deps:
            lines.append("```")
            for module, imports in sorted(deps.items()):
                lines.append(f"{module} -> {', '.join(imports)}")
            lines.append("```\n")
        else:
            lines.append("*No internal dependencies*\n")
        
        return '\n'.join(lines)


def generate_markdown_hybrid(project: ProjectInfo, detail: str = 'full') -> str:
    """Convenience function to generate Markdown hybrid format."""
    generator = MarkdownHybridGenerator()
    spec = generator.generate(project, detail)
    return spec.content


def generate_file_markdown(file_path: str) -> str:
    """Generate Markdown hybrid for a single file."""
    from .analyzer import ProjectAnalyzer
    
    path = Path(file_path)
    analyzer = ProjectAnalyzer()
    project = analyzer.analyze(str(path.parent))
    
    # Filter to just this file
    target_name = path.name
    project.modules = [m for m in project.modules if Path(m.path).name == target_name]
    
    generator = MarkdownHybridGenerator()
    spec = generator.generate(project)
    return spec.content
