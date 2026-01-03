"""
Output generators for code2logic.

This module contains 5 generators for different output formats:
- CSV: Tabular data for spreadsheet analysis
- YAML: Human-readable structured data
- JSON: Machine-readable structured data  
- Compact: Minimal text representation
- Markdown: Documentation-friendly format
"""

import csv
import json
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any

from .models import Project, Module, Function, Class, Dependency


class BaseGenerator(ABC):
    """Base class for all output generators."""
    
    @abstractmethod
    def generate(self, project: Project, output_path: str) -> None:
        """Generate output for the given project."""
        pass
    
    def _ensure_output_dir(self, output_path: str) -> Path:
        """Ensure output directory exists."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class CSVGenerator(BaseGenerator):
    """Generate CSV output for spreadsheet analysis."""
    
    def generate(self, project: Project, output_path: str) -> None:
        """Generate CSV files for modules, functions, classes, and dependencies."""
        output_path = self._ensure_output_dir(output_path)
        
        # Generate modules CSV
        modules_path = output_path.with_suffix('.modules.csv')
        with open(modules_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'path', 'lines_of_code', 'functions', 'classes', 'imports'])
            for module in project.modules:
                writer.writerow([
                    module.name,
                    module.path,
                    module.lines_of_code,
                    len(module.functions),
                    len(module.classes),
                    ', '.join(module.imports)
                ])
        
        # Generate functions CSV
        functions_path = output_path.with_suffix('.functions.csv')
        with open(functions_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['module', 'name', 'lines_of_code', 'complexity', 'docstring'])
            for module in project.modules:
                for func in module.functions:
                    writer.writerow([
                        module.name,
                        func.name,
                        func.lines_of_code,
                        func.complexity,
                        func.docstring is not None
                    ])
        
        # Generate classes CSV
        classes_path = output_path.with_suffix('.classes.csv')
        with open(classes_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['module', 'name', 'methods', 'base_classes', 'lines_of_code'])
            for module in project.modules:
                for cls in module.classes:
                    writer.writerow([
                        module.name,
                        cls.name,
                        len(cls.methods),
                        ', '.join(cls.base_classes),
                        cls.lines_of_code
                    ])
        
        # Generate dependencies CSV
        dependencies_path = output_path.with_suffix('.dependencies.csv')
        with open(dependencies_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target', 'type', 'strength'])
            for dep in project.dependencies:
                writer.writerow([dep.source, dep.target, dep.type, dep.strength])


class YAMLGenerator(BaseGenerator):
    """Generate YAML output for human-readable structured data."""
    
    def generate(self, project: Project, output_path: str) -> None:
        """Generate YAML representation of the project."""
        output_path = self._ensure_output_dir(output_path)
        
        data = self._project_to_dict(project)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2, allow_unicode=True)
    
    def _project_to_dict(self, project: Project) -> Dict[str, Any]:
        """Convert project to dictionary for YAML serialization."""
        return {
            'project': {
                'name': project.name,
                'path': project.path,
                'metadata': project.metadata,
                'modules': [
                    {
                        'name': module.name,
                        'path': module.path,
                        'lines_of_code': module.lines_of_code,
                        'imports': module.imports,
                        'functions': [
                            {
                                'name': func.name,
                                'lines_of_code': func.lines_of_code,
                                'complexity': func.complexity,
                                'parameters': func.parameters,
                                'docstring': func.docstring,
                                'code': func.code[:200] + '...' if len(func.code) > 200 else func.code
                            }
                            for func in module.functions
                        ],
                        'classes': [
                            {
                                'name': cls.name,
                                'lines_of_code': cls.lines_of_code,
                                'base_classes': cls.base_classes,
                                'methods': [
                                    {
                                        'name': method.name,
                                        'lines_of_code': method.lines_of_code,
                                        'complexity': method.complexity,
                                        'parameters': method.parameters,
                                        'docstring': method.docstring
                                    }
                                    for method in cls.methods
                                ]
                            }
                            for cls in module.classes
                        ]
                    }
                    for module in project.modules
                ],
                'dependencies': [
                    {
                        'source': dep.source,
                        'target': dep.target,
                        'type': dep.type,
                        'strength': dep.strength
                    }
                    for dep in project.dependencies
                ],
                'similarities': project.similarities
            }
        }


class JSONGenerator(BaseGenerator):
    """Generate JSON output for machine-readable structured data."""
    
    def generate(self, project: Project, output_path: str) -> None:
        """Generate JSON representation of the project."""
        output_path = self._ensure_output_dir(output_path)
        
        data = self._project_to_dict(project)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _project_to_dict(self, project: Project) -> Dict[str, Any]:
        """Convert project to dictionary for JSON serialization."""
        return {
            'project': {
                'name': project.name,
                'path': project.path,
                'metadata': project.metadata,
                'statistics': {
                    'total_modules': len(project.modules),
                    'total_functions': sum(len(m.functions) for m in project.modules),
                    'total_classes': sum(len(m.classes) for m in project.modules),
                    'total_dependencies': len(project.dependencies),
                    'total_lines_of_code': sum(m.lines_of_code for m in project.modules)
                },
                'modules': [
                    {
                        'name': module.name,
                        'path': module.path,
                        'lines_of_code': module.lines_of_code,
                        'imports': module.imports,
                        'functions': [
                            {
                                'name': func.name,
                                'lines_of_code': func.lines_of_code,
                                'complexity': func.complexity,
                                'parameters': func.parameters,
                                'has_docstring': func.docstring is not None
                            }
                            for func in module.functions
                        ],
                        'classes': [
                            {
                                'name': cls.name,
                                'lines_of_code': cls.lines_of_code,
                                'base_classes': cls.base_classes,
                                'method_count': len(cls.methods)
                            }
                            for cls in module.classes
                        ]
                    }
                    for module in project.modules
                ],
                'dependencies': [
                    {
                        'source': dep.source,
                        'target': dep.target,
                        'type': dep.type,
                        'strength': dep.strength
                    }
                    for dep in project.dependencies
                ],
                'similarities': project.similarities
            }
        }


class CompactGenerator(BaseGenerator):
    """Generate minimal text representation."""
    
    def generate(self, project: Project, output_path: str) -> None:
        """Generate compact text representation."""
        output_path = self._ensure_output_dir(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Project: {project.name} ({project.path})\n")
            f.write(f"Modules: {len(project.modules)}\n")
            f.write(f"Functions: {sum(len(m.functions) for m in project.modules)}\n")
            f.write(f"Classes: {sum(len(m.classes) for m in project.modules)}\n")
            f.write(f"Dependencies: {len(project.dependencies)}\n")
            f.write(f"LOC: {sum(m.lines_of_code for m in project.modules)}\n\n")
            
            for module in project.modules:
                f.write(f"{module.name} ({module.lines_of_code} LOC)\n")
                if module.functions:
                    f.write(f"  Functions: {', '.join(f.name for f in module.functions)}\n")
                if module.classes:
                    f.write(f"  Classes: {', '.join(c.name for c in module.classes)}\n")
                if module.imports:
                    f.write(f"  Imports: {', '.join(module.imports[:5])}")
                    if len(module.imports) > 5:
                        f.write("...")
                    f.write("\n")
                f.write("\n")


class MarkdownGenerator(BaseGenerator):
    """Generate Markdown output for documentation."""
    
    def generate(self, project: Project, output_path: str) -> None:
        """Generate Markdown documentation."""
        output_path = self._ensure_output_dir(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# {project.name}\n\n")
            f.write(f"**Path:** `{project.path}`\n\n")
            
            # Statistics
            total_functions = sum(len(m.functions) for m in project.modules)
            total_classes = sum(len(m.classes) for m in project.modules)
            total_loc = sum(m.lines_of_code for m in project.modules)
            
            f.write("## Statistics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Modules | {len(project.modules)} |\n")
            f.write(f"| Functions | {total_functions} |\n")
            f.write(f"| Classes | {total_classes} |\n")
            f.write(f"| Dependencies | {len(project.dependencies)} |\n")
            f.write(f"| Lines of Code | {total_loc} |\n\n")
            
            # Modules
            f.write("## Modules\n\n")
            for module in project.modules:
                f.write(f"### {module.name}\n\n")
                f.write(f"**Path:** `{module.path}`\n\n")
                f.write(f"**Lines of Code:** {module.lines_of_code}\n\n")
                
                if module.imports:
                    f.write("**Imports:**\n")
                    for imp in module.imports:
                        f.write(f"- `{imp}`\n")
                    f.write("\n")
                
                if module.functions:
                    f.write("**Functions:**\n")
                    for func in module.functions:
                        doc_indicator = " 📝" if func.docstring else ""
                        f.write(f"- `{func.name}()` ({func.lines_of_code} LOC, complexity: {func.complexity}){doc_indicator}\n")
                    f.write("\n")
                
                if module.classes:
                    f.write("**Classes:**\n")
                    for cls in module.classes:
                        f.write(f"- `{cls.name}` ({len(cls.methods)} methods")
                        if cls.base_classes:
                            f.write(f", inherits from: {', '.join(cls.base_classes)}")
                        f.write(")\n")
                    f.write("\n")
                
                f.write("---\n\n")
            
            # Dependencies
            if project.dependencies:
                f.write("## Dependencies\n\n")
                f.write("| Source | Target | Type | Strength |\n")
                f.write("|--------|--------|------|----------|\n")
                for dep in project.dependencies:
                    f.write(f"| `{dep.source}` | `{dep.target}` | {dep.type} | {dep.strength:.2f} |\n")
                f.write("\n")
            
            # Similarities
            if project.similarities:
                f.write("## Similarities\n\n")
                for similarity in project.similarities:
                    f.write(f"- **{similarity['item1']}** ↔ **{similarity['item2']}** "
                           f"(similarity: {similarity['score']:.2f})\n")


# Generator registry
GENERATORS = {
    'csv': CSVGenerator,
    'yaml': YAMLGenerator,
    'json': JSONGenerator,
    'compact': CompactGenerator,
    'markdown': MarkdownGenerator,
}


def get_generator(format_name: str) -> BaseGenerator:
    """Get generator by format name."""
    if format_name not in GENERATORS:
        raise ValueError(f"Unsupported format: {format_name}")
    return GENERATORS[format_name]()
