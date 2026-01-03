#!/usr/bin/env python3
"""
Example: Generate API Documentation from Code Analysis.

Creates comprehensive API documentation in multiple formats:
1. Markdown API reference
2. OpenAPI/Swagger spec
3. TypeScript definitions
4. Python stubs

Usage:
    python api_documentation.py /path/to/project
    python api_documentation.py /path/to/project --format openapi
    python api_documentation.py /path/to/project --output docs/
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic import analyze_project, ProjectInfo


def generate_markdown_docs(project: ProjectInfo) -> str:
    """Generate Markdown API documentation."""
    lines = [
        f"# {project.name} API Reference",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> Files: {project.total_files} | Lines: {project.total_lines}",
        "",
        "## Table of Contents",
        "",
    ]
    
    # TOC
    for module in sorted(project.modules, key=lambda m: m.path):
        name = Path(module.path).stem
        lines.append(f"- [{name}](#{name.replace('.', '-').lower()})")
    
    lines.append("")
    
    # Modules
    for module in sorted(project.modules, key=lambda m: m.path):
        name = Path(module.path).stem
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"**Path:** `{module.path}`")
        lines.append(f"**Lines:** {module.lines_code}")
        lines.append("")
        
        if module.docstring:
            lines.append(module.docstring)
            lines.append("")
        
        # Classes
        if module.classes:
            lines.append("### Classes")
            lines.append("")
            for cls in module.classes:
                lines.append(f"#### `{cls.name}`")
                if cls.bases:
                    lines.append(f"**Extends:** {', '.join(cls.bases)}")
                if cls.docstring:
                    lines.append(f"\n{cls.docstring}")
                lines.append("")
                
                if cls.methods:
                    lines.append("**Methods:**")
                    lines.append("")
                    for method in cls.methods:
                        sig = f"({', '.join(method.params)})"
                        ret = f" -> {method.return_type}" if method.return_type else ""
                        async_prefix = "async " if method.is_async else ""
                        lines.append(f"- `{async_prefix}{method.name}{sig}{ret}`")
                        if method.intent:
                            lines.append(f"  - {method.intent}")
                    lines.append("")
        
        # Functions
        if module.functions:
            lines.append("### Functions")
            lines.append("")
            for func in module.functions:
                sig = f"({', '.join(func.params)})"
                ret = f" -> {func.return_type}" if func.return_type else ""
                async_prefix = "async " if func.is_async else ""
                
                lines.append(f"#### `{async_prefix}{func.name}{sig}{ret}`")
                lines.append("")
                if func.docstring:
                    lines.append(func.docstring)
                    lines.append("")
                if func.intent:
                    lines.append(f"**Intent:** {func.intent}")
                    lines.append("")
                if func.raises:
                    lines.append(f"**Raises:** {', '.join(func.raises)}")
                    lines.append("")
        
        lines.append("---")
        lines.append("")
    
    return '\n'.join(lines)


def generate_openapi_spec(project: ProjectInfo) -> Dict[str, Any]:
    """Generate OpenAPI/Swagger specification from project analysis."""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": f"{project.name} API",
            "version": "1.0.0",
            "description": f"Auto-generated from {project.total_files} files",
        },
        "paths": {},
        "components": {
            "schemas": {}
        }
    }
    
    # Find API-like functions
    for module in project.modules:
        for func in module.functions:
            # Detect HTTP handlers
            method = None
            if any(d in ['get', 'Get', 'GET'] for d in func.decorators) or func.name.startswith('get_'):
                method = 'get'
            elif any(d in ['post', 'Post', 'POST'] for d in func.decorators) or func.name.startswith('create_'):
                method = 'post'
            elif any(d in ['put', 'Put', 'PUT'] for d in func.decorators) or func.name.startswith('update_'):
                method = 'put'
            elif any(d in ['delete', 'Delete', 'DELETE'] for d in func.decorators) or func.name.startswith('delete_'):
                method = 'delete'
            
            if method:
                path = f"/{func.name.replace('_', '-')}"
                spec["paths"][path] = {
                    method: {
                        "summary": func.intent or func.name,
                        "operationId": func.name,
                        "parameters": [
                            {"name": p.split(':')[0], "in": "query", "schema": {"type": "string"}}
                            for p in func.params if p and p != 'self'
                        ],
                        "responses": {
                            "200": {"description": "Success"}
                        }
                    }
                }
        
        # Generate schemas from classes
        for cls in module.classes:
            schema = {
                "type": "object",
                "properties": {}
            }
            for prop in cls.properties:
                schema["properties"][prop] = {"type": "string"}
            if schema["properties"]:
                spec["components"]["schemas"][cls.name] = schema
    
    return spec


def generate_typescript_defs(project: ProjectInfo) -> str:
    """Generate TypeScript type definitions."""
    lines = [
        "// Auto-generated TypeScript definitions",
        f"// Source: {project.name}",
        f"// Generated: {datetime.now().isoformat()}",
        "",
    ]
    
    type_map = {
        'str': 'string',
        'int': 'number',
        'float': 'number',
        'bool': 'boolean',
        'list': 'Array<any>',
        'dict': 'Record<string, any>',
        'None': 'void',
        'Any': 'any',
        'Optional': 'undefined |',
    }
    
    def convert_type(py_type: Optional[str]) -> str:
        if not py_type:
            return 'any'
        for py, ts in type_map.items():
            py_type = py_type.replace(py, ts)
        return py_type
    
    for module in project.modules:
        module_name = Path(module.path).stem.replace('-', '_')
        
        # Interfaces from classes
        for cls in module.classes:
            lines.append(f"export interface {cls.name} {{")
            for prop in cls.properties:
                lines.append(f"  {prop}: any;")
            for method in cls.methods:
                params = ', '.join(
                    f"{p.split(':')[0]}: {convert_type(p.split(':')[1] if ':' in p else None)}"
                    for p in method.params if p and p != 'self'
                )
                ret = convert_type(method.return_type)
                lines.append(f"  {method.name}({params}): {ret};")
            lines.append("}")
            lines.append("")
        
        # Functions
        if module.functions:
            lines.append(f"// Functions from {module.path}")
            for func in module.functions:
                params = ', '.join(
                    f"{p.split(':')[0]}: {convert_type(p.split(':')[1] if ':' in p else None)}"
                    for p in func.params if p
                )
                ret = convert_type(func.return_type)
                async_prefix = "async " if func.is_async else ""
                lines.append(f"export {async_prefix}function {func.name}({params}): {ret};")
            lines.append("")
    
    return '\n'.join(lines)


def generate_python_stubs(project: ProjectInfo) -> str:
    """Generate Python stub files (.pyi)."""
    lines = [
        "# Auto-generated Python stub file",
        f"# Source: {project.name}",
        "from typing import Any, Optional, List, Dict",
        "",
    ]
    
    for module in project.modules:
        lines.append(f"# {module.path}")
        
        for cls in module.classes:
            bases = f"({', '.join(cls.bases)})" if cls.bases else ""
            lines.append(f"class {cls.name}{bases}:")
            if not cls.methods and not cls.properties:
                lines.append("    ...")
            else:
                for prop in cls.properties:
                    lines.append(f"    {prop}: Any")
                for method in cls.methods:
                    params = ', '.join(method.params) or 'self'
                    ret = f" -> {method.return_type}" if method.return_type else ""
                    async_def = "async def" if method.is_async else "def"
                    lines.append(f"    {async_def} {method.name}({params}){ret}: ...")
            lines.append("")
        
        for func in module.functions:
            params = ', '.join(func.params) if func.params else ""
            ret = f" -> {func.return_type}" if func.return_type else ""
            async_def = "async def" if func.is_async else "def"
            lines.append(f"{async_def} {func.name}({params}){ret}: ...")
        
        lines.append("")
    
    return '\n'.join(lines)


def main():
    """Generate API documentation."""
    if len(sys.argv) < 2:
        print("Usage: python api_documentation.py /path/to/project [options]")
        print("")
        print("Options:")
        print("  --format FORMAT   Output format: markdown, openapi, typescript, stubs, all")
        print("  --output DIR      Output directory (default: ./api_docs)")
        sys.exit(1)
    
    project_path = sys.argv[1]
    output_format = 'all'
    output_dir = Path('./api_docs')
    
    if '--format' in sys.argv:
        idx = sys.argv.index('--format')
        output_format = sys.argv[idx + 1]
    
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        output_dir = Path(sys.argv[idx + 1])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing: {project_path}")
    project = analyze_project(project_path)
    print(f"Found {project.total_files} files, {project.total_lines} lines")
    
    generated = []
    
    if output_format in ('markdown', 'all'):
        md = generate_markdown_docs(project)
        (output_dir / 'API.md').write_text(md)
        generated.append('API.md')
    
    if output_format in ('openapi', 'all'):
        spec = generate_openapi_spec(project)
        (output_dir / 'openapi.json').write_text(json.dumps(spec, indent=2))
        generated.append('openapi.json')
    
    if output_format in ('typescript', 'all'):
        ts = generate_typescript_defs(project)
        (output_dir / 'types.d.ts').write_text(ts)
        generated.append('types.d.ts')
    
    if output_format in ('stubs', 'all'):
        stubs = generate_python_stubs(project)
        (output_dir / f'{project.name}.pyi').write_text(stubs)
        generated.append(f'{project.name}.pyi')
    
    print(f"\nGenerated files in {output_dir}:")
    for f in generated:
        size = (output_dir / f).stat().st_size
        print(f"  {f}: {size:,} bytes")


if __name__ == '__main__':
    main()
