from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..generators import JSONGenerator, YAMLGenerator
from ..gherkin import GherkinGenerator
from ..logicml import LogicMLGenerator
from ..markdown_format import MarkdownHybridGenerator
from ..models import ProjectInfo
from ..toon_format import TOONGenerator


def create_single_project(module_info, file_path: Path) -> ProjectInfo:
    return ProjectInfo(
        name=file_path.name,
        root_path=str(file_path.parent),
        languages={getattr(module_info, "language", "python") or "python": 1},
        modules=[module_info],
        dependency_graph={},
        dependency_metrics={},
        entrypoints=[],
        similar_functions={},
        total_files=1,
        total_lines=module_info.lines_total,
        generated_at=datetime.now().isoformat(),
    )


def generate_spec(project: ProjectInfo, fmt: str) -> str:
    if fmt == "gherkin":
        gen = GherkinGenerator()
        return gen.generate(project)
    if fmt == "yaml":
        gen = YAMLGenerator()
        return gen.generate(project, detail="full")
    if fmt == "markdown":
        gen = MarkdownHybridGenerator()
        spec = gen.generate(project)
        return spec.content
    if fmt == "json":
        gen = JSONGenerator()
        return gen.generate(project, detail="full")
    if fmt == "logicml":
        gen = LogicMLGenerator()
        spec = gen.generate(project)
        return spec.content
    if fmt == "toon":
        gen = TOONGenerator()
        return gen.generate(project, detail="full")
    return ""


def _generate_token_json(project: ProjectInfo) -> str:
    """Generate compact, token-friendly JSON spec (used by examples/11_token_benchmark.py)."""
    data = {
        "project": project.name,
        "files": project.total_files,
        "lines": project.total_lines,
        "modules": [],
    }

    for m in project.modules:
        module: dict = {
            "path": m.path,
            "language": m.language,
            "imports": m.imports[:10],
            "exports": m.exports[:10],
        }

        if m.classes:
            module["classes"] = []
            for c in m.classes[:20]:
                cls = {
                    "name": c.name,
                    "bases": c.bases,
                    "doc": (c.docstring[:80] if c.docstring else ""),
                    "properties": c.properties[:15],
                    "methods": [
                        {
                            "name": method.name,
                            "params": method.params[:5],
                            "returns": method.return_type or "None",
                            "doc": (method.intent[:50] if method.intent else ""),
                            "async": method.is_async,
                        }
                        for method in c.methods[:15]
                    ],
                }
                module["classes"].append(cls)

        if m.functions:
            module["functions"] = [
                {
                    "name": f.name,
                    "params": f.params[:6],
                    "returns": f.return_type or "None",
                    "doc": (f.intent[:60] if f.intent else ""),
                    "async": f.is_async,
                    "lines": f.lines,
                }
                for f in m.functions[:20]
            ]

        data["modules"].append(module)

    return json.dumps(data, indent=2)


def _generate_token_json_compact(project: ProjectInfo) -> str:
    data = json.loads(_generate_token_json(project))
    return json.dumps(data, separators=(",", ":"))


def generate_spec_token(project: ProjectInfo, fmt: str) -> str:
    """Generate spec optimized for token benchmark (keeps historical behavior).

    Notes:
    - json/json_compact use the token-friendly JSON representation.
    - other formats delegate to generate_spec.
    """
    if fmt == "json":
        return _generate_token_json(project)
    if fmt == "json_compact":
        return _generate_token_json_compact(project)
    return generate_spec(project, fmt)


def get_async_reproduction_prompt(spec: str, fmt: str, file_name: str, with_tests: bool = False) -> str:
    base_prompts = {
        "gherkin": f"""Generate Python code from this Gherkin/BDD specification.
Implement all scenarios as working, production-ready code.

{spec[:6000]}

Requirements:
- Generate complete, working Python code for {file_name}
- Include all imports
- Use type hints
- Add docstrings""",
        "yaml": f"""Generate Python code from this YAML specification.
Match the structure exactly with all classes and functions.

{spec[:6000]}

Requirements:
- Generate complete, working Python code for {file_name}
- Include all imports
- Use type hints
- Implement all methods with actual logic""",
        "markdown": f"""Generate Python code from this Markdown specification.
It contains embedded Gherkin (behaviors) and YAML (structures).

{spec[:6000]}

Requirements:
- Generate complete, working Python code for {file_name}
- Include all imports
- Implement all classes and functions
- Use type hints throughout""",
    }

    prompt = base_prompts.get(fmt, base_prompts["yaml"])

    if with_tests:
        prompt += """

IMPORTANT: Also generate a unittest test class at the end of the file.
Include tests for each function/method with at least 2 test cases each.
Use unittest.TestCase as base class.
Name the test class Test<ClassName> or TestFunctions."""

    return prompt


def get_token_reproduction_prompt(spec: str, fmt: str, file_name: str) -> str:
    format_hints = {
        "json": "Parse the JSON structure and implement all classes and functions.",
        "json_compact": "Parse the compact JSON and implement all elements.",
        "yaml": "Parse the YAML structure and implement all classes and functions with exact signatures.",
        "gherkin": "Implement scenarios as SIMPLE, MINIMAL Python code. NO extra error classes, NO over-engineering. Keep code short and direct.",
        "markdown": "Parse embedded Gherkin (behaviors) and YAML (structures).",
        "logicml": """Parse LogicML and generate VALID Python code:
- 'sig: (params) -> Type' = def func(params) -> Type
- 'sig: async (params)' = async def func(params)
- 'sig: @property (self)' = @property decorator
- 'bases: [BaseModel]' = class X(BaseModel) with Field()
- 'type: re-export' = from .module import X
CRITICAL: Ensure valid syntax - balanced brackets, proper indentation, no undefined variables.""",
        "toon": """Parse TOON (Token-Oriented Object Notation) format carefully:

STRUCTURE:
- 'imports[N]: mod1,mod2' = import statements to include
- 'classes[N]{name,bases,decorators,props,methods}:' = class definitions
- 'functions[N]{name,sig,decorators,async,category,lines}:' = function definitions
- 'function_docs:' section = docstrings/intent for each function

SIGNATURE FORMAT '(params)->ReturnType':
- 'sig: (self;x: int;y: str)->bool' = def func(self, x: int, y: str) -> bool
- 'sig: (self)->None' = def func(self) -> None
- Semicolons separate params, '->' indicates return type

DECORATORS:
- 'decorators: @property' = add @property decorator
- 'decorators: @staticmethod|@cache' = multiple decorators

CRITICAL: Use imports[], function_docs, and exact signatures to reproduce code accurately.""",
    }

    max_spec = 5000
    spec_truncated = spec[:max_spec] if len(spec) > max_spec else spec

    prompt = f"""Generate Python code from this {fmt.upper()} specification.
{format_hints.get(fmt, '')}

{spec_truncated}

Requirements:
- Complete, working Python code for {file_name}
- Include imports and type hints
- Implement all functions with actual logic

```python
"""
    return prompt


def get_simple_reproduction_prompt(spec: str, fmt: str, file_name: str) -> str:
    prompts = {
        "gherkin": f"""Generate Python code from this Gherkin/BDD specification.
Implement all scenarios as working code.

{spec[:5000]}

Generate complete Python code for {file_name}:""",
        "yaml": f"""Generate Python code from this YAML specification.
Match the structure exactly.

{spec[:5000]}

Generate complete Python code for {file_name}:""",
        "markdown": f"""Generate Python code from this Markdown specification.
It contains embedded Gherkin and YAML sections.

{spec[:5000]}

Generate complete Python code for {file_name}:""",
        "logicml": f"""Generate Python code from this LogicML specification.
'sig:' = EXACT function signature, 'does:' = docstring, 'attrs:' = class attributes.
Match signatures EXACTLY.

{spec[:5000]}

Generate complete Python code for {file_name}:""",
    }

    return prompts.get(fmt, prompts["yaml"])
