from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..generators import CSVGenerator, JSONGenerator, YAMLGenerator
from ..function_logic import FunctionLogicGenerator
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
    if fmt == "function.toon":
        gen = FunctionLogicGenerator()
        return gen.generate_toon(
            project,
            detail="full",
            no_repeat_name=True,
            no_repeat_details=True,
            include_does=True,
        )
    if fmt == "csv":
        gen = CSVGenerator()
        return gen.generate(project, detail="full")
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


def get_token_reproduction_prompt(spec: str, fmt: str, file_name: str, language: str = "python") -> str:
    format_hints = {
        "json": "Parse the JSON structure and implement all classes and functions with exact signatures.",
        "json_compact": "Parse the compact JSON and implement all elements with exact signatures.",
        "yaml": "Parse the YAML structure and implement all classes and functions with exact signatures.",
        "gherkin": """Parse Gherkin/BDD scenarios and implement them as working code:
- Each Feature maps to a class or module
- Each Scenario maps to a function
- Given/When/Then steps describe the logic flow
- Implement actual logic, not just stubs
Focus on the described behavior and implement it directly.""",
        "markdown": "Parse embedded Gherkin (behaviors) and YAML (structures). Implement all described classes and functions.",
        "logicml": """Parse LogicML and generate VALID code:
- 'sig:' lines describe function signatures (translate to the target language)
- 'type: re-export' means this module primarily re-exports symbols
- 'attrs:' = instance attributes to set in constructor
- 'bases:' = parent classes to inherit from
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
        "csv": """Parse the CSV table where each row describes a code element:
- Columns: path, type (class/method/function), name, signature, language, intent, category, domain, imports
- 'method' rows belong to the class in the preceding 'class' row
- Implement all elements with the exact signatures shown
Generate complete code with all classes, methods, and functions.""",
        "function.toon": """Parse the function-logic TOON format:
- 'modules[N]{path,lang,items}:' lists files
- 'function_details:' contains per-module function listings
- Each function has: line number, name, signature, description
- 'ClassName.method_name' = method of that class
- 'cc:N' after name = cyclomatic complexity
Implement all listed functions with matching signatures and described behavior.""",
    }

    # Language-specific guidance appended to prompt
    lang_hints = {
        "javascript": "Use ES6+ syntax (const/let, arrow functions, classes). Use module.exports or export.",
        "typescript": "Use TypeScript syntax with interfaces, type annotations, and export statements.",
        "go": "Use proper Go syntax: package declaration, func receivers for methods, error returns.",
        "rust": "Use proper Rust syntax: impl blocks for methods, pub fn, Result/Option types, ownership.",
        "java": "Use proper Java syntax: public class, access modifiers, typed parameters, semicolons.",
        "csharp": "Use proper C# syntax: namespaces, access modifiers, typed parameters, semicolons.",
        "sql": "Use standard SQL: CREATE TABLE/VIEW/FUNCTION, proper column types, constraints.",
    }

    max_spec = 8000
    spec_truncated = spec[:max_spec] if len(spec) > max_spec else spec

    language_norm = (language or "python").strip().lower()
    lang_label_map = {
        "python": "Python",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "go": "Go",
        "rust": "Rust",
        "java": "Java",
        "csharp": "C#",
        "sql": "SQL",
    }
    lang_label = lang_label_map.get(language_norm, language_norm)

    lang_hint = lang_hints.get(language_norm, '')
    lang_hint_line = f"\n{lang_hint}" if lang_hint else ''

    prompt = f"""Generate {lang_label} code from this {fmt.upper()} specification.
{format_hints.get(fmt, '')}{lang_hint_line}

{spec_truncated}

Requirements:
- Complete, working {lang_label} code for {file_name}
- Include imports and type hints
- Implement all functions with actual logic

```{language_norm}
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
