from __future__ import annotations

from typing import List, Tuple

from .models import ProjectInfo, FunctionInfo
from .shared_utils import remove_self_from_params, truncate_docstring


class FunctionLogicGenerator:
    FILE_EXTENSION: str = ".functions.logicml"

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def generate(self, project: ProjectInfo, detail: str = 'full') -> str:
        lines: List[str] = []

        for module in project.modules:
            items: List[Tuple[str, str, FunctionInfo]] = []

            for f in module.functions:
                items.append(('function', f.name, f))

            for cls in module.classes:
                for method in cls.methods:
                    items.append(('method', f"{cls.name}.{method.name}", method))

            if not items:
                continue

            lines.append(f"# {module.path} | {module.language} | {len(items)} functions")
            lines.append("functions:")

            for kind, qname, func in items:
                lines.extend(self._format_function(kind, qname, func, detail, indent=2))

            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _format_function(
        self,
        kind: str,
        qualified_name: str,
        func: FunctionInfo,
        detail: str,
        indent: int,
    ) -> List[str]:
        prefix = ' ' * indent
        sub = ' ' * (indent + 2)

        lines: List[str] = [f"{prefix}{qualified_name}:"]

        lines.append(f"{sub}kind: {kind}")

        clean_params = remove_self_from_params((func.params or [])[:10])
        params_str = ', '.join(clean_params)
        ret = func.return_type or 'None'
        sig = f"({params_str}) -> {ret}" if params_str else f"() -> {ret}"
        if getattr(func, 'is_async', False):
            sig = f"async {sig}"
        lines.append(f"{sub}sig: {sig}")

        start_line = getattr(func, 'start_line', 0) or 0
        end_line = getattr(func, 'end_line', 0) or 0
        if start_line and end_line:
            lines.append(f"{sub}loc: {start_line}-{end_line}")

        if detail in ('standard', 'full'):
            does_src = func.docstring or func.intent or ''
            does = truncate_docstring(does_src, max_length=120) if does_src else ''
            does = does.replace('\n', ' ').replace('"', "'").strip()
            if does:
                lines.append(f"{sub}does: \"{does}\"")

        if detail == 'full':
            if getattr(func, 'lines', 0):
                lines.append(f"{sub}lines: {func.lines}")
            if getattr(func, 'complexity', 0):
                lines.append(f"{sub}complexity: {func.complexity}")

            decorators = getattr(func, 'decorators', []) or []
            if decorators:
                lines.append(f"{sub}decorators: [{', '.join(decorators[:10])}]")

            if getattr(func, 'calls', None):
                calls = (func.calls or [])
                if calls:
                    lines.append(f"{sub}calls: [{', '.join(calls[:40])}]")

            if getattr(func, 'raises', None):
                raises = (func.raises or [])
                if raises:
                    lines.append(f"{sub}raises: [{', '.join(raises[:20])}]")

        return lines
