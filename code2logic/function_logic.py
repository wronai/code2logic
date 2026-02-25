from __future__ import annotations

from typing import List, Tuple

from .models import FunctionInfo, ProjectInfo
from .shared_utils import remove_self_from_params, truncate_docstring
from .toon_format import TOONGenerator


class FunctionLogicGenerator:
    FILE_EXTENSION: str = ".functions.logicml"

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def generate(self, project: ProjectInfo, detail: str = 'full') -> str:
        if detail == 'detailed':
            detail = 'full'
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
                lines.extend(self._format_function(kind, qname, func, detail, indent=2, module_language=module.language))

            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def generate_json(self, project: ProjectInfo, detail: str = 'full') -> str:
        if detail == 'detailed':
            detail = 'full'
        import json

        data = self._build_data(project, detail)
        return json.dumps(data, ensure_ascii=False, indent=2)

    def generate_yaml(self, project: ProjectInfo, detail: str = 'full') -> str:
        if detail == 'detailed':
            detail = 'full'
        data = self._build_data(project, detail)
        try:
            import yaml
        except ImportError:
            return self.generate(project, detail)
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    def generate_toon(
        self,
        project: ProjectInfo,
        detail: str = 'full',
        no_repeat_name: bool = False,
        no_repeat_details: bool = False,
    ) -> str:
        if detail == 'detailed':
            detail = 'full'
        toon = TOONGenerator()
        delim = toon.delimiter
        dm = toon.delim_marker

        lines: List[str] = []
        lines.append(f"project: {toon._quote(project.name)}")
        if getattr(project, 'generated_at', None):
            lines.append(f"generated: {toon._quote(project.generated_at)}")

        modules = list(project.modules or [])
        lines.append(f"modules[{len(modules)}]{{path{dm}lang{dm}items}}:")
        prev_dir: str | None = None
        for m in modules:
            items = self._module_items(m)
            if no_repeat_name:
                compressed_path, prev_dir = toon._compress_module_path(m.path, prev_dir)
                path_out = compressed_path
            else:
                path_out = m.path
            lines.append(f"  {toon._quote(path_out)}{delim}{toon._short_lang(m.language)}{delim}{len(items)}")

        lines.append("")
        lines.append("function_details:")

        prev_dir = None
        for m in modules:
            items = self._module_items(m)
            if not items:
                continue

            if no_repeat_details:
                compressed_path, prev_dir = toon._compress_module_path(m.path, prev_dir)
                details_key = compressed_path
            else:
                details_key = m.path
            lines.append(f"  {toon._quote(details_key)}:")

            header = f"line{dm}name{dm}kind{dm}sig{dm}async{dm}cc"
            if detail in ('standard', 'full'):
                header += f"{dm}does"
            if detail == 'full':
                header += f"{dm}decorators{dm}calls{dm}raises"

            lines.append(f"    functions[{len(items)}]{{{header}}}:")

            for kind, qname, func in items:
                sig = self._build_sig(func, include_async_prefix=False, language=m.language)
                is_async = 'true' if getattr(func, 'is_async', False) else 'false'
                start_line = str(getattr(func, 'start_line', 0) or 0)
                row = [
                    start_line,
                    toon._quote(qname),
                    toon._quote(kind),
                    toon._quote(sig),
                    is_async,
                    str(getattr(func, 'complexity', 1) or 1),
                ]

                if detail in ('standard', 'full'):
                    does = self._build_does(func)
                    row.append(toon._quote(does))

                if detail == 'full':
                    decorators = '|'.join((getattr(func, 'decorators', []) or [])[:10]) or '-'
                    calls = '|'.join((getattr(func, 'calls', []) or [])[:40]) or '-'
                    raises = '|'.join((getattr(func, 'raises', []) or [])[:20]) or '-'
                    row.append(toon._quote(decorators))
                    row.append(toon._quote(calls))
                    row.append(toon._quote(raises))

                lines.append(f"      {delim.join(row)}")

        return "\n".join(lines).rstrip() + "\n"

    def _build_data(self, project: ProjectInfo, detail: str) -> dict:
        modules_data = []
        for m in project.modules or []:
            items = self._module_items(m)
            if not items:
                continue
            modules_data.append({
                'path': m.path,
                'language': m.language,
                'functions': [self._item_to_dict(kind, qname, func, detail, module_language=m.language) for kind, qname, func in items]
            })

        data = {
            'project': project.name,
            'generated_at': getattr(project, 'generated_at', ''),
            'modules': modules_data,
        }
        return data

    def _module_items(self, module) -> List[Tuple[str, str, FunctionInfo]]:
        items: List[Tuple[str, str, FunctionInfo]] = []

        for f in getattr(module, 'functions', []) or []:
            items.append(('function', f.name, f))

        for cls in getattr(module, 'classes', []) or []:
            for method in getattr(cls, 'methods', []) or []:
                items.append(('method', f"{cls.name}.{method.name}", method))

        return items

    def _build_sig(self, func: FunctionInfo, include_async_prefix: bool = True, language: str = '') -> str:
        clean_params = remove_self_from_params((func.params or [])[:10])
        params_str = ', '.join(clean_params)
        ret = getattr(func, 'return_type', None)
        if isinstance(ret, str):
            ret = ret.strip()

        if language in ('javascript', 'typescript') and ret == 'None':
            ret = None

        sig = f"({params_str})" if params_str else "()"
        if ret:
            sig = f"{sig} -> {ret}"
        if include_async_prefix and getattr(func, 'is_async', False):
            sig = f"async {sig}"
        return sig

    def _build_loc(self, func: FunctionInfo) -> str:
        start_line = getattr(func, 'start_line', 0) or 0
        end_line = getattr(func, 'end_line', 0) or 0
        if start_line and end_line:
            return f"{start_line}-{end_line}"
        return "-"

    def _build_does(self, func: FunctionInfo) -> str:
        does_src = func.docstring or func.intent or ''
        does = truncate_docstring(does_src, max_length=120) if does_src else ''
        does = does.replace('\n', ' ').replace('"', "'").strip()
        return does or '-'

    def _item_to_dict(self, kind: str, qualified_name: str, func: FunctionInfo, detail: str, module_language: str = '') -> dict:
        data = {
            'name': qualified_name,
            'kind': kind,
            'sig': self._build_sig(func, include_async_prefix=True, language=module_language),
        }

        start_line = getattr(func, 'start_line', 0) or 0
        end_line = getattr(func, 'end_line', 0) or 0
        if start_line:
            data['start_line'] = start_line
        if end_line:
            data['end_line'] = end_line

        if detail in ('standard', 'full'):
            does = self._build_does(func)
            if does and does != '-':
                data['does'] = does

        if detail == 'full':
            if getattr(func, 'lines', 0):
                data['lines'] = func.lines
            if getattr(func, 'complexity', 0):
                data['complexity'] = func.complexity

            decorators = getattr(func, 'decorators', []) or []
            if decorators:
                data['decorators'] = decorators[:10]

            calls = getattr(func, 'calls', []) or []
            if calls:
                data['calls'] = calls[:40]

            raises = getattr(func, 'raises', []) or []
            if raises:
                data['raises'] = raises[:20]

        return data

    def _format_function(
        self,
        kind: str,
        qualified_name: str,
        func: FunctionInfo,
        detail: str,
        indent: int,
        module_language: str = '',
    ) -> List[str]:
        prefix = ' ' * indent
        sub = ' ' * (indent + 2)

        lines: List[str] = [f"{prefix}{qualified_name}:"]

        lines.append(f"{sub}kind: {kind}")

        sig = self._build_sig(func, include_async_prefix=True, language=module_language)
        lines.append(f"{sub}sig: {sig}")

        loc = self._build_loc(func)
        if loc != '-':
            lines.append(f"{sub}loc: {loc}")

        if detail in ('standard', 'full'):
            does = self._build_does(func)
            if does != '-':
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
