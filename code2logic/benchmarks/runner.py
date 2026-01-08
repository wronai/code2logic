"""
Unified Benchmark Runner for Code2Logic.

Provides a standardized API for running various benchmark types:
- Format comparison benchmarks
- File reproduction benchmarks
- Function-level reproduction benchmarks
- Project-level benchmarks

Usage:
    from code2logic.benchmarks import BenchmarkRunner, BenchmarkConfig

    runner = BenchmarkRunner()
    result = runner.run_format_benchmark('tests/samples/', formats=['yaml', 'toon'])
    result.save('output/benchmark.json')
"""

import sys
import time
from pathlib import Path
from typing import List, Optional

from ..analyzer import analyze_project
from ..llm_clients import BaseLLMClient, get_client
from ..metrics import ReproductionMetrics
from ..terminal import render
from ..utils import estimate_tokens
from .common import create_single_project, generate_spec_token, get_token_reproduction_prompt
from .results import BenchmarkConfig, BenchmarkResult, FileResult, FormatResult, FunctionResult


def _test_python_syntax(code: str) -> bool:
    """Test if Python code has valid syntax."""
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False


def _test_python_runs(code: str, timeout: int = 5) -> bool:
    """Test if Python code runs without errors."""
    import subprocess
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, timeout=timeout
            )
            return result.returncode == 0
    except Exception:
        return True  # Timeout might mean waiting for input


def _extract_code(response: str) -> str:
    """Extract code from LLM response."""
    if not response:
        return ""

    # Try to find code block
    for marker in ['```python', '```py', '```']:
        if marker in response:
            start = response.find(marker) + len(marker)
            if start < len(response) and response[start] == '\n':
                start += 1
            end = response.find('```', start)
            if end > start:
                return response[start:end].strip()
            return response[start:].strip()

    return response.strip()


class BenchmarkRunner:
    """
    Unified benchmark runner for code2logic.

    Consolidates benchmarking logic from multiple example scripts
    into a standardized, reusable API.
    """

    def __init__(
        self,
        client: Optional[BaseLLMClient] = None,
        config: Optional[BenchmarkConfig] = None
    ):
        """
        Initialize benchmark runner.

        Args:
            client: LLM client (auto-detected if None)
            config: Benchmark configuration
        """
        self.client = client
        self.config = config or BenchmarkConfig()
        self._metrics = ReproductionMetrics()

    def _should_use_llm(self) -> bool:
        """Return whether this runner should call an LLM."""
        return bool(getattr(self.config, "use_llm", True))

    def _get_client(self) -> BaseLLMClient:
        """Get or create LLM client."""
        if not self._should_use_llm():
            raise RuntimeError("LLM usage disabled (BenchmarkConfig.use_llm=False)")
        if self.client is None:
            self.client = get_client()
        return self.client

    def _template_generate_code(self, spec: str, fmt: str, file_name: str) -> str:
        """Generate minimal Python code without an LLM (fallback mode)."""
        import re

        # Try to infer class/function names from spec
        classes: List[str] = []
        functions: List[str] = []

        # Common patterns
        classes.extend(re.findall(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)", spec))
        classes.extend(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$", spec, re.MULTILINE))
        functions.extend(re.findall(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", spec))
        functions.extend(re.findall(r"\bFunction:\s*([A-Za-z_][A-Za-z0-9_]*)", spec))
        functions.extend(re.findall(r"\bScenario:\s*([A-Za-z_][A-Za-z0-9_]*)", spec))

        # Deduplicate while preserving order
        def uniq(items: List[str]) -> List[str]:
            seen = set()
            out: List[str] = []
            for it in items:
                if it and it not in seen:
                    seen.add(it)
                    out.append(it)
            return out

        classes = [c for c in uniq(classes) if c.isidentifier()][:5]
        functions = [f for f in uniq(functions) if f.isidentifier() and f not in classes][:10]

        code = """from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, List, Dict

"""

        if not classes and not functions:
            # Always emit something valid
            safe_name = Path(file_name).stem.replace('-', '_').replace('.', '_')
            safe_name = safe_name if safe_name.isidentifier() else "GeneratedModule"
            classes = ["GeneratedClass"]
            functions = ["generated_function"]

        for cls in classes:
            code += f"""@dataclass
class {cls}:
    \"\"\"Generated placeholder for {file_name} ({fmt}).\"\"\"
    value: Any = None

"""

        for fn in functions:
            code += f"""def {fn}(*args: Any, **kwargs: Any) -> Any:
    \"\"\"Generated placeholder for {file_name} ({fmt}).\"\"\"
    return None

"""

        return code

    def run_format_benchmark(
        self,
        folder: str,
        formats: List[str] = None,
        limit: Optional[int] = None,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """
        Run format comparison benchmark.

        Compares reproduction quality across different spec formats.

        Args:
            folder: Folder containing source files
            formats: Formats to test (default from config)
            limit: Max files to process
            verbose: Print progress

        Returns:
            BenchmarkResult with format comparison data
        """
        formats = formats or self.config.formats

        client: Optional[BaseLLMClient]
        if self._should_use_llm():
            try:
                client = self._get_client()
            except Exception as e:
                client = None
                if verbose:
                    render.warning(f"LLM not available ({str(e)[:80]}). Falling back to template mode.")
        else:
            client = None

        result = BenchmarkResult(
            benchmark_type='format',
            source_path=folder,
            provider=getattr(client, 'provider', 'none') if client else 'none',
            model=getattr(client, 'model', 'none') if client else 'none',
        )

        # Analyze project
        path = Path(folder)
        py_files = list(path.glob('*.py'))
        if limit:
            py_files = py_files[:limit]

        result.total_files = len(py_files)

        if verbose:
            render.heading(2, "Format Benchmark")
            render.codeblock("yaml", f"folder: {folder}\nfiles: {len(py_files)}\nformats: [{', '.join(formats)}]")

        project = analyze_project(str(path), use_treesitter=False)

        start_time = time.time()

        # Process each file with each format
        for py_file in py_files:
            original = py_file.read_text()

            # Find module info
            module_info = None
            for m in project.modules:
                if Path(m.path).name == py_file.name:
                    module_info = m
                    break

            if not module_info:
                continue

            single_project = create_single_project(module_info, py_file)

            file_result = FileResult(
                file_path=str(py_file),
                language='python',
                original_size=len(original),
            )

            for fmt in formats:
                fmt_result = self._test_format(
                    single_project, original, fmt, py_file.name, client, verbose
                )
                file_result.format_results[fmt] = fmt_result

            # Set best result as file score
            if file_result.format_results:
                best = max(file_result.format_results.values(), key=lambda r: r.score)
                file_result.score = best.score
                file_result.syntax_ok = best.syntax_ok
                file_result.runs_ok = best.runs_ok

            result.file_results.append(file_result)

        result.total_time = time.time() - start_time

        # Calculate format aggregates
        for fmt in formats:
            scores = [
                fr.format_results[fmt].score
                for fr in result.file_results
                if fmt in fr.format_results and fr.format_results[fmt].score > 0
            ]
            if scores:
                result.format_scores[fmt] = sum(scores) / len(scores)

        result.calculate_aggregates()

        return result

    def _test_format(
        self,
        project,
        original: str,
        fmt: str,
        file_name: str,
        client: Optional[BaseLLMClient],
        verbose: bool = False,
    ) -> FormatResult:
        """Test a single format."""
        result = FormatResult(format_name=fmt)

        try:
            # Generate spec
            spec = generate_spec_token(project, fmt)
            result.spec_size = len(spec)
            result.spec_tokens = estimate_tokens(spec)

            # Generate prompt
            prompt = get_token_reproduction_prompt(spec, fmt, file_name)

            # Reproduce
            start = time.time()
            if client is None:
                generated = self._template_generate_code(spec, fmt, file_name)
                result.gen_time = 0.0
            else:
                response = client.generate(prompt, max_tokens=self.config.max_tokens)
                result.gen_time = time.time() - start
                generated = _extract_code(response)
            result.generated_size = len(generated)

            # Test quality
            result.syntax_ok = _test_python_syntax(generated)
            if result.syntax_ok:
                result.runs_ok = _test_python_runs(generated)

            # Calculate metrics
            if original and generated:
                analysis = self._metrics.analyze(original, generated, spec, format_name=fmt)
                result.score = analysis.overall_score
                result.similarity = analysis.overall_score

            # Efficiency
            if result.spec_size and len(original) > 0:
                result.compression_ratio = result.spec_size / len(original)
            if result.score > 0 and result.spec_tokens > 0:
                result.token_efficiency = result.score / result.spec_tokens * 100

            if verbose:
                if result.score > 50:
                    render.task(f"{fmt}: {result.score:.1f}%", "done")
                else:
                    render.task(f"{fmt}: {result.score:.1f}%", "pending")

        except Exception as e:
            result.error = str(e)[:100]
            if verbose:
                render.task(f"{fmt}: {str(e)[:50]}", "failed")

        return result

    def run_file_benchmark(
        self,
        file_path: str,
        formats: List[str] = None,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """
        Run benchmark on a single file.

        Args:
            file_path: Path to source file
            formats: Formats to test
            verbose: Print progress

        Returns:
            BenchmarkResult
        """
        formats = formats or self.config.formats

        client: Optional[BaseLLMClient]
        if self._should_use_llm():
            try:
                client = self._get_client()
            except Exception as e:
                client = None
                if verbose:
                    render.warning(f"LLM not available ({str(e)[:80]}). Falling back to template mode.")
        else:
            client = None

        result = BenchmarkResult(
            benchmark_type='file',
            source_path=file_path,
            total_files=1,
            provider=getattr(client, 'provider', 'none') if client else 'none',
            model=getattr(client, 'model', 'none') if client else 'none',
        )

        path = Path(file_path)
        original = path.read_text()

        # Analyze
        project = analyze_project(str(path.parent), use_treesitter=False)

        module_info = None
        for m in project.modules:
            if Path(m.path).name == path.name:
                module_info = m
                break

        if not module_info:
            result.file_results.append(FileResult(
                file_path=file_path,
                language='python',
                error="Module not found in analysis"
            ))
            return result

        single_project = create_single_project(module_info, path)

        file_result = FileResult(
            file_path=file_path,
            language='python',
            original_size=len(original),
        )

        start_time = time.time()

        for fmt in formats:
            if verbose:
                render.task(f"Testing {fmt}", "running")

            fmt_result = self._test_format(
                single_project, original, fmt, path.name, client, verbose=False
            )
            file_result.format_results[fmt] = fmt_result

            if verbose:
                if fmt_result.error:
                    render.task(f"{fmt}: {fmt_result.error[:30]}", "failed")
                elif fmt_result.score > 50:
                    render.task(f"{fmt}: {fmt_result.score:.1f}%", "done")
                else:
                    render.task(f"{fmt}: {fmt_result.score:.1f}%", "pending")

        result.total_time = time.time() - start_time

        # Best format
        if file_result.format_results:
            best = max(file_result.format_results.values(), key=lambda r: r.score)
            file_result.score = best.score
            file_result.syntax_ok = best.syntax_ok
            result.best_format = best.format_name
            result.best_score = best.score

        result.file_results.append(file_result)
        result.calculate_aggregates()

        return result

    def run_function_benchmark(
        self,
        file_path: str,
        function_names: List[str] = None,
        limit: Optional[int] = None,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """
        Run function-level reproduction benchmark.

        Args:
            file_path: Path to source file
            function_names: Specific functions to test (None = all)
            limit: Max functions to test
            verbose: Print progress

        Returns:
            BenchmarkResult with function results
        """
        from ..parsers import UniversalParser

        client: Optional[BaseLLMClient]
        if self._should_use_llm():
            try:
                client = self._get_client()
            except Exception as e:
                client = None
                if verbose:
                    render.warning(f"LLM not available ({str(e)[:80]}). Falling back to template mode.")
        else:
            client = None

        result = BenchmarkResult(
            benchmark_type='function',
            source_path=file_path,
            provider=getattr(client, 'provider', 'none') if client else 'none',
            model=getattr(client, 'model', 'none') if client else 'none',
        )

        path = Path(file_path)
        content = path.read_text()

        # Detect language
        ext_to_lang = {'.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.go': 'go'}
        language = ext_to_lang.get(path.suffix, 'python')

        # Parse file
        parser = UniversalParser()
        module = parser.parse(str(path), content, language)

        functions = module.functions
        if function_names:
            functions = [f for f in functions if f.name in function_names]
        if limit:
            functions = functions[:limit]

        result.total_functions = len(functions)

        if verbose:
            print(f"Function Benchmark: {file_path}")
            print(f"Functions: {len(functions)}")

        start_time = time.time()

        for func in functions:
            func_result = self._test_function(func, content, language, path, client, verbose)
            result.function_results.append(func_result)

        result.total_time = time.time() - start_time
        result.calculate_aggregates()

        return result

    def _test_function(
        self,
        func,
        content: str,
        language: str,
        file_path: Path,
        client: Optional[BaseLLMClient],
        verbose: bool = False,
    ) -> FunctionResult:
        """Test reproduction of a single function."""
        result = FunctionResult(
            file_path=str(file_path),
            function_name=func.name,
            language=language,
        )

        try:
            # Extract original code
            lines = content.split('\n')
            start = func.start_line - 1
            end = func.end_line if func.end_line else start + func.lines

            # Include decorators for Python
            if language == 'python' and start > 0:
                i = start - 1
                while i >= 0 and lines[i].strip().startswith('@'):
                    start = i
                    i -= 1

            result.original_code = '\n'.join(lines[start:end])

            # Create spec
            spec = f"""Function: {func.name}
Language: {language}
Signature: {func.name}({', '.join(func.params)}) -> {func.return_type or 'None'}
Description: {func.intent or func.docstring or 'No description'}
Is Async: {func.is_async}
Decorators: {', '.join(func.decorators) if func.decorators else 'None'}
Lines: {func.lines}
"""

            prompt = f"""Generate ONLY the function code based on this specification:

{spec}

Requirements:
- Generate complete, working {language} function
- Match the signature exactly
- Output ONLY the function code

```{language}
"""

            if client is None:
                # Offline fallback: emit a skeleton function with matching name/params.
                params = ", ".join(func.params) if getattr(func, "params", None) else "*args, **kwargs"
                rt = func.return_type or "Any"
                async_kw = "async " if getattr(func, "is_async", False) else ""
                result.reproduced_code = f"{async_kw}def {func.name}({params}) -> {rt}:\n    return None\n"
                result.gen_time = 0.0
            else:
                start_time = time.time()
                response = client.generate(prompt, max_tokens=2000)
                result.gen_time = time.time() - start_time
                result.reproduced_code = _extract_code(response)

            # Test syntax
            if language == 'python':
                result.syntax_ok = _test_python_syntax(result.reproduced_code)
            else:
                result.syntax_ok = len(result.reproduced_code) > 10

            # Calculate similarity
            from difflib import SequenceMatcher
            orig_norm = ' '.join(result.original_code.split())
            repr_norm = ' '.join(result.reproduced_code.split())
            result.similarity = SequenceMatcher(None, orig_norm, repr_norm).ratio() * 100

            if verbose:
                syntax = "S✓" if result.syntax_ok else "S✗"
                print(f"  {func.name}: {result.similarity:.1f}% {syntax}")

        except Exception as e:
            result.error = str(e)[:100]
            if verbose:
                print(f"  {func.name}: ERROR - {e}")

        return result

    def run_project_benchmark(
        self,
        project_path: str,
        formats: List[str] = None,
        limit: Optional[int] = None,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """
        Run benchmark on entire project.

        Args:
            project_path: Path to project
            formats: Formats to test
            limit: Max files to process
            verbose: Print progress

        Returns:
            BenchmarkResult with project-level data
        """
        formats = formats or self.config.formats

        client: Optional[BaseLLMClient]
        if self._should_use_llm():
            try:
                client = self._get_client()
            except Exception as e:
                client = None
                if verbose:
                    render.warning(f"LLM not available ({str(e)[:80]}). Falling back to template mode.")
        else:
            client = None

        result = BenchmarkResult(
            benchmark_type='project',
            source_path=project_path,
            provider=getattr(client, 'provider', 'none') if client else 'none',
            model=getattr(client, 'model', 'none') if client else 'none',
        )

        # Analyze
        project = analyze_project(project_path, use_treesitter=False)
        modules = project.modules[:limit] if limit else project.modules
        result.total_files = len(modules)

        if verbose:
            print(f"Project Benchmark: {project_path}")
            print(f"Files: {len(modules)}, Formats: {', '.join(formats)}")

        start_time = time.time()

        for fmt in formats:
            if verbose:
                print(f"\n--- Format: {fmt.upper()} ---")

            for i, module in enumerate(modules):
                file_result = self._reproduce_module(
                    module, fmt, project_path, client, verbose
                )

                # Add format to result
                if file_result.format_results:
                    fmt_result = list(file_result.format_results.values())[0]
                else:
                    fmt_result = FormatResult(format_name=fmt, score=file_result.score)
                    file_result.format_results[fmt] = fmt_result

                # Find existing file result or create new
                existing = None
                for fr in result.file_results:
                    if fr.file_path == file_result.file_path:
                        existing = fr
                        break

                if existing:
                    existing.format_results[fmt] = fmt_result
                else:
                    result.file_results.append(file_result)

                if verbose:
                    status = "✓" if file_result.score > 50 else "○"
                    print(f"  [{i+1}/{len(modules)}] {Path(file_result.file_path).name}: {file_result.score:.1f}% {status}")

        result.total_time = time.time() - start_time

        # Calculate format aggregates
        for fmt in formats:
            scores = []
            for fr in result.file_results:
                if fmt in fr.format_results:
                    score = fr.format_results[fmt].score
                    if score > 0:
                        scores.append(score)
            if scores:
                result.format_scores[fmt] = sum(scores) / len(scores)

        result.calculate_aggregates()

        return result

    def _reproduce_module(
        self,
        module_info,
        fmt: str,
        project_root: str,
        client: Optional[BaseLLMClient],
        verbose: bool = False,
    ) -> FileResult:
        """Reproduce a single module."""
        from datetime import datetime

        from ..models import ProjectInfo

        # Build path
        rel_path = module_info.path
        if not Path(rel_path).is_absolute():
            abs_path = Path(project_root) / rel_path
        else:
            abs_path = Path(rel_path)

        file_result = FileResult(
            file_path=str(abs_path),
            language=module_info.language,
        )

        try:
            original = abs_path.read_text()
            file_result.original_size = len(original)
        except Exception as e:
            file_result.error = f"Cannot read file: {e}"
            return file_result

        try:
            # Create single-file project
            single_project = ProjectInfo(
                name=abs_path.name,
                root_path=str(abs_path.parent),
                languages={module_info.language: 1},
                modules=[module_info],
                dependency_graph={},
                dependency_metrics={},
                entrypoints=[],
                similar_functions={},
                total_files=1,
                total_lines=module_info.lines_total,
                generated_at=datetime.now().isoformat(),
            )

            # Test format
            fmt_result = self._test_format(
                single_project, original, fmt, abs_path.name, client, verbose=False
            )

            file_result.format_results[fmt] = fmt_result
            file_result.score = fmt_result.score
            file_result.syntax_ok = fmt_result.syntax_ok
            file_result.runs_ok = fmt_result.runs_ok
            file_result.gen_time = fmt_result.gen_time

        except Exception as e:
            file_result.error = str(e)[:100]

        return file_result


def run_benchmark(
    source: str,
    benchmark_type: str = 'format',
    formats: List[str] = None,
    limit: Optional[int] = None,
    output: Optional[str] = None,
    verbose: bool = False,
) -> BenchmarkResult:
    """
    Convenience function to run benchmarks.

    Args:
        source: Source file or folder
        benchmark_type: 'format', 'file', 'function', or 'project'
        formats: Formats to test
        limit: Max items to process
        output: Output file path
        verbose: Print progress

    Returns:
        BenchmarkResult

    Example:
        result = run_benchmark('tests/samples/', 'format', formats=['yaml', 'toon'])
        result.save('output/benchmark.json')
    """
    runner = BenchmarkRunner()

    if benchmark_type == 'format':
        result = runner.run_format_benchmark(source, formats, limit, verbose)
    elif benchmark_type == 'file':
        result = runner.run_file_benchmark(source, formats, verbose)
    elif benchmark_type == 'function':
        result = runner.run_function_benchmark(source, limit=limit, verbose=verbose)
    elif benchmark_type == 'project':
        result = runner.run_project_benchmark(source, formats, limit, verbose)
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")

    if output:
        result.save(output)

    return result
