#!/usr/bin/env python3
"""
Example: Generate code from CSV analysis using local Ollama.

This script demonstrates how to:
1. Analyze a project with code2logic
2. Send the analysis to Ollama for code generation
3. Generate equivalent code in another language

Requirements:
    pip install httpx
    ollama serve  # Start Ollama
    ollama pull qwen2.5-coder:7b
"""

import sys
import json
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic import analyze_project, CSVGenerator

OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen2.5-coder:7b"


def check_ollama():
    """Check if Ollama is running."""
    try:
        response = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def generate_with_ollama(prompt: str, system: str = None) -> str:
    """Generate text using Ollama API."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 2000,
        }
    }

    if system:
        payload["system"] = system

    response = httpx.post(
        f"{OLLAMA_HOST}/api/generate",
        json=payload,
        timeout=120.0
    )
    response.raise_for_status()
    return response.json().get("response", "")


def generate_code_from_csv(csv_content: str, target_lang: str) -> dict:
    """Generate code from CSV analysis."""
    import csv
    from io import StringIO

    # Parse CSV
    reader = csv.DictReader(StringIO(csv_content))
    rows = list(reader)

    # Group by module
    modules = {}
    for row in rows:
        path = row.get('path', 'unknown')
        if path not in modules:
            modules[path] = []
        modules[path].append(row)

    results = {}

    for path, elements in list(modules.items())[:3]:  # Limit to 3 modules
        # Build context
        context = f"Module: {path}\n\nElements:\n"
        for e in elements[:10]:
            context += f"- {e.get('type', 'unknown')}: {e.get('name', '')} {e.get('signature', '')}\n"
            if e.get('intent'):
                context += f"  Intent: {e['intent']}\n"

        system = """You are an expert software engineer.
Generate clean, idiomatic code with proper type annotations and documentation.
Output only code without explanations."""

        prompt = f"""Generate {target_lang} code from this specification:

{context}

Requirements:
1. Full type annotations
2. Docstrings/comments
3. Error handling
4. Same public API"""

        print(f"\n{'=' * 60}")
        print(f"Generating {target_lang} for: {path}")
        print('=' * 60)

        code = generate_with_ollama(prompt, system)
        results[path] = code

        # Print preview
        preview = code[:500] + "..." if len(code) > 500 else code
        print(preview)

    return results


def main():
    """Main example."""
    if len(sys.argv) < 2:
        print("Usage: python generate_code.py /path/to/project [target_lang]")
        print("Example: python generate_code.py . typescript")
        sys.exit(1)

    project_path = sys.argv[1]
    target_lang = sys.argv[2] if len(sys.argv) > 2 else "typescript"

    # Check Ollama
    if not check_ollama():
        print(f"Error: Ollama not running at {OLLAMA_HOST}")
        print("Start with: ollama serve")
        sys.exit(1)

    print(f"Analyzing project: {project_path}")

    # Analyze project
    project = analyze_project(project_path)
    print(f"Found {project.total_files} files, {project.total_lines} lines")

    # Generate CSV
    csv_gen = CSVGenerator()
    csv_content = csv_gen.generate(project, detail='standard')

    # Generate code
    results = generate_code_from_csv(csv_content, target_lang)

    # Save results
    output_file = f"generated_{target_lang}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()