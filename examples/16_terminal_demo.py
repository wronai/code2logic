#!/usr/bin/env python3
"""
Terminal Rendering Demo - Colorized Markdown Output in Shell.

Demonstrates the ShellRenderer for colorized terminal output.

Usage:
    python examples/16_terminal_demo.py
    python examples/16_terminal_demo.py --no-color
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic.terminal import render, ShellRenderer, get_renderer, set_renderer


def demo_headings():
    """Demo heading styles."""
    render.heading(1, "Terminal Rendering Demo")
    render.heading(2, "Heading Level 2")
    render.heading(3, "Heading Level 3")


def demo_codeblocks():
    """Demo syntax-highlighted code blocks."""
    render.heading(2, "Code Blocks with Syntax Highlighting")
    
    # YAML
    render.codeblock("yaml", """# Configuration file
name: code2logic
version: 1.0.0
settings:
  verbose: true
  max_tokens: 4000
  formats:
    - yaml
    - json
    - gherkin""")
    
    # JSON
    render.codeblock("json", """{
  "name": "benchmark",
  "score": 95.5,
  "passed": true,
  "items": null
}""")
    
    # Python
    render.codeblock("python", """from dataclasses import dataclass
from typing import List, Optional

@dataclass
class BenchmarkResult:
    \"\"\"Benchmark result container.\"\"\"
    score: float
    passed: bool = True
    
    def is_success(self) -> bool:
        # Check if score is above threshold
        return self.score > 50.0""")
    
    # Gherkin
    render.codeblock("gherkin", """Feature: Code Reproduction
  @benchmark
  Scenario: Reproduce Python file
    Given a Python source file "models.py"
    When I generate a YAML specification
    And I reproduce code from the spec
    Then the similarity should be above 80%""")
    
    # Bash
    render.codeblock("bash", """# Run benchmarks
cd ~/projects/code2logic
python -m pytest tests/ -v
code2logic analyze --format yaml""")


def demo_status_messages():
    """Demo status messages."""
    render.heading(2, "Status Messages")
    
    render.codeblock(
        "log",
        "\n".join(
            [
                "✅ All tests passed!",
                "❌ Failed to connect to LLM",
                "⚠️ Token limit exceeded, using fallback",
                "ℹ️ Processing 15 files...",
            ]
        ),
    )


def demo_progress():
    """Demo progress bars."""
    render.heading(2, "Progress Bars")
    
    render.codeblock(
        "log",
        "\n".join(
            [
                "[ ░░░░░░░░░░░░░░░░░░░░ ] 0% Starting...",
                "[ ██████░░░░░░░░░░░░░░ ] 30% Processing files",
                "[ ██████████████░░░░░░ ] 70% Running tests",
                "[ ████████████████████ ] 100% Complete!",
            ]
        ),
    )


def demo_tasks():
    """Demo task status."""
    render.heading(2, "Task Status")
    
    render.codeblock(
        "log",
        "\n".join(
            [
                "✅ Create output folders (0.2s)",
                "✅ Generate specifications (1.5s)",
                "🔄 Reproduce code",
                "⏳ Run tests",
                "⏳ Validate output",
                "❌ Failed task example (0.1s)",
            ]
        ),
    )


def demo_key_value():
    """Demo key-value pairs."""
    render.heading(2, "Key-Value Pairs")
    
    render.kv("Files", 15)
    render.kv("Formats", "yaml, json, gherkin")
    render.kv("Score", 95.5)
    render.kv("Passed", True)
    render.kv("Provider", "openrouter")


def demo_tables():
    """Demo table output."""
    render.heading(2, "Tables")
    
    headers = ["Format", "Score", "Syntax", "Status"]
    rows = [
        ["yaml", "95.5%", "✓", "passed"],
        ["json", "87.2%", "✓", "passed"],
        ["gherkin", "72.1%", "✓", "passed"],
        ["toon", "n/a", "-", "demo"],
    ]
    render.table(headers, rows, widths=[12, 10, 10, 10])


def demo_markdown():
    """Demo full markdown rendering."""
    render.heading(2, "Full Markdown Rendering")
    
    markdown_text = """## Benchmark Results

The following formats were tested:

```yaml
formats:
  - yaml: 95.5%
  - json: 87.2%
  - gherkin: 72.1%
```

### Summary

All tests **passed** successfully.

```log
✅ Benchmark complete
📊 Progress: 10/10 done
→ Saving results...
```
"""
    render.markdown(markdown_text)


def demo_log_highlighting():
    """Demo log message highlighting."""
    render.heading(2, "Log Message Highlighting")
    
    render.codeblock("log", """✅ All tests passed
❌ Error: Connection refused
⚠️ Warning: Rate limit approaching
ℹ️ Info: Using fallback provider
🚀 Starting benchmark...
📦 Installing dependencies...
📊 Progress: 5/10 done
→ Processing file: models.py
💬 Generating specification...""")


def main():
    parser = argparse.ArgumentParser(description='Terminal Rendering Demo')
    parser.add_argument('--no-color', action='store_true', help='Disable colors')
    parser.add_argument('--folder', '-f', default=None, help='Ignored (kept for CLI compatibility)')
    args = parser.parse_args()
    
    if args.no_color:
        set_renderer(ShellRenderer(use_colors=False))
    
    demo_headings()
    render.separator()
    
    demo_codeblocks()
    render.separator()
    
    demo_status_messages()
    render.separator()
    
    demo_progress()
    render.separator()
    
    demo_tasks()
    render.separator()
    
    demo_key_value()
    render.separator()
    
    demo_tables()
    render.separator()
    
    demo_log_highlighting()
    render.separator()
    
    demo_markdown()
    
    render.heading(2, "Demo Complete")
    render.success("Terminal rendering demo finished!")


if __name__ == '__main__':
    main()
