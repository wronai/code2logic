# CLI Reference

> Command-line interface for Code2Logic

[← README](../README.md) | [← Configuration](configuration.md) | [Python API →](python-api.md)

## Basic Usage

```bash
code2logic /path/to/project [options]
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--format` | `-f` | Output format (markdown, compact, json, yaml, csv, gherkin) |
| `--output` | `-o` | Output file path |
| `--detail` | `-d` | Detail level (minimal, standard, full) |
| `--flat` | | Flat JSON structure (for json format) |
| `--verbose` | `-v` | Verbose output with timing |
| `--debug` | | Debug output (very verbose) |
| `--quiet` | `-q` | Suppress output except errors |
| `--no-install` | | Skip auto-installation of dependencies |
| `--no-treesitter` | | Disable Tree-sitter parser |
| `--status` | | Show library availability |
| `--version` | | Show version |

## Output Formats

### Markdown (default)

```bash
code2logic /path/to/project -f markdown -o analysis.md
```

Human-readable documentation format with headers and code blocks.

### Gherkin (Best for LLM)

```bash
code2logic /path/to/project -f gherkin -o analysis.feature
```

BDD specification format. **Recommended for LLM code generation** (95% accuracy).

### CSV

```bash
code2logic /path/to/project -f csv -o analysis.csv
```

Tabular format, good for data analysis and spreadsheets.

### JSON

```bash
# Nested JSON
code2logic /path/to/project -f json -o analysis.json

# Flat JSON (for RAG/embeddings)
code2logic /path/to/project -f json --flat -o analysis.json
```

### YAML

```bash
code2logic /path/to/project -f yaml -o analysis.yaml
```

Human-readable structured format.

### Compact

```bash
code2logic /path/to/project -f compact -o analysis.txt
```

Ultra-minimal format (~200 tokens for entire project).

## Detail Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `minimal` | Basic structure only | Quick overview, low tokens |
| `standard` | Structure + metadata | General analysis |
| `full` | Everything including docstrings | Detailed analysis |

```bash
# Minimal detail (smallest output)
code2logic /path/to/project -f csv -d minimal

# Standard detail (default)
code2logic /path/to/project -f csv -d standard

# Full detail (most comprehensive)
code2logic /path/to/project -f csv -d full
```

## Examples

### Basic Analysis

```bash
# Analyze current directory
code2logic .

# Analyze specific project
code2logic /path/to/myproject
```

### Save to File

```bash
# Save as Gherkin
code2logic /path/to/project -f gherkin -o tests.feature

# Save as JSON
code2logic /path/to/project -f json -o analysis.json
```

### Verbose Mode

```bash
# Show progress and statistics
code2logic /path/to/project -v

# Output:
# CODE2LOGIC
# ══════════
#     Version: 1.0.1
# [1] Checking dependencies... (0.00s)
# [2] Analyzing project: /path/to/project (0.01s)
# ✓ Analysis complete (0.19s)
# ──────────────────────────────────────────────────
#     Files: 30
#     Lines: 10,835
#     Languages: python
#     Functions: 115
#     Classes: 28
```

### Pipe to Other Tools

```bash
# Pipe to jq for JSON processing
code2logic /path/to/project -f json | jq '.modules[0]'

# Pipe to grep
code2logic /path/to/project -f csv | grep "function"

# Count functions
code2logic /path/to/project -f csv -d minimal | wc -l
```

### Integration with LLM

```bash
# Generate context for LLM prompt
code2logic /path/to/project -f gherkin -d minimal > context.feature

# Use with Ollama
cat context.feature | ollama run qwen2.5-coder:7b "Generate tests for this"
```

## Environment Variables

```bash
# Skip dependency installation
CODE2LOGIC_NO_INSTALL=1 code2logic /path/to/project

# Enable verbose mode
CODE2LOGIC_VERBOSE=true code2logic /path/to/project
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid path, parse error, etc.) |

## See Also

- [Output Formats](output-formats.md) - Detailed format comparison
- [Python API](python-api.md) - Programmatic usage
- [Examples](examples.md) - More usage examples

---

[← Configuration](configuration.md) | [Python API →](python-api.md)
