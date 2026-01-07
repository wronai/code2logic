# Logic2Code Examples

Example scripts demonstrating logic2code usage.

## Prerequisites

```bash
# Install logic2code
pip install logic2code

# For LLM-enhanced generation
pip install logic2code[llm]
```

## Examples

### 01_quickstart.py

Basic code generation from Code2Logic output.

```bash
python 01_quickstart.py
```

Demonstrates:

- Loading a YAML logic file
- Getting project summary
- Generating Python code
- Custom configuration
- Stubs-only generation

### 02_llm_enhanced.py

LLM-enhanced code generation.

```bash
# First, configure LLM provider
export OPENROUTER_API_KEY="your-key"

python 02_llm_enhanced.py
```

Demonstrates:

- LLM-powered implementations
- Hybrid generation workflow
- Comparing stub vs full output

### sample_project.yaml

Sample Code2Logic output file for testing.

## CLI Usage

```bash
# Show summary
logic2code sample_project.yaml --summary

# Generate Python code
logic2code sample_project.yaml -o output/

# Generate stubs only
logic2code sample_project.yaml -o output/ --stubs-only

# Specific modules only
logic2code sample_project.yaml -o output/ --modules calculator.py
```

## Workflow: Logic to Code

```bash
# 1. Analyze existing code
code2logic src/ -f yaml -o project.yaml

# 2. Modify logic (add functions, change signatures)
# Edit project.yaml

# 3. Generate new code
logic2code project.yaml -o new_src/

# 4. Compare and merge
diff -r src/ new_src/
```

## Generated Structure

```text
output/
├── __init__.py
├── calculator.py
├── models/
│   ├── __init__.py
│   └── product.py
└── services/
    ├── __init__.py
    └── inventory.py
```
