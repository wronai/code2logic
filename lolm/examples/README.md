# LOLM Examples

Example scripts demonstrating lolm usage.

## Prerequisites

```bash
# Install lolm
pip install lolm

# Or install with all features
pip install lolm[full]

# Configure at least one provider:
export OPENROUTER_API_KEY="sk-or-v1-..."
# or have Ollama running locally
```

## Examples

### 01_quickstart.py

Basic usage - getting started with lolm.

```bash
python 01_quickstart.py
```

Demonstrates:
- Auto-detecting provider
- Using specific provider
- LLMManager for control
- Fallback between providers

### 02_configuration.py

Configuration and settings management.

```bash
python 02_configuration.py
```

Demonstrates:
- Default models and priorities
- Recommended models
- User configuration
- Environment variables

### 03_code_generation.py

Using lolm for code generation tasks.

```bash
python 03_code_generation.py
```

Demonstrates:
- Generating Python functions
- Generating Python classes
- Code explanation
- Code review

## Quick Test

```bash
# Check provider status
python -m lolm status

# Test generation
python -m lolm test

# List recommended models
python -m lolm models
```
