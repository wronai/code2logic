# Logic2Test Examples

Example scripts demonstrating logic2test usage.

## Prerequisites

```bash
# Install logic2test
pip install logic2test

# Or for property testing
pip install logic2test[hypothesis]
```

## Examples

### 01_quickstart.py

Basic test generation from Code2Logic output.

```bash
python 01_quickstart.py
```

Demonstrates:

- Loading a YAML logic file
- Getting project summary
- Generating unit tests
- Custom configuration

### 02_custom_templates.py

Customizing test generation templates.

```bash
python 02_custom_templates.py
```

Demonstrates:

- Function test templates
- Class test templates
- Dataclass test templates
- Async function tests

### sample_project.yaml

Sample Code2Logic output file for testing.

## CLI Usage

```bash
# Show summary
logic2test sample_project.yaml --summary

# Generate unit tests
logic2test sample_project.yaml -o tests/

# Generate all test types
logic2test sample_project.yaml -o tests/ --type all

# Include private methods
logic2test sample_project.yaml -o tests/ --include-private
```

## Generated Test Structure

```
tests/
├── unit/
│   ├── test_calculator.py
│   ├── test_user.py
│   └── test_client.py
├── integration/
│   └── test_integration.py
└── property/
    └── test_properties.py
```
