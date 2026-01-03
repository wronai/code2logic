#!/usr/bin/env python3
"""
Quick Start Guide for code2logic.

This script demonstrates the most common use cases in 5 minutes.

Run this script to see all features in action:
    python quick_start.py

Or import code2logic in your own code:
    from code2logic import analyze_project, GherkinGenerator
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                     CODE2LOGIC QUICK START GUIDE                      ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
    
    # =========================================================================
    # 1. Basic Analysis
    # =========================================================================
    print("="*70)
    print("1. BASIC ANALYSIS")
    print("="*70)
    print("""
>>> from code2logic import analyze_project
>>> project = analyze_project("/path/to/project")
>>> print(f"Files: {project.total_files}, Lines: {project.total_lines}")
""")
    
    from code2logic import analyze_project
    project = analyze_project(".")
    print(f"Result: Files: {project.total_files}, Lines: {project.total_lines}")
    print(f"        Languages: {list(project.languages.keys())}")
    
    # =========================================================================
    # 2. Output Formats
    # =========================================================================
    print("\n" + "="*70)
    print("2. OUTPUT FORMATS (6 formats, 3 detail levels)")
    print("="*70)
    print("""
Available formats:
  - csv      : Best for LLM (~50% smaller than JSON)
  - gherkin  : Best for LLM (50x compression, 95% accuracy)
  - json     : Standard format (nested or flat)
  - yaml     : Human-readable
  - compact  : Ultra-minimal
  - markdown : Documentation
  
Detail levels: minimal, standard, full
""")
    
    from code2logic import CSVGenerator, GherkinGenerator
    
    csv_gen = CSVGenerator()
    csv_output = csv_gen.generate(project, detail='standard')
    
    gherkin_gen = GherkinGenerator()
    gherkin_output = gherkin_gen.generate(project, detail='standard')
    
    csv_tokens = len(csv_output) // 4
    gherkin_tokens = len(gherkin_output) // 4
    
    print(f"Token comparison:")
    print(f"  CSV standard:     ~{csv_tokens:,} tokens")
    print(f"  Gherkin standard: ~{gherkin_tokens:,} tokens")
    print(f"  Compression:      {csv_tokens/max(gherkin_tokens,1):.0f}x")
    
    # =========================================================================
    # 3. CLI Usage
    # =========================================================================
    print("\n" + "="*70)
    print("3. COMMAND-LINE USAGE")
    print("="*70)
    print("""
# Basic analysis (auto-installs dependencies)
code2logic /path/to/project

# Specific format and detail level
code2logic /path/to/project -f csv -d full

# Gherkin for BDD tests (50x token savings!)
code2logic /path/to/project -f gherkin -o tests.feature

# JSON flat for RAG/embeddings
code2logic /path/to/project -f json --flat -o analysis.json

# Skip auto-install
code2logic /path/to/project --no-install

# Check library status
code2logic --status
""")
    
    # =========================================================================
    # 4. BDD/Gherkin Generation
    # =========================================================================
    print("="*70)
    print("4. BDD/GHERKIN GENERATION (recommended for LLM)")
    print("="*70)
    print("""
Why Gherkin?
  - 50x token compression vs CSV
  - 95% LLM accuracy (models <30B)
  - Native BDD test framework integration
  - Business-readable specifications
""")
    
    from code2logic import GherkinGenerator, StepDefinitionGenerator
    
    gherkin_gen = GherkinGenerator(language='en')
    features = gherkin_gen.generate_test_scenarios(project)
    
    print(f"Generated {len(features)} features:")
    for feature in features[:3]:
        print(f"  - {feature.name} ({len(feature.scenarios)} scenarios)")
    
    print("\nGherkin preview:")
    print("-"*40)
    print(gherkin_output[:400])
    print("...")
    
    # =========================================================================
    # 5. Python API
    # =========================================================================
    print("\n" + "="*70)
    print("5. PYTHON API")
    print("="*70)
    print("""
# Analyze and explore
from code2logic import analyze_project

project = analyze_project("/path/to/project")

for module in project.modules:
    print(f"Module: {module.path}")
    for func in module.functions:
        print(f"  Function: {func.name}")
        print(f"    Intent: {func.intent}")
        print(f"    Params: {func.params}")

# Generate Gherkin for LLM
from code2logic import GherkinGenerator

gen = GherkinGenerator()
gherkin = gen.generate(project, detail='standard')

# Convert CSV to Gherkin (utility)
from code2logic import csv_to_gherkin

gherkin = csv_to_gherkin(csv_content)
""")
    
    # =========================================================================
    # 6. LLM Integration
    # =========================================================================
    print("="*70)
    print("6. LLM INTEGRATION (Ollama/LiteLLM)")
    print("="*70)
    print("""
# Using with local Ollama
from code2logic.llm import CodeAnalyzer

analyzer = CodeAnalyzer(model="qwen2.5-coder:7b")

# Get refactoring suggestions
suggestions = analyzer.suggest_refactoring(project)

# Find semantic duplicates
duplicates = analyzer.find_semantic_duplicates(project)

# Generate code in another language
code = analyzer.generate_code(project, target_lang="typescript")
""")
    
    # =========================================================================
    # 7. MCP Server (Claude Desktop)
    # =========================================================================
    print("="*70)
    print("7. MCP SERVER (Claude Desktop Integration)")
    print("="*70)
    print("""
Add to claude_desktop_config.json:

{
  "mcpServers": {
    "code2logic": {
      "command": "python",
      "args": ["-m", "code2logic.mcp_server"]
    }
  }
}

Then in Claude Desktop:
  "Analyze my project at /path/to/project"
  "Find duplicates in my codebase"
  "Generate BDD tests for my API"
""")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("="*70)
    print("EXAMPLES AVAILABLE")
    print("="*70)
    print("""
Run these examples for more details:

  python examples/bdd_workflow.py .           # Complete BDD workflow
  python examples/token_efficiency.py .       # Compare all formats
  python examples/duplicate_detection.py .    # Find duplicates
  python examples/generate_tests.py . pytest  # Generate test files
  python examples/translate_code.py . ts      # Translate to TypeScript
  python examples/llm_pipeline.py .           # LLM integration
  python examples/api_usage.py                # API demonstration

Or use the master script:

  ./examples/run_examples.sh analyze .
  ./examples/run_examples.sh gherkin . --steps
  ./examples/run_examples.sh compare /project1 /project2
  ./examples/run_examples.sh batch /projects/dir
""")
    
    print("\n✅ Quick start complete! Happy coding!")


if __name__ == '__main__':
    main()
