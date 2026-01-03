"""
Example: Complete BDD Workflow with code2logic.

Demonstrates the full BDD workflow:
1. Analyze codebase → code2logic
2. Generate Gherkin (~50x token compression)
3. Generate step definitions (pytest-bdd/behave/cucumber-js)
4. Run tests

Token Efficiency:
- CSV full: ~16K tokens per 100 functions
- Gherkin:  ~300 tokens per 100 functions = 50x compression

LLM Accuracy (models <30B):
- Gherkin: 95%
- YAML: 90%
- JSON: 75%
- Raw Python: 25%

Usage:
    python bdd_workflow.py /path/to/project
    python bdd_workflow.py /path/to/project --framework behave
    python bdd_workflow.py /path/to/project --lang pl
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic import (
    analyze_project,
    GherkinGenerator,
    StepDefinitionGenerator,
    CucumberYAMLGenerator,
    CSVGenerator,
    csv_to_gherkin,
    gherkin_to_test_data,
)


def calculate_token_savings(csv_content: str, gherkin_content: str) -> dict:
    """Calculate token savings between formats."""
    csv_chars = len(csv_content)
    gherkin_chars = len(gherkin_content)
    
    # Approximate tokens (~4 chars per token)
    csv_tokens = csv_chars // 4
    gherkin_tokens = gherkin_chars // 4
    
    return {
        'csv_chars': csv_chars,
        'csv_tokens': csv_tokens,
        'gherkin_chars': gherkin_chars,
        'gherkin_tokens': gherkin_tokens,
        'savings_chars': csv_chars - gherkin_chars,
        'savings_tokens': csv_tokens - gherkin_tokens,
        'compression_ratio': csv_tokens / max(gherkin_tokens, 1),
    }


def main():
    """Run BDD workflow."""
    if len(sys.argv) < 2:
        print("Usage: python bdd_workflow.py /path/to/project [options]")
        print("")
        print("Options:")
        print("  --framework FRAMEWORK  Step definition framework:")
        print("                         pytest-bdd, behave, cucumber-js")
        print("  --lang LANG            Gherkin language: en, pl, de")
        print("  --output DIR           Output directory")
        print("  --yaml                 Also generate Cucumber YAML")
        sys.exit(1)
    
    project_path = sys.argv[1]
    framework = 'pytest-bdd'
    lang = 'en'
    generate_yaml = '--yaml' in sys.argv
    output_dir = Path('./bdd_output')
    
    # Parse args
    if '--framework' in sys.argv:
        idx = sys.argv.index('--framework')
        framework = sys.argv[idx + 1]
    
    if '--lang' in sys.argv:
        idx = sys.argv.index('--lang')
        lang = sys.argv[idx + 1]
    
    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        output_dir = Path(sys.argv[idx + 1])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / 'features'
    features_dir.mkdir(exist_ok=True)
    steps_dir = output_dir / 'steps'
    steps_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # Step 1: Analyze Project
    # =========================================================================
    print("="*70)
    print("STEP 1: Analyze Project")
    print("="*70)
    
    project = analyze_project(project_path)
    print(f"Project: {project.name}")
    print(f"Files: {project.total_files}")
    print(f"Lines: {project.total_lines}")
    print(f"Languages: {list(project.languages.keys())}")
    
    # =========================================================================
    # Step 2: Generate Formats & Compare
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: Generate Formats & Token Comparison")
    print("="*70)
    
    # Generate CSV for comparison
    csv_gen = CSVGenerator()
    csv_content = csv_gen.generate(project, detail='full')
    
    # Generate Gherkin
    gherkin_gen = GherkinGenerator(language=lang)
    gherkin_content = gherkin_gen.generate(project, detail='standard')
    
    # Calculate savings
    savings = calculate_token_savings(csv_content, gherkin_content)
    
    print(f"\nToken Efficiency Comparison:")
    print(f"  CSV (full):   {savings['csv_chars']:>10,} chars / ~{savings['csv_tokens']:,} tokens")
    print(f"  Gherkin:      {savings['gherkin_chars']:>10,} chars / ~{savings['gherkin_tokens']:,} tokens")
    print(f"  Compression:  {savings['compression_ratio']:.1f}x")
    print(f"  Savings:      {savings['savings_tokens']:,} tokens")
    
    # Save Gherkin
    gherkin_file = features_dir / f"{project.name}.feature"
    gherkin_file.write_text(gherkin_content)
    print(f"\nGherkin saved: {gherkin_file}")
    
    # =========================================================================
    # Step 3: Generate Step Definitions
    # =========================================================================
    print("\n" + "="*70)
    print(f"STEP 3: Generate Step Definitions ({framework})")
    print("="*70)
    
    features = gherkin_gen.generate_test_scenarios(project)
    step_gen = StepDefinitionGenerator()
    
    if framework == 'pytest-bdd':
        steps = step_gen.generate_pytest_bdd(features)
        steps_file = steps_dir / 'test_steps.py'
    elif framework == 'behave':
        steps = step_gen.generate_behave(features)
        steps_file = steps_dir / 'steps.py'
    elif framework == 'cucumber-js':
        steps = step_gen.generate_cucumber_js(features)
        steps_file = steps_dir / 'steps.js'
    else:
        print(f"Unknown framework: {framework}")
        sys.exit(1)
    
    steps_file.write_text(steps)
    print(f"Step definitions saved: {steps_file}")
    
    print(f"\nGenerated {len(features)} features:")
    for feature in features:
        print(f"  - {feature.name} ({len(feature.scenarios)} scenarios)")
    
    # =========================================================================
    # Step 4: Generate Cucumber YAML (optional)
    # =========================================================================
    if generate_yaml:
        print("\n" + "="*70)
        print("STEP 4: Generate Cucumber YAML")
        print("="*70)
        
        yaml_gen = CucumberYAMLGenerator()
        yaml_content = yaml_gen.generate(project, detail='standard')
        
        yaml_file = output_dir / 'cucumber.yaml'
        yaml_file.write_text(yaml_content)
        print(f"Cucumber YAML saved: {yaml_file}")
    
    # =========================================================================
    # Step 5: Generate Test Data for LLM
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: Generate LLM-Optimized Test Data")
    print("="*70)
    
    test_data = gherkin_to_test_data(gherkin_content)
    
    test_data_file = output_dir / 'test_data.json'
    test_data_file.write_text(json.dumps(test_data, indent=2))
    print(f"Test data saved: {test_data_file}")
    
    print(f"\nTest data summary:")
    print(f"  Features: {len(test_data['features'])}")
    print(f"  Total scenarios: {test_data['total_scenarios']}")
    print(f"  Token compression: {test_data['compression']}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("BDD WORKFLOW COMPLETE")
    print("="*70)
    
    print(f"\nGenerated files:")
    for f in output_dir.rglob('*'):
        if f.is_file():
            print(f"  {f.relative_to(output_dir)}: {f.stat().st_size:,} bytes")
    
    print(f"\nNext steps for {framework}:")
    
    if framework == 'pytest-bdd':
        print("  1. pip install pytest-bdd")
        print("  2. Implement TODO steps in steps/test_steps.py")
        print("  3. pytest --bdd features/")
    elif framework == 'behave':
        print("  1. pip install behave")
        print("  2. Implement TODO steps in steps/steps.py")
        print("  3. behave features/")
    elif framework == 'cucumber-js':
        print("  1. npm install @cucumber/cucumber")
        print("  2. Implement TODO steps in steps/steps.js")
        print("  3. npx cucumber-js features/")
    
    print(f"\nTotal token savings: {savings['savings_tokens']:,} tokens ({savings['compression_ratio']:.0f}x)")


if __name__ == '__main__':
    main()
