#!/usr/bin/env python3
"""
Quick Start Tutorial - 5 minute introduction to code2logic.

This example demonstrates the basic functionality of code2logic
in a simple, easy-to-understand workflow.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic import ProjectAnalyzer, JSONGenerator, MarkdownGenerator
from code2logic.models import create_project, create_module, create_function


def create_sample_project():
    """Create a simple sample project for demonstration."""
    print("🏗️  Creating sample project...")
    
    # Create project structure
    project = create_project(
        name="quickstart_project",
        path="/tmp/quickstart_project"
    )
    
    # Add modules
    main_module = create_module(
        name="main",
        path="/tmp/quickstart_project/main.py",
        functions=[
            create_function(
                name="calculate_sum",
                parameters=["a", "b"],
                lines_of_code=3,
                complexity=1,
                docstring="Calculate sum of two numbers",
                code="def calculate_sum(a, b):\n    return a + b"
            ),
            create_function(
                name="factorial",
                parameters=["n"],
                lines_of_code=8,
                complexity=3,
                docstring="Calculate factorial",
                code="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
            )
        ],
        imports=["os", "sys"],
        lines_of_code=15
    )
    
    utils_module = create_module(
        name="utils",
        path="/tmp/quickstart_project/utils.py",
        functions=[
            create_function(
                name="validate_email",
                parameters=["email"],
                lines_of_code=5,
                complexity=2,
                docstring="Validate email address",
                code="def validate_email(email):\n    return '@' in email and '.' in email"
            )
        ],
        imports=["re"],
        lines_of_code=8
    )
    
    project.modules.extend([main_module, utils_module])
    
    print(f"✅ Created project with {len(project.modules)} modules")
    return project


def basic_analysis(project):
    """Perform basic project analysis."""
    print("\n🔍 Performing basic analysis...")
    
    # Get project statistics
    stats = project.get_statistics()
    
    print(f"📊 Project Statistics:")
    print(f"   Name: {project.name}")
    print(f"   Path: {project.path}")
    print(f"   Modules: {stats['modules']}")
    print(f"   Functions: {stats['functions']}")
    print(f"   Classes: {stats['classes']}")
    print(f"   Dependencies: {stats['dependencies']}")
    print(f"   Lines of Code: {stats['lines_of_code']}")
    
    return stats


def explore_modules(project):
    """Explore project modules in detail."""
    print("\n📁 Exploring modules...")
    
    for i, module in enumerate(project.modules, 1):
        print(f"\n{i}. Module: {module.name}")
        print(f"   Path: {module.path}")
        print(f"   Lines of Code: {module.lines_of_code}")
        print(f"   Imports: {', '.join(module.imports)}")
        
        if module.functions:
            print(f"   Functions:")
            for func in module.functions:
                print(f"     - {func.name}({', '.join(func.parameters)}) "
                      f"- {func.lines_of_code} LOC, complexity: {func.complexity}")
                if func.docstring:
                    print(f"       📝 {func.docstring}")
        
        if module.classes:
            print(f"   Classes:")
            for cls in module.classes:
                print(f"     - {cls.name} ({len(cls.methods)} methods)")


def generate_outputs(project):
    """Generate different output formats."""
    print("\n📄 Generating outputs...")
    
    # Create output directory
    output_dir = Path("./quickstart_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate JSON output
    print("   📋 Generating JSON...")
    json_gen = JSONGenerator()
    json_gen.generate(project, str(output_dir / "analysis.json"))
    
    # Generate Markdown output
    print("   📝 Generating Markdown...")
    md_gen = MarkdownGenerator()
    md_gen.generate(project, str(output_dir / "analysis.md"))
    
    print(f"✅ Outputs generated in: {output_dir}")
    return output_dir


def demonstrate_advanced_features(project):
    """Demonstrate advanced features."""
    print("\n🚀 Advanced Features Demo...")
    
    # Find complex functions
    complex_functions = []
    for module in project.modules:
        for func in module.functions:
            if func.complexity > 2:
                complex_functions.append((module.name, func.name, func.complexity))
    
    if complex_functions:
        print("🧠 Complex Functions Found:")
        for module_name, func_name, complexity in complex_functions:
            print(f"   - {module_name}.{func_name} (complexity: {complexity})")
    
    # Show import analysis
    all_imports = []
    for module in project.modules:
        all_imports.extend(module.imports)
    
    unique_imports = list(set(all_imports))
    print(f"\n📦 Import Analysis:")
    print(f"   Total imports: {len(all_imports)}")
    print(f"   Unique imports: {len(unique_imports)}")
    print(f"   Import list: {', '.join(unique_imports)}")


def show_performance_stats(start_time):
    """Show performance statistics."""
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n⏱️  Performance:")
    print(f"   Total time: {duration:.2f} seconds")
    print(f"   Analysis speed: Fast! 🚀")


def main():
    """Main quick start tutorial."""
    print("🎯 code2logic Quick Start Tutorial")
    print("=" * 50)
    print("This 5-minute tutorial will show you the basics of code2logic")
    print()
    
    start_time = time.time()
    
    try:
        # Step 1: Create sample project
        project = create_sample_project()
        
        # Step 2: Basic analysis
        stats = basic_analysis(project)
        
        # Step 3: Explore modules
        explore_modules(project)
        
        # Step 4: Generate outputs
        output_dir = generate_outputs(project)
        
        # Step 5: Advanced features
        demonstrate_advanced_features(project)
        
        # Step 6: Performance stats
        show_performance_stats(start_time)
        
        # Summary
        print("\n🎉 Quick Start Complete!")
        print("=" * 30)
        print("What you learned:")
        print("✅ How to create and analyze projects")
        print("✅ How to explore modules and functions")
        print("✅ How to generate different output formats")
        print("✅ How to identify complex code")
        print("✅ How to analyze imports")
        print()
        print("Next steps:")
        print("📖 Try the other examples in the examples/ directory")
        print("🔧 Use the CLI: code2logic /path/to/your/project")
        print("🤖 Explore LLM integration with Ollama")
        print("📊 Generate dependency graphs and similarity analysis")
        print()
        print(f"📁 Check your outputs in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This is a demonstration - the error is expected in some environments")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
