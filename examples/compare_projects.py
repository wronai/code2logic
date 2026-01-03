#!/usr/bin/env python3
"""
Compare multiple projects using code2logic.

This example demonstrates how to analyze and compare multiple code projects
to identify similarities, differences, and patterns.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic.analyzer import ProjectAnalyzer
from code2logic.models import Project, Module, Function, Class, Similarity
from code2logic.dependency import DependencyAnalyzer
from code2logic.similarity import SimilarityDetector


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def analyze_projects(project_paths: List[str]) -> List[Project]:
    """Analyze multiple projects."""
    projects = []
    
    for project_path in project_paths:
        print(f"🔍 Analyzing project: {project_path}")
        
        try:
            analyzer = ProjectAnalyzer(project_path)
            project = analyzer.analyze()
            projects.append(project)
            
            stats = project.get_statistics()
            print(f"  ✅ {stats['modules']} modules, {stats['functions']} functions, {stats['classes']} classes")
            
        except Exception as e:
            print(f"  ❌ Failed to analyze {project_path}: {e}")
            continue
    
    return projects


def compare_project_statistics(projects: List[Project]) -> Dict[str, Any]:
    """Compare basic statistics between projects."""
    print("\n📊 Comparing project statistics...")
    
    comparison = {
        "projects": [],
        "summary": {}
    }
    
    # Collect statistics
    for project in projects:
        stats = project.get_statistics()
        project_info = {
            "name": project.name,
            "path": project.path,
            "statistics": stats
        }
        comparison["projects"].append(project_info)
    
    # Calculate summary statistics
    if len(projects) > 1:
        all_stats = [p["statistics"] for p in comparison["projects"]]
        
        comparison["summary"] = {
            "total_projects": len(projects),
            "average_modules": sum(s["modules"] for s in all_stats) / len(all_stats),
            "average_functions": sum(s["functions"] for s in all_stats) / len(all_stats),
            "average_classes": sum(s["classes"] for s in all_stats) / len(all_stats),
            "average_dependencies": sum(s["dependencies"] for s in all_stats) / len(all_stats),
            "average_loc": sum(s["lines_of_code"] for s in all_stats) / len(all_stats),
            "total_loc": sum(s["lines_of_code"] for s in all_stats)
        }
    
    return comparison


def find_similar_modules(projects: List[Project]) -> List[Dict[str, Any]]:
    """Find similar modules across projects."""
    print("\n🔍 Finding similar modules across projects...")
    
    # Collect all modules from all projects
    all_modules = []
    for project in projects:
        for module in project.modules:
            all_modules.append((project.name, module))
    
    # Use similarity detector
    similarity_detector = SimilarityDetector()
    similarities = similarity_detector.detect_similarities(
        [module for _, module in all_modules]
    )
    
    # Filter and format similarities
    similar_modules = []
    for similarity in similarities:
        if similarity.score >= 0.6:  # Threshold for similarity
            similar_modules.append({
                "item1": similarity.item1,
                "item2": similarity.item2,
                "score": similarity.score,
                "type": similarity.similarity_type,
                "details": similarity.details
            })
    
    # Sort by similarity score
    similar_modules.sort(key=lambda x: x["score"], reverse=True)
    
    return similar_modules[:20]  # Return top 20 similarities


def compare_dependencies(projects: List[Project]) -> Dict[str, Any]:
    """Compare dependency patterns across projects."""
    print("\n🔗 Comparing dependency patterns...")
    
    dependency_analysis = {
        "projects": {},
        "common_dependencies": set(),
        "unique_dependencies": {},
        "dependency_density": {}
    }
    
    # Analyze each project's dependencies
    for project in projects:
        project_deps = set()
        dependency_types = {}
        
        for dep in project.dependencies:
            project_deps.add(dep.target)
            dep_type = dep.type.value if hasattr(dep.type, 'value') else str(dep.type)
            dependency_types[dep_type] = dependency_types.get(dep_type, 0) + 1
        
        dependency_analysis["projects"][project.name] = {
            "total_dependencies": len(project.dependencies),
            "unique_targets": len(project_deps),
            "dependency_types": dependency_types,
            "dependencies": list(project_deps)
        }
        
        # Calculate dependency density (dependencies per module)
        density = len(project.dependencies) / max(len(project.modules), 1)
        dependency_analysis["dependency_density"][project.name] = density
    
    # Find common dependencies across projects
    if len(projects) > 1:
        project_names = list(dependency_analysis["projects"].keys())
        common_deps = set(dependency_analysis["projects"][project_names[0]]["dependencies"])
        
        for project_name in project_names[1:]:
            project_deps = set(dependency_analysis["projects"][project_name]["dependencies"])
            common_deps &= project_deps
        
        dependency_analysis["common_dependencies"] = list(common_deps)
        
        # Find unique dependencies for each project
        for project_name in project_names:
            project_deps = set(dependency_analysis["projects"][project_name]["dependencies"])
            unique_deps = project_deps - set(common_deps)
            dependency_analysis["unique_dependencies"][project_name] = list(unique_deps)
    
    return dependency_analysis


def analyze_complexity_patterns(projects: List[Project]) -> Dict[str, Any]:
    """Analyze complexity patterns across projects."""
    print("\n🧠 Analyzing complexity patterns...")
    
    complexity_analysis = {
        "projects": {},
        "summary": {}
    }
    
    all_functions = []
    all_classes = []
    
    for project in projects:
        project_complexity = {
            "functions": [],
            "classes": [],
            "average_function_complexity": 0,
            "average_class_size": 0,
            "complex_functions": [],
            "large_classes": []
        }
        
        function_complexities = []
        class_sizes = []
        
        # Analyze functions
        for module in project.modules:
            for func in module.functions:
                func_info = {
                    "name": f"{module.name}.{func.name}",
                    "complexity": func.complexity,
                    "lines_of_code": func.lines_of_code,
                    "has_docstring": func.docstring is not None
                }
                project_complexity["functions"].append(func_info)
                function_complexities.append(func.complexity)
                all_functions.append(func_info)
                
                if func.complexity > 10:
                    project_complexity["complex_functions"].append(func_info["name"])
        
        # Analyze classes
        for module in project.modules:
            for cls in module.classes:
                class_info = {
                    "name": f"{module.name}.{cls.name}",
                    "methods": len(cls.methods),
                    "lines_of_code": cls.lines_of_code,
                    "has_docstring": cls.docstring is not None,
                    "base_classes": cls.base_classes
                }
                project_complexity["classes"].append(class_info)
                class_sizes.append(len(cls.methods))
                all_classes.append(class_info)
                
                if len(cls.methods) > 15:
                    project_complexity["large_classes"].append(class_info["name"])
        
        # Calculate averages
        project_complexity["average_function_complexity"] = (
            sum(function_complexities) / len(function_complexities) if function_complexities else 0
        )
        project_complexity["average_class_size"] = (
            sum(class_sizes) / len(class_sizes) if class_sizes else 0
        )
        
        complexity_analysis["projects"][project.name] = project_complexity
    
    # Calculate overall summary
    if all_functions:
        all_complexities = [f["complexity"] for f in all_functions]
        complexity_analysis["summary"] = {
            "total_functions": len(all_functions),
            "average_complexity": sum(all_complexities) / len(all_complexities),
            "max_complexity": max(all_complexities),
            "complex_functions_count": len([f for f in all_functions if f["complexity"] > 10]),
            "functions_with_docs": len([f for f in all_functions if f["has_docstring"]])
        }
    
    if all_classes:
        all_method_counts = [c["methods"] for c in all_classes]
        complexity_analysis["summary"].update({
            "total_classes": len(all_classes),
            "average_methods": sum(all_method_counts) / len(all_method_counts),
            "max_methods": max(all_method_counts),
            "large_classes_count": len([c for c in all_classes if c["methods"] > 15]),
            "classes_with_docs": len([c for c in all_classes if c["has_docstring"]])
        })
    
    return complexity_analysis


def identify_code_patterns(projects: List[Project]) -> Dict[str, Any]:
    """Identify common code patterns and conventions."""
    print("\n🎨 Identifying code patterns...")
    
    patterns_analysis = {
        "naming_conventions": {},
        "documentation_patterns": {},
        "import_patterns": {},
        "architecture_patterns": {}
    }
    
    # Analyze naming conventions
    function_names = []
    class_names = []
    module_names = []
    
    for project in projects:
        for module in project.modules:
            module_names.append(module.name)
            
            for func in module.functions:
                function_names.append(func.name)
            
            for cls in module.classes:
                class_names.append(cls.name)
    
    # Naming patterns
    patterns_analysis["naming_conventions"] = {
        "snake_case_functions": len([n for n in function_names if '_' in n and n.islower()]),
        "camel_case_functions": len([n for n in function_names if n[0].islower() and not '_' in n]),
        "pascal_case_classes": len([n for n in class_names if n[0].isupper() and not '_' in n]),
        "snake_case_modules": len([n for n in module_names if '_' in n and n.islower()])
    }
    
    # Documentation patterns
    total_functions = len(function_names)
    total_classes = len(class_names)
    
    functions_with_docs = 0
    classes_with_docs = 0
    
    for project in projects:
        for module in project.modules:
            for func in module.functions:
                if func.docstring:
                    functions_with_docs += 1
            
            for cls in module.classes:
                if cls.docstring:
                    classes_with_docs += 1
    
    patterns_analysis["documentation_patterns"] = {
        "functions_with_docstrings": functions_with_docs,
        "classes_with_docstrings": classes_with_docs,
        "function_documentation_coverage": functions_with_docs / total_functions if total_functions > 0 else 0,
        "class_documentation_coverage": classes_with_docs / total_classes if total_classes > 0 else 0
    }
    
    # Import patterns
    all_imports = []
    import_types = {"standard_library": 0, "third_party": 0, "local": 0}
    
    for project in projects:
        for module in project.modules:
            all_imports.extend(module.imports)
    
    # Categorize imports (simplified)
    standard_lib_modules = {'os', 'sys', 'json', 're', 'datetime', 'collections', 'itertools', 'functools'}
    
    for imp in all_imports:
        imp_base = imp.split('.')[0]
        if imp_base in standard_lib_modules:
            import_types["standard_library"] += 1
        elif imp.startswith(('http', 'sql', 'numpy', 'pandas', 'requests', 'django', 'flask')):
            import_types["third_party"] += 1
        else:
            import_types["local"] += 1
    
    patterns_analysis["import_patterns"] = import_types
    
    return patterns_analysis


def generate_comparison_report(
    projects: List[Project],
    output_path: str
) -> None:
    """Generate comprehensive comparison report."""
    print(f"\n📝 Generating comparison report...")
    
    report = {
        "metadata": {
            "generated_at": str(Path.cwd()),
            "projects_analyzed": len(projects),
            "project_names": [p.name for p in projects]
        },
        "statistics_comparison": compare_project_statistics(projects),
        "similar_modules": find_similar_modules(projects),
        "dependency_comparison": compare_dependencies(projects),
        "complexity_analysis": analyze_complexity_patterns(projects),
        "code_patterns": identify_code_patterns(projects)
    }
    
    # Save report
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Comparison report saved to: {output_path}")
        
        # Print summary
        print("\n📋 Comparison Summary:")
        print(f"  Projects analyzed: {len(projects)}")
        print(f"  Similar modules found: {len(report['similar_modules'])}")
        print(f"  Common dependencies: {len(report['dependency_comparison']['common_dependencies'])}")
        
        if report['statistics_comparison']['summary']:
            summary = report['statistics_comparison']['summary']
            print(f"  Average LOC per project: {summary['average_loc']:.1f}")
            print(f"  Average modules per project: {summary['average_modules']:.1f}")
        
    except Exception as e:
        print(f"❌ Failed to save report: {e}")


def print_summary(projects: List[Project]) -> None:
    """Print a summary of the comparison."""
    print("\n" + "="*60)
    print("PROJECT COMPARISON SUMMARY")
    print("="*60)
    
    for i, project in enumerate(projects, 1):
        stats = project.get_statistics()
        print(f"\n{i}. {project.name}")
        print(f"   Path: {project.path}")
        print(f"   Modules: {stats['modules']}")
        print(f"   Functions: {stats['functions']}")
        print(f"   Classes: {stats['classes']}")
        print(f"   Dependencies: {stats['dependencies']}")
        print(f"   Lines of Code: {stats['lines_of_code']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Compare multiple code projects using code2logic'
    )
    
    parser.add_argument(
        'projects',
        nargs='+',
        help='Paths to project directories to compare'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./project_comparison.json',
        help='Output file for comparison report'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.6,
        help='Threshold for module similarity detection'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate project paths
    valid_projects = []
    for project_path in args.projects:
        path = Path(project_path)
        if not path.exists():
            print(f"❌ Project path does not exist: {project_path}")
            continue
        if not path.is_dir():
            print(f"❌ Path is not a directory: {project_path}")
            continue
        
        valid_projects.append(str(path))
    
    if not valid_projects:
        print("❌ No valid project paths provided")
        sys.exit(1)
    
    print(f"🚀 Comparing {len(valid_projects)} projects...")
    
    # Analyze projects
    projects = analyze_projects(valid_projects)
    
    if not projects:
        print("❌ No projects could be analyzed")
        sys.exit(1)
    
    # Generate comparison report
    generate_comparison_report(projects, args.output)
    
    # Print summary
    print_summary(projects)
    
    print(f"\n✅ Project comparison completed!")
    print(f"📄 Detailed report available at: {args.output}")


if __name__ == '__main__':
    main()
