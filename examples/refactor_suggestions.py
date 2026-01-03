#!/usr/bin/env python3
"""
Generate refactoring suggestions using code2logic.

This example demonstrates how to use code2logic's intent analysis
and LLM integration to provide intelligent refactoring suggestions.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic.analyzer import ProjectAnalyzer
from code2logic.intent import IntentAnalyzer, IntentType
from code2logic.llm import LLMInterface, LLMConfig
from code2logic.models import Project, Module, Function, Class, CodeSmell


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def analyze_code_smells(project: Project) -> List[CodeSmell]:
    """Analyze code smells in the project."""
    print("🔍 Analyzing code smells...")
    
    intent_analyzer = IntentAnalyzer()
    code_smells_data = intent_analyzer.detect_code_smells(project)
    
    code_smells = []
    for smell_data in code_smells_data:
        code_smell = CodeSmell(
            type=smell_data['type'],
            severity=smell_data['severity'],
            target=smell_data['target'],
            description=smell_data['description'],
            suggestion=smell_data['suggestion']
        )
        code_smells.append(code_smell)
    
    print(f"  Found {len(code_smells)} code smells")
    return code_smells


def generate_intent_suggestions(project: Project, query: str) -> Dict[str, Any]:
    """Generate suggestions based on user intent."""
    print(f"🎯 Analyzing intent: '{query}'")
    
    intent_analyzer = IntentAnalyzer()
    intents = intent_analyzer.analyze_intent(query, project)
    
    suggestions = {
        "query": query,
        "detected_intents": [],
        "suggestions": []
    }
    
    for intent in intents:
        intent_info = {
            "type": intent.type.value,
            "confidence": intent.confidence,
            "target": intent.target,
            "description": intent.description
        }
        suggestions["detected_intents"].append(intent_info)
        suggestions["suggestions"].extend(intent.suggestions)
    
    return suggestions


def generate_llm_suggestions(
    llm: LLMInterface,
    target: str,
    project: Project,
    suggestion_type: str = "refactor"
) -> Dict[str, Any]:
    """Generate LLM-powered suggestions."""
    print(f"🤖 Generating {suggestion_type} suggestions for: {target}")
    
    # Find target object
    target_obj = None
    target_type = None
    
    parts = target.split('.')
    if len(parts) == 1:
        # Module
        target_obj = project.get_module_by_name(parts[0])
        target_type = "module"
    elif len(parts) == 2:
        # Module.Class or Module.Function
        module_name, item_name = parts
        module = project.get_module_by_name(module_name)
        if module:
            # Check for class
            for cls in module.classes:
                if cls.name == item_name:
                    target_obj = cls
                    target_type = "class"
                    break
            # Check for function
            if not target_obj:
                for func in module.functions:
                    if func.name == item_name:
                        target_obj = func
                        target_type = "function"
                        break
    
    if not target_obj:
        return {"error": f"Target not found: {target}"}
    
    try:
        if suggestion_type == "refactor":
            if isinstance(target_obj, Function):
                suggestions = llm.suggest_refactoring(target_obj, project)
            elif isinstance(target_obj, Class):
                suggestions = llm.suggest_refactoring(target_obj, project)
            elif isinstance(target_obj, Module):
                suggestions = llm.suggest_refactoring(target_obj, project)
            else:
                suggestions = ["No refactoring suggestions available for this target type"]
        
        elif suggestion_type == "optimize":
            suggestions = generate_optimization_suggestions(llm, target_obj, target_type)
        
        elif suggestion_type == "document":
            if isinstance(target_obj, (Function, Class)):
                doc = llm.generate_documentation(target_obj)
                suggestions = [f"Add documentation:\n{doc}"]
            else:
                suggestions = ["Documentation generation not available for this target type"]
        
        else:
            suggestions = ["Unknown suggestion type"]
        
        return {
            "target": target,
            "type": target_type,
            "suggestions": suggestions
        }
        
    except Exception as e:
        return {"error": f"LLM suggestion generation failed: {e}"}


def generate_optimization_suggestions(
    llm: LLMInterface,
    target_obj,
    target_type: str
) -> List[str]:
    """Generate performance optimization suggestions."""
    if isinstance(target_obj, Function):
        prompt = f"""
Suggest performance optimizations for this function:

```python
{target_obj.code}
```

Focus on:
1. Algorithm efficiency
2. Memory usage
3. Computational complexity
4. Caching opportunities
5. Parallel processing potential

Provide specific, actionable suggestions.
"""
    else:
        prompt = f"""
Suggest performance optimizations for this {target_type}:

{target_obj}

Focus on:
1. Memory efficiency
2. Computational overhead
3. Initialization costs
4. Caching strategies
5. Resource management

Provide specific, actionable suggestions.
"""
    
    try:
        response = llm._call_llm(prompt)
        return [response.strip()]
    except Exception:
        return ["Failed to generate optimization suggestions"]


def analyze_dependencies(project: Project) -> Dict[str, Any]:
    """Analyze dependency patterns for refactoring opportunities."""
    print("🔗 Analyzing dependencies...")
    
    dependency_analysis = {
        "circular_dependencies": [],
        "high_coupling": [],
        "unused_imports": [],
        "refactoring_opportunities": []
    }
    
    # Find circular dependencies
    from code2logic.dependency import DependencyAnalyzer
    dep_analyzer = DependencyAnalyzer()
    
    if project.dependencies:
        dep_analyzer.graph = dep_analyzer.analyze_dependencies(project.modules)
        circular_deps = dep_analyzer.get_circular_dependencies()
        
        for cycle in circular_deps:
            dependency_analysis["circular_dependencies"].append({
                "cycle": " -> ".join(cycle),
                "length": len(cycle),
                "suggestion": "Break the circular dependency by introducing an interface or using dependency injection"
            })
    
    # Find high coupling modules
    module_deps = {}
    for dep in project.dependencies:
        source_module = dep.source.split('.')[0]
        module_deps[source_module] = module_deps.get(source_module, 0) + 1
    
    for module, dep_count in module_deps.items():
        if dep_count > 10:  # Threshold for high coupling
            dependency_analysis["high_coupling"].append({
                "module": module,
                "dependencies": dep_count,
                "suggestion": f"Consider reducing dependencies of {module} or splitting it into smaller modules"
            })
    
    return dependency_analysis


def generate_refactoring_plan(
    project: Project,
    code_smells: List[CodeSmell],
    intent_suggestions: Dict[str, Any],
    llm_suggestions: List[Dict[str, Any]],
    dependency_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a comprehensive refactoring plan."""
    print("📋 Generating refactoring plan...")
    
    plan = {
        "project": project.name,
        "summary": {
            "code_smells_found": len(code_smells),
            "high_priority_issues": len([s for s in code_smells if s.severity in ['high', 'critical']]),
            "intent_suggestions": len(intent_suggestions.get("suggestions", [])),
            "llm_suggestions": len(llm_suggestions),
            "dependency_issues": len(dependency_analysis.get("circular_dependencies", [])) + 
                              len(dependency_analysis.get("high_coupling", []))
        },
        "prioritized_actions": [],
        "detailed_analysis": {
            "code_smells": [],
            "intent_analysis": intent_suggestions,
            "llm_recommendations": llm_suggestions,
            "dependency_issues": dependency_analysis
        }
    }
    
    # Prioritize code smells by severity
    critical_smells = [s for s in code_smells if s.severity == 'critical']
    high_smells = [s for s in code_smells if s.severity == 'high']
    medium_smells = [s for s in code_smells if s.severity == 'medium']
    low_smells = [s for s in code_smells if s.severity == 'low']
    
    # Create prioritized actions
    for smell in critical_smells + high_smells:
        plan["prioritized_actions"].append({
            "priority": "high",
            "type": "code_smell",
            "target": smell.target,
            "description": smell.description,
            "suggestion": smell.suggestion,
            "estimated_effort": "medium"
        })
    
    # Add dependency issues
    for circular_dep in dependency_analysis.get("circular_dependencies", []):
        plan["prioritized_actions"].append({
            "priority": "high",
            "type": "dependency",
            "target": circular_dep["cycle"],
            "description": f"Circular dependency detected: {circular_dep['cycle']}",
            "suggestion": circular_dep["suggestion"],
            "estimated_effort": "high"
        })
    
    # Add medium priority items
    for smell in medium_smells:
        plan["prioritized_actions"].append({
            "priority": "medium",
            "type": "code_smell",
            "target": smell.target,
            "description": smell.description,
            "suggestion": smell.suggestion,
            "estimated_effort": "low"
        })
    
    # Add LLM suggestions
    for llm_suggestion in llm_suggestions:
        if "error" not in llm_suggestion:
            plan["prioritized_actions"].append({
                "priority": "medium",
                "type": "llm_suggestion",
                "target": llm_suggestion["target"],
                "description": f"AI-powered refactoring suggestions for {llm_suggestion['type']}",
                "suggestion": "\n".join(llm_suggestion["suggestions"]),
                "estimated_effort": "variable"
            })
    
    # Format code smells for detailed analysis
    for smell in code_smells:
        plan["detailed_analysis"]["code_smells"].append({
            "type": smell.type,
            "severity": smell.severity,
            "target": smell.target,
            "description": smell.description,
            "suggestion": smell.suggestion
        })
    
    return plan


def save_refactoring_plan(plan: Dict[str, Any], output_path: str) -> None:
    """Save refactoring plan to file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        print(f"✅ Refactoring plan saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save refactoring plan: {e}")


def print_refactoring_summary(plan: Dict[str, Any]) -> None:
    """Print a summary of the refactoring plan."""
    print("\n" + "="*60)
    print("REFACTORING PLAN SUMMARY")
    print("="*60)
    
    summary = plan["summary"]
    print(f"\nProject: {plan['project']}")
    print(f"Code smells found: {summary['code_smells_found']}")
    print(f"High priority issues: {summary['high_priority_issues']}")
    print(f"Intent-based suggestions: {summary['intent_suggestions']}")
    print(f"AI-powered suggestions: {summary['llm_suggestions']}")
    print(f"Dependency issues: {summary['dependency_issues']}")
    
    print(f"\nTop 5 Prioritized Actions:")
    for i, action in enumerate(plan["prioritized_actions"][:5], 1):
        print(f"{i}. [{action['priority'].upper()}] {action['target']}")
        print(f"   {action['description']}")
        print(f"   Effort: {action['estimated_effort']}")
        print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate refactoring suggestions using code2logic'
    )
    
    parser.add_argument(
        'target',
        help='Target project path or file to analyze'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./refactor_suggestions.json',
        help='Output file for refactoring plan'
    )
    
    parser.add_argument(
        '--query', '-q',
        help='Natural language query for intent analysis'
    )
    
    parser.add_argument(
        '--llm-targets',
        nargs='*',
        help='Specific targets for LLM analysis (e.g., module.Class module.function)'
    )
    
    parser.add_argument(
        '--suggestion-types',
        nargs='+',
        choices=['refactor', 'optimize', 'document'],
        default=['refactor'],
        help='Types of suggestions to generate'
    )
    
    parser.add_argument(
        '--model',
        default='codellama',
        help='Ollama model to use'
    )
    
    parser.add_argument(
        '--provider',
        choices=['ollama', 'litellm'],
        default='ollama',
        help='LLM provider to use'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Skip LLM-powered suggestions'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate target
    target_path = Path(args.target)
    if not target_path.exists():
        print(f"❌ Target does not exist: {args.target}")
        sys.exit(1)
    
    print(f"🚀 Analyzing target: {args.target}")
    
    # Analyze project
    if target_path.is_dir():
        analyzer = ProjectAnalyzer(str(target_path))
        project = analyzer.analyze()
    else:
        # For single files, create a minimal project
        print("⚠️  Single file analysis - creating minimal project structure")
        # This would need to be implemented for single file analysis
        print("❌ Single file analysis not yet implemented")
        sys.exit(1)
    
    if not project.modules:
        print("❌ No modules found in project")
        sys.exit(1)
    
    print(f"✅ Analyzed project: {project.name}")
    print(f"   Modules: {len(project.modules)}")
    print(f"   Functions: {sum(len(m.functions) for m in project.modules)}")
    print(f"   Classes: {sum(len(m.classes) for m in project.modules)}")
    
    # Analyze code smells
    code_smells = analyze_code_smells(project)
    
    # Generate intent-based suggestions
    intent_suggestions = {}
    if args.query:
        intent_suggestions = generate_intent_suggestions(project, args.query)
    
    # Generate LLM suggestions
    llm_suggestions = []
    if not args.no_llm:
        try:
            config = LLMConfig(
                provider=args.provider,
                model=args.model,
                temperature=0.7,
                max_tokens=2000
            )
            llm = LLMInterface(config)
            
            # Check LLM health
            health = llm.health_check()
            if not health['available']:
                print(f"⚠️  LLM service not available: {health.get('error', 'Unknown error')}")
                print("   Skipping LLM-powered suggestions")
            else:
                print("✅ LLM service is healthy")
                
                # Generate suggestions for specified targets or top complex items
                targets = args.llm_targets if args.llm_targets else []
                
                if not targets:
                    # Auto-select targets (most complex functions/classes)
                    all_functions = []
                    all_classes = []
                    
                    for module in project.modules:
                        for func in module.functions:
                            if func.complexity > 5:  # Threshold for complexity
                                all_functions.append((f"{module.name}.{func.name}", func.complexity))
                        
                        for cls in module.classes:
                            if len(cls.methods) > 5:  # Threshold for class size
                                all_classes.append((f"{module.name}.{cls.name}", len(cls.methods)))
                    
                    # Sort by complexity/size and take top 5
                    all_functions.sort(key=lambda x: x[1], reverse=True)
                    all_classes.sort(key=lambda x: x[1], reverse=True)
                    
                    targets = [f[0] for f in all_functions[:3]] + [c[0] for c in all_classes[:2]]
                
                # Generate suggestions for each target and suggestion type
                for target in targets:
                    for suggestion_type in args.suggestion_types:
                        suggestion = generate_llm_suggestions(llm, target, project, suggestion_type)
                        if "error" not in suggestion:
                            suggestion["suggestion_type"] = suggestion_type
                            llm_suggestions.append(suggestion)
                        else:
                            print(f"⚠️  {suggestion['error']}")
        
        except Exception as e:
            print(f"⚠️  Failed to initialize LLM: {e}")
            print("   Skipping LLM-powered suggestions")
    
    # Analyze dependencies
    dependency_analysis = analyze_dependencies(project)
    
    # Generate comprehensive refactoring plan
    refactoring_plan = generate_refactoring_plan(
        project,
        code_smells,
        intent_suggestions,
        llm_suggestions,
        dependency_analysis
    )
    
    # Save refactoring plan
    save_refactoring_plan(refactoring_plan, args.output)
    
    # Print summary
    print_refactoring_summary(refactoring_plan)
    
    print(f"\n✅ Refactoring analysis completed!")
    print(f"📄 Detailed plan available at: {args.output}")


if __name__ == '__main__':
    main()
