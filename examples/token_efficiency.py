#!/usr/bin/env python3
"""
Token Efficiency Analysis Example.

This example demonstrates how to analyze and optimize token usage
when working with LLMs in code2logic, helping to reduce costs
and improve performance.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic import ProjectAnalyzer
from code2logic.models import create_project, create_module, create_function, create_class


@dataclass
class TokenMetrics:
    """Metrics for token usage analysis."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    processing_time: float
    efficiency_score: float


class TokenAnalyzer:
    """Analyzes and optimizes token usage for LLM operations."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize token analyzer.
        
        Args:
            model_name: Name of the LLM model
        """
        self.model_name = model_name
        self.token_costs = self._get_token_costs()
        self.metrics_history: List[TokenMetrics] = []
    
    def _get_token_costs(self) -> Dict[str, Dict[str, float]]:
        """Get token costs for different models (USD per 1K tokens)."""
        return {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "codellama": {"input": 0.0001, "output": 0.0001}  # Local model
        }
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Estimated token count
        """
        # Simple token estimation (rough approximation)
        # In practice, you'd use the actual tokenizer
        words = text.split()
        # Approximate: 1 token ≈ 4 characters or 0.75 words
        char_based = len(text) // 4
        word_based = len(words) // 0.75
        return max(char_based, word_based)
    
    def analyze_project_tokens(self, project) -> Dict[str, Any]:
        """
        Analyze token usage for project analysis.
        
        Args:
            project: Project to analyze
            
        Returns:
            Token analysis results
        """
        print("🔍 Analyzing project token usage...")
        
        # Serialize project to JSON
        project_json = self._serialize_project(project)
        
        # Estimate tokens
        input_tokens = self.estimate_tokens(project_json)
        
        # Estimate output tokens (typical analysis response)
        output_tokens = input_tokens // 3  # Rough estimate
        
        # Calculate cost
        costs = self.token_costs.get(self.model_name, {"input": 0.001, "output": 0.002})
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "model": self.model_name
        }
    
    def _serialize_project(self, project) -> str:
        """Serialize project to JSON string for token estimation."""
        project_data = {
            "name": project.name,
            "path": project.path,
            "modules": []
        }
        
        for module in project.modules:
            module_data = {
                "name": module.name,
                "path": module.path,
                "lines_of_code": module.lines_of_code,
                "imports": module.imports,
                "functions": [
                    {
                        "name": func.name,
                        "parameters": func.parameters,
                        "lines_of_code": func.lines_of_code,
                        "complexity": func.complexity
                    }
                    for func in module.functions
                ],
                "classes": [
                    {
                        "name": cls.name,
                        "methods": len(cls.methods),
                        "lines_of_code": cls.lines_of_code
                    }
                    for cls in module.classes
                ]
            }
            project_data["modules"].append(module_data)
        
        return json.dumps(project_data, indent=2)
    
    def optimize_project_representation(self, project) -> Dict[str, Any]:
        """
        Generate optimized project representation for minimal token usage.
        
        Args:
            project: Project to optimize
            
        Returns:
            Optimized representation and token savings
        """
        print("⚡ Optimizing project representation...")
        
        # Full representation
        full_analysis = self.analyze_project_tokens(project)
        
        # Optimized representation (summary only)
        optimized_data = {
            "project_summary": {
                "name": project.name,
                "modules": len(project.modules),
                "total_functions": sum(len(m.functions) for m in project.modules),
                "total_classes": sum(len(m.classes) for m in project.modules),
                "total_loc": sum(m.lines_of_code for m in project.modules)
            },
            "complex_modules": [
                {
                    "name": m.name,
                    "functions": len(m.functions),
                    "classes": len(m.classes),
                    "complexity": sum(f.complexity for f in m.functions)
                }
                for m in project.modules if sum(f.complexity for f in m.functions) > 10
            ],
            "key_functions": [
                {
                    "module": m.name,
                    "name": f.name,
                    "complexity": f.complexity,
                    "params": len(f.parameters)
                }
                for m in project.modules 
                for f in m.functions 
                if f.complexity > 5 or len(f.parameters) > 3
            ]
        }
        
        optimized_json = json.dumps(optimized_data, indent=2)
        optimized_tokens = self.estimate_tokens(optimized_json)
        
        # Calculate savings
        token_savings = full_analysis["input_tokens"] - optimized_tokens
        cost_savings = (token_savings / 1000) * self.token_costs[self.model_name]["input"]
        
        return {
            "optimized_tokens": optimized_tokens,
            "full_tokens": full_analysis["input_tokens"],
            "token_savings": token_savings,
            "savings_percentage": (token_savings / full_analysis["input_tokens"]) * 100,
            "cost_savings": cost_savings,
            "optimized_data": optimized_data
        }
    
    def generate_token_efficiency_report(self, project) -> Dict[str, Any]:
        """
        Generate comprehensive token efficiency report.
        
        Args:
            project: Project to analyze
            
        Returns:
            Token efficiency report
        """
        print("📊 Generating token efficiency report...")
        
        # Analyze full project
        full_analysis = self.analyze_project_tokens(project)
        
        # Analyze optimized version
        optimization = self.optimize_project_representation(project)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(project, full_analysis, optimization)
        
        report = {
            "project": project.name,
            "model": self.model_name,
            "full_analysis": full_analysis,
            "optimization": optimization,
            "recommendations": recommendations,
            "summary": {
                "potential_savings": f"{optimization['savings_percentage']:.1f}%",
                "cost_reduction": f"${optimization['cost_savings']:.4f}",
                "efficiency_score": self._calculate_efficiency_score(full_analysis, optimization)
            }
        }
        
        return report
    
    def _generate_recommendations(self, project, full_analysis, optimization) -> List[str]:
        """Generate token optimization recommendations."""
        recommendations = []
        
        # Size-based recommendations
        if full_analysis["input_tokens"] > 10000:
            recommendations.append("Consider splitting large projects into smaller analyses")
        
        if optimization["savings_percentage"] > 50:
            recommendations.append("Use optimized representation for large projects")
        
        # Module-based recommendations
        complex_modules = [m for m in project.modules if sum(f.complexity for f in m.functions) > 20]
        if len(complex_modules) > 0:
            recommendations.append(f"Focus analysis on {len(complex_modules)} complex modules")
        
        # Function-based recommendations
        all_functions = [f for m in project.modules for f in m.functions]
        complex_functions = [f for f in all_functions if f.complexity > 10]
        if len(complex_functions) > 0:
            recommendations.append(f"Prioritize {len(complex_functions)} complex functions for detailed analysis")
        
        # Cost-based recommendations
        if full_analysis["total_cost"] > 0.10:  # $0.10 threshold
            recommendations.append("Consider using local models (Ollama) for cost efficiency")
        
        # Model-based recommendations
        if self.model_name == "gpt-4":
            recommendations.append("Consider using GPT-3.5-turbo for cost-sensitive operations")
        elif self.model_name == "gpt-3.5-turbo":
            recommendations.append("Consider using local models for high-volume analysis")
        
        return recommendations
    
    def _calculate_efficiency_score(self, full_analysis, optimization) -> float:
        """Calculate efficiency score (0-100)."""
        # Base score on optimization potential
        savings_score = min(optimization["savings_percentage"], 100)
        
        # Penalize very high token usage
        if full_analysis["input_tokens"] > 50000:
            savings_score *= 0.8
        
        # Bonus for low cost
        if full_analysis["total_cost"] < 0.01:
            savings_score = min(savings_score * 1.2, 100)
        
        return round(savings_score, 1)
    
    def compare_models(self, project) -> Dict[str, Any]:
        """Compare token efficiency across different models."""
        print("🔄 Comparing token efficiency across models...")
        
        models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "claude-3-sonnet", "codellama"]
        comparison = {}
        
        # Get base token count
        base_tokens = self.estimate_tokens(self._serialize_project(project))
        
        for model in models:
            costs = self.token_costs.get(model, {"input": 0.001, "output": 0.002})
            
            # Estimate output tokens
            output_tokens = base_tokens // 3
            
            # Calculate costs
            input_cost = (base_tokens / 1000) * costs["input"]
            output_cost = (output_tokens / 1000) * costs["output"]
            total_cost = input_cost + output_cost
            
            comparison[model] = {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "relative_cost": total_cost / comparison.get("gpt-3.5-turbo", {}).get("total_cost", 1)
            }
        
        return comparison


def create_sample_project():
    """Create a sample project for token analysis."""
    print("🏗️  Creating sample project for token analysis...")
    
    project = create_project(
        name="token_analysis_demo",
        path="/tmp/token_analysis_demo"
    )
    
    # Add modules with varying complexity
    modules_data = [
        {
            "name": "simple_module",
            "functions": 5,
            "complexity": 2,
            "classes": 1
        },
        {
            "name": "complex_module", 
            "functions": 15,
            "complexity": 8,
            "classes": 5
        },
        {
            "name": "very_complex_module",
            "functions": 25,
            "complexity": 15,
            "classes": 10
        }
    ]
    
    for module_data in modules_data:
        functions = []
        for i in range(module_data["functions"]):
            func = create_function(
                name=f"function_{i}",
                parameters=[f"param_{j}" for j in range(min(3, i % 5))],
                lines_of_code=10 + i * 2,
                complexity=min(module_data["complexity"], i + 1),
                docstring=f"Function {i} documentation"
            )
            functions.append(func)
        
        classes = []
        for i in range(module_data["classes"]):
            cls = create_class(
                name=f"Class{i}",
                methods=[
                    create_function(
                        name=f"method_{j}",
                        parameters=[],
                        lines_of_code=5,
                        complexity=2
                    )
                    for j in range(min(5, i + 2))
                ],
                lines_of_code=20 + i * 5
            )
            classes.append(cls)
        
        module = create_module(
            name=module_data["name"],
            path=f"/tmp/token_analysis_demo/{module_data['name']}.py",
            functions=functions,
            classes=classes,
            imports=[f"module_{j}" for j in range(min(5, module_data["functions"]))],
            lines_of_code=sum(f.lines_of_code for f in functions) + sum(c.lines_of_code for c in classes)
        )
        
        project.modules.append(module)
    
    print(f"✅ Created project with {len(project.modules)} modules")
    return project


def main():
    """Main token efficiency analysis."""
    print("💰 Token Efficiency Analysis")
    print("=" * 50)
    print("This example demonstrates token usage optimization")
    print("for LLM operations in code2logic.")
    print()
    
    try:
        # Create sample project
        project = create_sample_project()
        
        # Initialize analyzer
        analyzer = TokenAnalyzer("gpt-3.5-turbo")
        
        # Generate efficiency report
        report = analyzer.generate_token_efficiency_report(project)
        
        # Display results
        print(f"\n📊 Token Analysis Results:")
        print(f"   Project: {report['project']}")
        print(f"   Model: {report['model']}")
        print(f"   Input tokens: {report['full_analysis']['input_tokens']:,}")
        print(f"   Output tokens: {report['full_analysis']['output_tokens']:,}")
        print(f"   Total cost: ${report['full_analysis']['total_cost']:.4f}")
        print(f"   Potential savings: {report['summary']['potential_savings']}")
        print(f"   Cost reduction: {report['summary']['cost_reduction']}")
        print(f"   Efficiency score: {report['summary']['efficiency_score']}/100")
        
        # Show recommendations
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        # Compare models
        print(f"\n🔄 Model Comparison:")
        comparison = analyzer.compare_models(project)
        for model, data in comparison.items():
            print(f"   {model}: ${data['total_cost']:.4f} (x{data['relative_cost']:.1f})")
        
        # Save report
        output_dir = Path("./token_analysis_output")
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / "token_efficiency_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Report saved to: {report_path}")
        
        # Show optimized data
        print(f"\n⚡ Optimized Representation:")
        optimized = report['optimization']
        print(f"   Original tokens: {optimized['full_tokens']:,}")
        print(f"   Optimized tokens: {optimized['optimized_tokens']:,}")
        print(f"   Token savings: {optimized['token_savings']:,} ({optimized['savings_percentage']:.1f}%)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
