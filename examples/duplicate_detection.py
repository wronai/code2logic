#!/usr/bin/env python3
"""
Duplicate Detection Example.

This example demonstrates advanced duplicate code detection
using code2logic's similarity analysis capabilities.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic import ProjectAnalyzer
from code2logic.similarity import SimilarityDetector, SimilarityConfig
from code2logic.models import create_project, create_module, create_function, create_class


@dataclass
class DuplicateReport:
    """Report for duplicate code detection."""
    total_duplicates: int
    high_similarity_pairs: List[Dict[str, Any]]
    duplicate_clusters: List[List[str]]
    potential_refactorings: List[Dict[str, Any]]
    estimated_savings: int  # Lines of code that could be saved


class DuplicateDetector:
    """Advanced duplicate code detector."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize duplicate detector.
        
        Args:
            similarity_threshold: Threshold for considering code as duplicate
        """
        self.similarity_threshold = similarity_threshold
        self.config = SimilarityConfig(
            structural_threshold=similarity_threshold,
            semantic_threshold=similarity_threshold - 0.1,
            syntactic_threshold=similarity_threshold + 0.1,
            min_lines=5
        )
        self.detector = SimilarityDetector(self.config)
    
    def detect_duplicates(self, project) -> DuplicateReport:
        """
        Detect duplicates in the project.
        
        Args:
            project: Project to analyze
            
        Returns:
            Duplicate detection report
        """
        print(f"🔍 Detecting duplicates with threshold {self.similarity_threshold}...")
        
        # Find all similarities
        similarities = self.detector.detect_similarities(project.modules)
        
        # Filter for high similarity
        high_similarity = [
            sim for sim in similarities 
            if sim.score >= self.similarity_threshold
        ]
        
        # Find duplicate clusters
        clusters = self.detector.get_similarity_clusters(self.similarity_threshold)
        
        # Generate refactoring suggestions
        refactorings = self._generate_refactoring_suggestions(high_similarity, project)
        
        # Calculate potential savings
        savings = self._calculate_savings(high_similarity, project)
        
        return DuplicateReport(
            total_duplicates=len(high_similarity),
            high_similarity_pairs=[
                {
                    "item1": sim.item1,
                    "item2": sim.item2,
                    "score": sim.score,
                    "type": sim.similarity_type,
                    "details": sim.details
                }
                for sim in high_similarity
            ],
            duplicate_clusters=clusters,
            potential_refactorings=refactorings,
            estimated_savings=savings
        )
    
    def _generate_refactoring_suggestions(self, similarities: List, project) -> List[Dict[str, Any]]:
        """Generate refactoring suggestions for duplicates."""
        suggestions = []
        
        # Group by similarity type
        by_type = {}
        for sim in similarities:
            sim_type = sim.similarity_type
            if sim_type not in by_type:
                by_type[sim_type] = []
            by_type[sim_type].append(sim)
        
        # Generate suggestions for each type
        for sim_type, sims in by_type.items():
            if sim_type == "structural":
                suggestion = {
                    "type": "Extract Common Function",
                    "description": f"Extract common structure from {len(sims)} similar items",
                    "items": [f"{sim.item1} ↔ {sim.item2}" for sim in sims[:3]],
                    "effort": "medium",
                    "impact": "high"
                }
            elif sim_type == "semantic":
                suggestion = {
                    "type": "Create Shared Utility",
                    "description": f"Create shared utility for {len(sims)} semantically similar items",
                    "items": [f"{sim.item1} ↔ {sim.item2}" for sim in sims[:3]],
                    "effort": "low",
                    "impact": "medium"
                }
            elif sim_type == "syntactic":
                suggestion = {
                    "type": "Refactor Similar Patterns",
                    "description": f"Refactor {len(sims)} syntactically similar items",
                    "items": [f"{sim.item1} ↔ {sim.item2}" for sim in sims[:3]],
                    "effort": "medium",
                    "impact": "medium"
                }
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _calculate_savings(self, similarities: List, project) -> int:
        """Calculate potential lines of code savings."""
        savings = 0
        
        for sim in similarities:
            # Find the actual code objects
            item1_obj = self._find_code_object(sim.item1, project)
            item2_obj = self._find_code_object(sim.item2, project)
            
            if item1_obj and item2_obj:
                # Estimate savings as 70% of the smaller item
                size1 = getattr(item1_obj, 'lines_of_code', 0)
                size2 = getattr(item2_obj, 'lines_of_code', 0)
                smaller_size = min(size1, size2)
                savings += int(smaller_size * 0.7)
        
        return savings
    
    def _find_code_object(self, item_name: str, project):
        """Find code object by name."""
        parts = item_name.split('.')
        if len(parts) == 1:
            # Module
            return project.get_module_by_name(parts[0])
        elif len(parts) == 2:
            # Module.Function or Module.Class
            module = project.get_module_by_name(parts[0])
            if module:
                # Check functions
                for func in module.functions:
                    if func.name == parts[1]:
                        return func
                # Check classes
                for cls in module.classes:
                    if cls.name == parts[1]:
                        return cls
        return None
    
    def generate_duplicate_report(self, project, output_path: str) -> None:
        """
        Generate comprehensive duplicate detection report.
        
        Args:
            project: Project to analyze
            output_path: Path for output report
        """
        print("📊 Generating duplicate detection report...")
        
        # Detect duplicates
        report = self.detect_duplicates(project)
        
        # Create comprehensive report
        full_report = {
            "project": {
                "name": project.name,
                "modules": len(project.modules),
                "functions": sum(len(m.functions) for m in project.modules),
                "classes": sum(len(m.classes) for m in project.modules),
                "total_loc": sum(m.lines_of_code for m in project.modules)
            },
            "analysis": {
                "similarity_threshold": self.similarity_threshold,
                "total_duplicates": report.total_duplicates,
                "duplicate_clusters": len(report.duplicate_clusters),
                "estimated_savings": report.estimated_savings,
                "savings_percentage": (report.estimated_savings / max(1, sum(m.lines_of_code for m in project.modules))) * 100
            },
            "duplicates": report.high_similarity_pairs,
            "clusters": [
                {
                    "cluster_id": i,
                    "items": cluster,
                    "size": len(cluster)
                }
                for i, cluster in enumerate(report.duplicate_clusters)
            ],
            "refactoring_suggestions": report.potential_refactorings,
            "detailed_analysis": self._generate_detailed_analysis(report, project)
        }
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Report saved to: {output_path}")
        return full_report
    
    def _generate_detailed_analysis(self, report: DuplicateReport, project) -> Dict[str, Any]:
        """Generate detailed analysis of duplicates."""
        analysis = {
            "duplicate_types": {},
            "module_analysis": {},
            "complexity_analysis": {},
            "recommendations": []
        }
        
        # Analyze by type
        for duplicate in report.high_similarity_pairs:
            dup_type = duplicate["type"]
            if dup_type not in analysis["duplicate_types"]:
                analysis["duplicate_types"][dup_type] = 0
            analysis["duplicate_types"][dup_type] += 1
        
        # Analyze by module
        module_counts = {}
        for duplicate in report.high_similarity_pairs:
            for item in [duplicate["item1"], duplicate["item2"]]:
                module = item.split('.')[0]
                if module not in module_counts:
                    module_counts[module] = 0
                module_counts[module] += 1
        
        analysis["module_analysis"] = module_counts
        
        # Complexity analysis
        high_complexity_duplicates = []
        for duplicate in report.high_similarity_pairs:
            if duplicate.get("details", {}).get("complexity", 0) > 5:
                high_complexity_duplicates.append(duplicate)
        
        analysis["complexity_analysis"] = {
            "high_complexity_duplicates": len(high_complexity_duplicates),
            "total_duplicates": len(report.high_similarity_pairs),
            "high_complexity_percentage": (len(high_complexity_duplicates) / max(1, len(report.high_similarity_pairs))) * 100
        }
        
        return analysis


def create_project_with_duplicates():
    """Create a sample project with intentional duplicates."""
    print("🏗️  Creating project with duplicates...")
    
    project = create_project(
        name="duplicate_demo",
        path="/tmp/duplicate_demo"
    )
    
    # Module 1: Original functions
    module1 = create_module(
        name="utils",
        path="/tmp/duplicate_demo/utils.py",
        functions=[
            create_function(
                name="calculate_sum",
                parameters=["numbers"],
                lines_of_code=8,
                complexity=2,
                docstring="Calculate sum of numbers",
                code="def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total"
            ),
            create_function(
                name="validate_email",
                parameters=["email"],
                lines_of_code=6,
                complexity=2,
                docstring="Validate email address",
                code="def validate_email(email):\n        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n        return re.match(pattern, email) is not None"
            ),
            create_function(
                name="process_data",
                parameters=["data", "options"],
                lines_of_code=12,
                complexity=4,
                docstring="Process data with options",
                code="def process_data(data, options):\n    if options.get('validate'):\n        return validate_data(data)\n    if options.get('transform'):\n        return transform_data(data)\n    return data"
            )
        ],
        imports=["re", "json"],
        lines_of_code=30
    )
    
    # Module 2: Similar/duplicate functions
    module2 = create_module(
        name="helpers",
        path="/tmp/duplicate_demo/helpers.py",
        functions=[
            create_function(
                name="compute_sum",
                parameters=["values"],
                lines_of_code=8,
                complexity=2,
                docstring="Compute sum of values",
                code="def compute_sum(values):\n    total = 0\n    for val in values:\n        total += val\n    return total"
            ),
            create_function(
                name="check_email",
                parameters=["email_address"],
                lines_of_code=6,
                complexity=2,
                docstring="Check email address format",
                code="def check_email(email_address):\n        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n        return re.match(pattern, email_address) is not None"
            ),
            create_function(
                name="handle_data",
                parameters=["input_data", "config"],
                lines_of_code=12,
                complexity=4,
                docstring="Handle input data with configuration",
                code="def handle_data(input_data, config):\n    if config.get('validate'):\n        return validate_input(input_data)\n    if config.get('transform'):\n        return transform_input(input_data)\n    return input_data"
            )
        ],
        imports=["re", "json"],
        lines_of_code=30
    )
    
    # Module 3: Partial duplicates
    module3 = create_module(
        name="processors",
        path="/tmp/duplicate_demo/processors.py",
        functions=[
            create_function(
                name="calculate_total",
                parameters=["items"],
                lines_of_code=6,
                complexity=1,
                docstring="Calculate total of items",
                code="def calculate_total(items):\n    return sum(items)"
            ),
            create_function(
                name="is_valid_email",
                parameters=["email"],
                lines_of_code=4,
                complexity=1,
                docstring="Check if email is valid",
                code="def is_valid_email(email):\n    return '@' in email and '.' in email"
            ),
            create_function(
                name="format_data",
                parameters=["raw_data"],
                lines_of_code=8,
                complexity=2,
                docstring="Format raw data",
                code="def format_data(raw_data):\n    if isinstance(raw_data, str):\n        return raw_data.strip()\n    return raw_data"
            )
        ],
        imports=["typing"],
        lines_of_code=20
    )
    
    # Module 4: Complex duplicate
    module4 = create_module(
        name="analytics",
        path="/tmp/duplicate_demo/analytics.py",
        functions=[
            create_function(
                name="process_user_data",
                parameters=["user_data", "options"],
                lines_of_code=15,
                complexity=5,
                docstring="Process user data with options",
                code="def process_user_data(user_data, options):\n    if options.get('validate'):\n        return validate_user(user_data)\n    if options.get('transform'):\n        return transform_user(user_data)\n    if options.get('enrich'):\n        return enrich_user(user_data)\n    return user_data"
            )
        ],
        imports=["datetime", "json"],
        lines_of_code=20
    )
    
    project.modules.extend([module1, module2, module3, module4])
    
    print(f"✅ Created project with {len(project.modules)} modules and intentional duplicates")
    return project


def main():
    """Main duplicate detection example."""
    print("🔍 Duplicate Detection Example")
    print("=" * 50)
    print("This example demonstrates advanced duplicate code detection")
    print("using code2logic's similarity analysis capabilities.")
    print()
    
    try:
        # Create project with duplicates
        project = create_project_with_duplicates()
        
        # Initialize detector
        detector = DuplicateDetector(similarity_threshold=0.7)
        
        # Generate report
        output_dir = Path("./duplicate_detection_output")
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / "duplicate_report.json"
        full_report = detector.generate_duplicate_report(project, str(report_path))
        
        # Display results
        print(f"\n📊 Duplicate Detection Results:")
        print(f"   Project: {full_report['project']['name']}")
        print(f"   Modules: {full_report['project']['modules']}")
        print(f"   Functions: {full_report['project']['functions']}")
        print(f"   Total LOC: {full_report['project']['total_loc']}")
        print(f"   Similarity threshold: {full_report['analysis']['similarity_threshold']}")
        print(f"   Duplicates found: {full_report['analysis']['total_duplicates']}")
        print(f"   Duplicate clusters: {full_report['analysis']['duplicate_clusters']}")
        print(f"   Estimated savings: {full_report['analysis']['estimated_savings']} LOC")
        print(f"   Savings percentage: {full_report['analysis']['savings_percentage']:.1f}%")
        
        # Show duplicate types
        print(f"\n📈 Duplicate Types:")
        for dup_type, count in full_report['analysis']['duplicate_types'].items():
            print(f"   {dup_type}: {count}")
        
        # Show module analysis
        print(f"\n📁 Modules with Most Duplicates:")
        module_analysis = full_report['analysis']['module_analysis']
        sorted_modules = sorted(module_analysis.items(), key=lambda x: x[1], reverse=True)
        for module, count in sorted_modules[:5]:
            print(f"   {module}: {count} duplicates")
        
        # Show refactoring suggestions
        print(f"\n🔧 Refactoring Suggestions:")
        for i, suggestion in enumerate(full_report['refactoring_suggestions'], 1):
            print(f"   {i}. {suggestion['type']}")
            print(f"      {suggestion['description']}")
            print(f"      Effort: {suggestion['effort']}, Impact: {suggestion['impact']}")
        
        # Show duplicate clusters
        if full_report['clusters']:
            print(f"\n🔗 Duplicate Clusters:")
            for cluster in full_report['clusters'][:3]:
                print(f"   Cluster {cluster['cluster_id']} ({cluster['size']} items):")
                for item in cluster['items'][:3]:
                    print(f"     - {item}")
                if len(cluster['items']) > 3:
                    print(f"     ... and {len(cluster['items']) - 3} more")
        
        # Show complexity analysis
        complexity = full_report['analysis']['complexity_analysis']
        print(f"\n🧠 Complexity Analysis:")
        print(f"   High complexity duplicates: {complexity['high_complexity_duplicates']}")
        print(f"   High complexity percentage: {complexity['high_complexity_percentage']:.1f}%")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
