"""
Gherkin and BDD support for code2logic.

This module provides functionality to generate Gherkin feature files
and Cucumber-style YAML configurations from analyzed code projects.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from .models import Project, Module, Function, Class


@dataclass
class GherkinFeature:
    """Represents a Gherkin feature."""
    name: str
    description: str
    scenarios: List['GherkinScenario']
    background: Optional[str] = None


@dataclass
class GherkinScenario:
    """Represents a Gherkin scenario."""
    name: str
    steps: List[str]
    tags: List[str] = None


@dataclass
class GherkinStep:
    """Represents a Gherkin step."""
    keyword: str  # Given, When, Then, And, But
    text: str
    docstring: Optional[str] = None
    table: Optional[List[Dict[str, str]]] = None


class GherkinGenerator:
    """Generates Gherkin feature files from code analysis."""
    
    def __init__(self):
        """Initialize the Gherkin generator."""
        self.features: List[GherkinFeature] = []
    
    def generate_features(self, project: Project) -> List[GherkinFeature]:
        """
        Generate Gherkin features from project analysis.
        
        Args:
            project: Analyzed project
            
        Returns:
            List of Gherkin features
        """
        self.features = []
        
        # Generate feature for each module
        for module in project.modules:
            feature = self._generate_module_feature(module)
            if feature:
                self.features.append(feature)
        
        # Generate integration features
        integration_feature = self._generate_integration_feature(project)
        if integration_feature:
            self.features.append(integration_feature)
        
        # Generate performance feature
        performance_feature = self._generate_performance_feature(project)
        if performance_feature:
            self.features.append(performance_feature)
        
        return self.features
    
    def _generate_module_feature(self, module: Module) -> Optional[GherkinFeature]:
        """Generate a feature for a specific module."""
        if not module.functions and not module.classes:
            return None
        
        feature_name = f"{module.name.replace('_', ' ').title()} Module"
        description = f"Behavior and functionality of the {module.name} module"
        
        scenarios = []
        
        # Generate scenarios for functions
        for func in module.functions:
            scenario = self._generate_function_scenario(module, func)
            scenarios.append(scenario)
        
        # Generate scenarios for classes
        for cls in module.classes:
            scenario = self._generate_class_scenario(module, cls)
            scenarios.append(scenario)
        
        return GherkinFeature(
            name=feature_name,
            description=description,
            scenarios=scenarios
        )
    
    def _generate_function_scenario(self, module: Module, function: Function) -> GherkinScenario:
        """Generate a scenario for a function."""
        scenario_name = f"Test {function.name} functionality"
        
        steps = [
            f"Given the {module.name} module is imported",
            f"When I call {function.name} with valid parameters",
            f"Then the function should execute successfully"
        ]
        
        # Add parameter-specific steps
        if function.parameters:
            param_list = ", ".join(function.parameters)
            steps.insert(1, f"And I provide parameters: {param_list}")
        
        # Add return value steps
        if function.return_type:
            steps.append(f"And the return type should be {function.return_type}")
        
        # Add complexity-based steps
        if function.complexity > 5:
            steps.append(f"And the function complexity is {function.complexity}")
        
        return GherkinScenario(
            name=scenario_name,
            steps=steps,
            tags=["function", "unit-test"]
        )
    
    def _generate_class_scenario(self, module: Module, cls: Class) -> GherkinScenario:
        """Generate a scenario for a class."""
        scenario_name = f"Test {cls.name} class functionality"
        
        steps = [
            f"Given the {module.name} module is imported",
            f"When I create an instance of {cls.name}",
            f"Then the object should be created successfully"
        ]
        
        # Add inheritance steps
        if cls.base_classes:
            base_list = ", ".join(cls.base_classes)
            steps.append(f"And the class inherits from: {base_list}")
        
        # Add method steps
        if cls.methods:
            method_names = [method.name for method in cls.methods[:3]]  # Limit to first 3
            steps.append(f"And the class has methods: {', '.join(method_names)}")
        
        return GherkinScenario(
            name=scenario_name,
            steps=steps,
            tags=["class", "unit-test"]
        )
    
    def _generate_integration_feature(self, project: Project) -> Optional[GherkinFeature]:
        """Generate integration test feature."""
        if len(project.modules) < 2:
            return None
        
        feature_name = "Module Integration"
        description = "Integration tests between different modules"
        
        scenarios = []
        
        # Generate dependency scenarios
        for dep in project.dependencies:
            scenario = GherkinScenario(
                name=f"Test dependency between {dep.source} and {dep.target}",
                steps=[
                    f"Given the {dep.source} module",
                    f"When it imports from {dep.target}",
                    f"Then the dependency should be resolved",
                    f"And the dependency strength is {dep.strength:.2f}"
                ],
                tags=["integration", "dependency"]
            )
            scenarios.append(scenario)
        
        return GherkinFeature(
            name=feature_name,
            description=description,
            scenarios=scenarios
        )
    
    def _generate_performance_feature(self, project: Project) -> Optional[GherkinFeature]:
        """Generate performance testing feature."""
        total_loc = sum(m.lines_of_code for m in project.modules)
        if total_loc < 100:  # Skip for very small projects
            return None
        
        feature_name = "Performance Testing"
        description = "Performance and scalability tests"
        
        scenarios = []
        
        # Generate complexity scenarios
        complex_functions = []
        for module in project.modules:
            for func in module.functions:
                if func.complexity > 10:
                    complex_functions.append((module.name, func.name, func.complexity))
        
        if complex_functions:
            scenario = GherkinScenario(
                name="Test complex function performance",
                steps=[
                    f"Given there are {len(complex_functions)} complex functions",
                    f"When I execute the most complex function",
                    f"Then the execution time should be acceptable",
                    f"And memory usage should be within limits"
                ],
                tags=["performance", "complexity"]
            )
            scenarios.append(scenario)
        
        return GherkinFeature(
            name=feature_name,
            description=description,
            scenarios=scenarios
        )
    
    def write_feature_files(self, output_dir: str) -> None:
        """
        Write Gherkin feature files to directory.
        
        Args:
            output_dir: Directory to write feature files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for feature in self.features:
            filename = self._sanitize_filename(feature.name) + ".feature"
            filepath = output_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self._format_feature(feature))
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize filename for feature file."""
        return name.lower().replace(' ', '_').replace('-', '_').replace('.', '_')
    
    def _format_feature(self, feature: GherkinFeature) -> str:
        """Format feature as Gherkin text."""
        lines = []
        
        # Feature header
        lines.append(f"Feature: {feature.name}")
        lines.append("")
        
        # Description
        if feature.description:
            lines.append(f"  {feature.description}")
            lines.append("")
        
        # Background
        if feature.background:
            lines.append("  Background:")
            lines.append(f"    {feature.background}")
            lines.append("")
        
        # Scenarios
        for scenario in feature.scenarios:
            lines.append(f"  Scenario: {scenario.name}")
            
            # Tags
            if scenario.tags:
                lines.append(f"    @{', @'.join(scenario.tags)}")
            
            # Steps
            for step in scenario.steps:
                lines.append(f"    {step}")
            
            lines.append("")
        
        return "\n".join(lines)


class CucumberYAMLGenerator:
    """Generates Cucumber-style YAML configurations."""
    
    def __init__(self):
        """Initialize the YAML generator."""
        pass
    
    def generate_cucumber_config(self, project: Project) -> Dict[str, Any]:
        """
        Generate Cucumber configuration YAML.
        
        Args:
            project: Analyzed project
            
        Returns:
            Cucumber configuration dictionary
        """
        config = {
            "default": "default",
            "step_definitions": self._generate_step_definitions(project),
            "features": self._generate_feature_paths(project),
            "glue": self._generate_glue_paths(project),
            "tags": self._generate_tag_config(project),
            "format": self._generate_format_config(),
            "publish": self._generate_publish_config(),
            "dry_run": False,
            "strict": True,
            "monochrome": False,
            "wip": False,
            "quiet": False,
            "verbose": False,
            "expand": False,
            "snippets": True,
            "source": True,
            "snippet_syntax": "underscore"
        }
        
        return config
    
    def _generate_step_definitions(self, project: Project) -> List[str]:
        """Generate step definition paths."""
        step_defs = []
        
        for module in project.modules:
            # Add step definitions for each module
            step_def_path = f"step_definitions/{module.name}_steps.rb"
            step_defs.append(step_def_path)
        
        # Add common step definitions
        step_defs.extend([
            "step_definitions/common_steps.rb",
            "step_definitions/api_steps.rb",
            "step_definitions/ui_steps.rb"
        ])
        
        return step_defs
    
    def _generate_feature_paths(self, project: Project) -> List[str]:
        """Generate feature file paths."""
        feature_paths = []
        
        for module in project.modules:
            feature_path = f"features/{module.name}.feature"
            feature_paths.append(feature_path)
        
        # Add common feature paths
        feature_paths.extend([
            "features/integration/",
            "features/performance/",
            "features/security/"
        ])
        
        return feature_paths
    
    def _generate_glue_paths(self, project: Project) -> List[str]:
        """Generate glue code paths."""
        glue_paths = []
        
        for module in project.modules:
            glue_path = f"glue/{module.name}_glue.rb"
            glue_paths.append(glue_path)
        
        return glue_paths
    
    def _generate_tag_config(self, project: Project) -> Dict[str, Any]:
        """Generate tag configuration."""
        tag_config = {
            "include": [
                "@complete",
                "@ready",
                "@integration"
            ],
            "exclude": [
                "@wip",
                "@skip",
                "@manual"
            ]
        }
        
        # Add module-specific tags
        for module in project.modules:
            tag_config["include"].append(f"@{module.name}")
        
        return tag_config
    
    def _generate_format_config(self) -> List[str]:
        """Generate format configuration."""
        return [
            "pretty",
            "json:reports/cucumber_report.json",
            "junit:reports/cucumber_report.xml",
            "html:reports/cucumber_report.html",
            "rerun:reports/rerun.txt"
        ]
    
    def _generate_publish_config(self) -> Dict[str, Any]:
        """Generate publish configuration."""
        return {
            "enabled": False,
            "host": "localhost",
            "port": 1985,
            "project": "code2logic-tests"
        }
    
    def write_cucumber_config(self, output_path: str, config: Dict[str, Any]) -> None:
        """
        Write Cucumber configuration to YAML file.
        
        Args:
            output_path: Path to write configuration
            config: Configuration dictionary
        """
        import yaml
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def generate_step_definitions(self, project: Project) -> Dict[str, str]:
        """
        Generate step definition code snippets.
        
        Args:
            project: Analyzed project
            
        Returns:
            Dictionary mapping step types to code snippets
        """
        step_definitions = {
            "given_steps": self._generate_given_steps(project),
            "when_steps": self._generate_when_steps(project),
            "then_steps": self._generate_then_steps(project)
        }
        
        return step_definitions
    
    def _generate_given_steps(self, project: Project) -> str:
        """Generate Given step definitions."""
        steps = []
        
        for module in project.modules:
            steps.append(f"""Given(/^the {module.name} module is imported$/) do
  # Module import logic
  require_relative '../../{module.path}'
end""")
        
        return "\n\n".join(steps)
    
    def _generate_when_steps(self, project: Project) -> str:
        """Generate When step definitions."""
        steps = []
        
        for module in project.modules:
            for func in module.functions:
                steps.append(f"""When(/^I call {func.name} with valid parameters$/) do
  # Function call logic
  result = {module.name}::{func.name}(params)
  @result = result
end""")
        
        return "\n\n".join(steps)
    
    def _generate_then_steps(self, project: Project) -> str:
        """Generate Then step definitions."""
        steps = []
        
        steps.append("""Then(/^the function should execute successfully$/) do
  expect(@result).not_to be_nil
end

Then(/^the return type should be (.+)$/) do |expected_type|
  expect(@result.class.name).to eq(expected_type)
end

Then(/^the execution time should be acceptable$/) do
  expect(@execution_time).to be < 1.0  # 1 second threshold
end""")
        
        return "\n\n".join(steps)


def generate_gherkin_from_project(project: Project, output_dir: str) -> None:
    """
    Generate complete Gherkin and Cucumber configuration from project.
    
    Args:
        project: Analyzed project
        output_dir: Output directory for generated files
    """
    # Generate Gherkin features
    gherkin_gen = GherkinGenerator()
    features = gherkin_gen.generate_features(project)
    gherkin_gen.write_feature_files(output_dir)
    
    # Generate Cucumber configuration
    yaml_gen = CucumberYAMLGenerator()
    config = yaml_gen.generate_cucumber_config(project)
    config_path = Path(output_dir) / "cucumber.yml"
    yaml_gen.write_cucumber_config(str(config_path), config)
    
    # Generate step definitions
    step_defs = yaml_gen.generate_step_definitions(project)
    steps_dir = Path(output_dir) / "step_definitions"
    steps_dir.mkdir(parents=True, exist_ok=True)
    
    for step_type, code in step_defs.items():
        step_file = steps_dir / f"{step_type}.rb"
        with open(step_file, 'w', encoding='utf-8') as f:
            f.write(code)
    
    print(f"Generated {len(features)} Gherkin features and Cucumber configuration")
    print(f"Output directory: {output_dir}")
