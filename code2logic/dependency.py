"""
Dependency graph analysis using NetworkX.

This module provides functionality to analyze and visualize
dependency relationships between modules, classes, and functions.
"""

import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

from .models import Module, Function, Class, Dependency


@dataclass
class DependencyMetrics:
    """Metrics for dependency analysis."""
    total_dependencies: int
    incoming_dependencies: int
    outgoing_dependencies: int
    circular_dependencies: int
    dependency_depth: int
    fan_in: int
    fan_out: int


class DependencyAnalyzer:
    """Analyzes dependency graphs using NetworkX."""
    
    def __init__(self):
        """Initialize the dependency analyzer."""
        self.graph: Optional[nx.DiGraph] = None
        self.metrics: Dict[str, DependencyMetrics] = {}
    
    def analyze_dependencies(self, modules: List[Module]) -> List[Dependency]:
        """
        Analyze dependencies between modules.
        
        Args:
            modules: List of modules to analyze
            
        Returns:
            List of Dependency objects
        """
        # Create directed graph
        self.graph = nx.DiGraph()
        
        # Add nodes for modules, classes, and functions
        for module in modules:
            self.graph.add_node(module.name, type='module', obj=module)
            
            for cls in module.classes:
                self.graph.add_node(
                    f"{module.name}.{cls.name}", 
                    type='class', 
                    obj=cls
                )
            
            for func in module.functions:
                self.graph.add_node(
                    f"{module.name}.{func.name}",
                    type='function', 
                    obj=func
                )
        
        # Extract dependencies from imports and function calls
        dependencies = []
        
        for module in modules:
            # Module-level dependencies
            for import_name in module.imports:
                dep = self._create_dependency(module.name, import_name, 'import')
                if dep:
                    dependencies.append(dep)
                    self.graph.add_edge(dep.source, dep.target, type='import')
            
            # Class dependencies
            for cls in module.classes:
                for base_class in cls.base_classes:
                    dep = self._create_dependency(
                        f"{module.name}.{cls.name}", 
                        base_class, 
                        'inheritance'
                    )
                    if dep:
                        dependencies.append(dep)
                        self.graph.add_edge(dep.source, dep.target, type='inheritance')
                
                for method in cls.methods:
                    method_deps = self._extract_function_dependencies(
                        f"{module.name}.{cls.name}.{method.name}",
                        method.code
                    )
                    dependencies.extend(method_deps)
            
            # Function dependencies
            for func in module.functions:
                func_deps = self._extract_function_dependencies(
                    f"{module.name}.{func.name}",
                    func.code
                )
                dependencies.extend(func_deps)
        
        # Calculate metrics
        self._calculate_metrics()
        
        return dependencies
    
    def _create_dependency(
        self, 
        source: str, 
        target: str, 
        dep_type: str
    ) -> Optional[Dependency]:
        """Create a Dependency object if target exists."""
        if target in self.graph.nodes:
            return Dependency(
                source=source,
                target=target,
                type=dep_type,
                strength=self._calculate_strength(source, target, dep_type)
            )
        return None
    
    def _extract_function_dependencies(
        self, 
        func_name: str, 
        code: str
    ) -> List[Dependency]:
        """Extract dependencies from function code."""
        dependencies = []
        
        # Simple regex-based extraction (can be enhanced with AST)
        import re
        
        # Find function calls
        function_calls = re.findall(r'(\w+)\(', code)
        for call in function_calls:
            if call in self.graph.nodes:
                dep = Dependency(
                    source=func_name,
                    target=call,
                    type='function_call',
                    strength=self._calculate_strength(func_name, call, 'function_call')
                )
                dependencies.append(dep)
                self.graph.add_edge(func_name, call, type='function_call')
        
        return dependencies
    
    def _calculate_strength(
        self, 
        source: str, 
        target: str, 
        dep_type: str
    ) -> float:
        """Calculate dependency strength based on type and context."""
        strength_map = {
            'import': 0.8,
            'inheritance': 0.9,
            'function_call': 0.6,
            'attribute_access': 0.4,
        }
        return strength_map.get(dep_type, 0.5)
    
    def _calculate_metrics(self) -> None:
        """Calculate dependency metrics for all nodes."""
        if not self.graph:
            return
        
        for node in self.graph.nodes:
            # Basic metrics
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            total_degree = in_degree + out_degree
            
            # Circular dependencies
            try:
                cycles = list(nx.simple_cycles(self.graph.subgraph(
                    nx.descendants(self.graph, node) | {node}
                )))
                circular_count = len(cycles)
            except:
                circular_count = 0
            
            # Dependency depth
            try:
                depth = nx.single_source_shortest_path_length(
                    self.graph, node
                )
                max_depth = max(depth.values()) if depth else 0
            except:
                max_depth = 0
            
            self.metrics[node] = DependencyMetrics(
                total_dependencies=total_degree,
                incoming_dependencies=in_degree,
                outgoing_dependencies=out_degree,
                circular_dependencies=circular_count,
                dependency_depth=max_depth,
                fan_in=in_degree,
                fan_out=out_degree
            )
    
    def get_circular_dependencies(self) -> List[List[str]]:
        """Get all circular dependencies in the graph."""
        if not self.graph:
            return []
        
        return list(nx.simple_cycles(self.graph))
    
    def get_strongly_connected_components(self) -> List[Set[str]]:
        """Get strongly connected components (potential circular dependencies)."""
        if not self.graph:
            return []
        
        return list(nx.strongly_connected_components(self.graph))
    
    def get_dependency_layers(self) -> Dict[int, Set[str]]:
        """Get dependency layers (topological levels)."""
        if not self.graph:
            return {}
        
        try:
            layers = {}
            for i, layer in enumerate(nx.topological_generations(self.graph)):
                layers[i] = set(layer)
            return layers
        except nx.NetworkXError:
            # Graph has cycles
            return {}
    
    def get_critical_path(self) -> List[str]:
        """Get the critical path (longest dependency chain)."""
        if not self.graph:
            return []
        
        try:
            # Find longest path in DAG
            longest_path = nx.dag_longest_path(self.graph)
            return longest_path
        except nx.NetworkXError:
            # Graph has cycles
            return []
    
    def export_graph(self, format: str = 'dot') -> str:
        """Export the dependency graph in specified format."""
        if not self.graph:
            return ""
        
        if format == 'dot':
            return nx.nx_agraph.to_agraph(self.graph).to_string()
        elif format == 'gexf':
            return '\n'.join(nx.generate_gexf(self.graph))
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def visualize_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get visualization-ready metrics."""
        if not self.metrics:
            return {}
        
        viz_metrics = {}
        for node, metrics in self.metrics.items():
            viz_metrics[node] = {
                'total_deps': metrics.total_dependencies,
                'fan_in': metrics.fan_in,
                'fan_out': metrics.fan_out,
                'depth': metrics.dependency_depth,
                'circular': metrics.circular_dependencies,
            }
        
        return viz_metrics
