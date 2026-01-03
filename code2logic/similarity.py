"""
Similarity detection for code2logic.

This module provides functionality to detect similarities between
code components using various algorithms and techniques.
"""

import difflib
import re
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from .models import Module, Function, Class, Similarity


@dataclass
class SimilarityConfig:
    """Configuration for similarity detection."""
    structural_threshold: float = 0.7
    semantic_threshold: float = 0.6
    syntactic_threshold: float = 0.8
    min_lines: int = 5
    ignore_whitespace: bool = True
    ignore_comments: bool = True
    ignore_variable_names: bool = False


class SimilarityDetector:
    """Detects similarities between code components."""
    
    def __init__(self, config: Optional[SimilarityConfig] = None):
        """
        Initialize similarity detector.
        
        Args:
            config: Similarity detection configuration
        """
        self.config = config or SimilarityConfig()
        self.similarities: List[Similarity] = []
    
    def detect_similarities(self, modules: List[Module]) -> List[Similarity]:
        """
        Detect similarities between all code components.
        
        Args:
            modules: List of modules to analyze
            
        Returns:
            List of detected similarities
        """
        self.similarities = []
        
        # Collect all functions
        all_functions = []
        for module in modules:
            for func in module.functions:
                all_functions.append((module.name, func))
        
        # Collect all classes
        all_classes = []
        for module in modules:
            for cls in module.classes:
                all_classes.append((module.name, cls))
        
        # Detect function similarities
        func_similarities = self._detect_function_similarities(all_functions)
        self.similarities.extend(func_similarities)
        
        # Detect class similarities
        class_similarities = self._detect_class_similarities(all_classes)
        self.similarities.extend(class_similarities)
        
        # Detect module similarities
        module_similarities = self._detect_module_similarities(modules)
        self.similarities.extend(module_similarities)
        
        return self.similarities
    
    def _detect_function_similarities(
        self, 
        functions: List[Tuple[str, Function]]
    ) -> List[Similarity]:
        """Detect similarities between functions."""
        similarities = []
        
        for i, (module1, func1) in enumerate(functions):
            for j, (module2, func2) in enumerate(functions[i+1:], i+1):
                if func1.lines_of_code < self.config.min_lines or func2.lines_of_code < self.config.min_lines:
                    continue
                
                # Structural similarity
                structural_score = self._calculate_structural_similarity(func1.code, func2.code)
                
                # Semantic similarity
                semantic_score = self._calculate_semantic_similarity(func1, func2)
                
                # Syntactic similarity
                syntactic_score = self._calculate_syntactic_similarity(func1.code, func2.code)
                
                # Overall similarity
                overall_score = (structural_score + semantic_score + syntactic_score) / 3
                
                if overall_score >= self.config.structural_threshold:
                    similarity_type = self._determine_similarity_type(
                        structural_score, semantic_score, syntactic_score
                    )
                    
                    similarity = Similarity(
                        item1=f"{module1}.{func1.name}",
                        item2=f"{module2}.{func2.name}",
                        score=overall_score,
                        similarity_type=similarity_type,
                        details={
                            'structural': structural_score,
                            'semantic': semantic_score,
                            'syntactic': syntactic_score,
                            'lines1': func1.lines_of_code,
                            'lines2': func2.lines_of_code
                        }
                    )
                    similarities.append(similarity)
        
        return similarities
    
    def _detect_class_similarities(
        self, 
        classes: List[Tuple[str, Class]]
    ) -> List[Similarity]:
        """Detect similarities between classes."""
        similarities = []
        
        for i, (module1, class1) in enumerate(classes):
            for j, (module2, class2) in enumerate(classes[i+1:], i+1):
                # Method similarity
                method_similarity = self._calculate_method_similarity(class1, class2)
                
                # Attribute similarity
                attribute_similarity = self._calculate_attribute_similarity(class1, class2)
                
                # Inheritance similarity
                inheritance_similarity = self._calculate_inheritance_similarity(class1, class2)
                
                # Overall similarity
                overall_score = (method_similarity + attribute_similarity + inheritance_similarity) / 3
                
                if overall_score >= self.config.structural_threshold:
                    similarity = Similarity(
                        item1=f"{module1}.{class1.name}",
                        item2=f"{module2}.{class2.name}",
                        score=overall_score,
                        similarity_type="structural",
                        details={
                            'method_similarity': method_similarity,
                            'attribute_similarity': attribute_similarity,
                            'inheritance_similarity': inheritance_similarity,
                            'methods1': len(class1.methods),
                            'methods2': len(class2.methods)
                        }
                    )
                    similarities.append(similarity)
        
        return similarities
    
    def _detect_module_similarities(self, modules: List[Module]) -> List[Similarity]:
        """Detect similarities between modules."""
        similarities = []
        
        for i, module1 in enumerate(modules):
            for j, module2 in enumerate(modules[i+1:], i+1):
                # Import similarity
                import_similarity = self._calculate_import_similarity(module1, module2)
                
                # Structure similarity
                structure_similarity = self._calculate_module_structure_similarity(module1, module2)
                
                # Overall similarity
                overall_score = (import_similarity + structure_similarity) / 2
                
                if overall_score >= self.config.structural_threshold:
                    similarity = Similarity(
                        item1=module1.name,
                        item2=module2.name,
                        score=overall_score,
                        similarity_type="structural",
                        details={
                            'import_similarity': import_similarity,
                            'structure_similarity': structure_similarity,
                            'functions1': len(module1.functions),
                            'functions2': len(module2.functions),
                            'classes1': len(module1.classes),
                            'classes2': len(module2.classes)
                        }
                    )
                    similarities.append(similarity)
        
        return similarities
    
    def _calculate_structural_similarity(self, code1: str, code2: str) -> float:
        """Calculate structural similarity between code blocks."""
        # Normalize code
        norm_code1 = self._normalize_code(code1)
        norm_code2 = self._normalize_code(code2)
        
        # Calculate sequence matcher similarity
        similarity = difflib.SequenceMatcher(None, norm_code1, norm_code2).ratio()
        
        return similarity
    
    def _calculate_semantic_similarity(self, func1: Function, func2: Function) -> float:
        """Calculate semantic similarity between functions."""
        similarity_score = 0.0
        factors = 0
        
        # Name similarity
        name_similarity = difflib.SequenceMatcher(None, func1.name, func2.name).ratio()
        similarity_score += name_similarity
        factors += 1
        
        # Parameter count similarity
        param_similarity = 1.0 - abs(len(func1.parameters) - len(func2.parameters)) / max(len(func1.parameters), len(func2.parameters), 1)
        similarity_score += param_similarity
        factors += 1
        
        # Complexity similarity
        complexity_similarity = 1.0 - abs(func1.complexity - func2.complexity) / max(func1.complexity, func2.complexity, 1)
        similarity_score += complexity_similarity
        factors += 1
        
        # Docstring similarity
        if func1.docstring and func2.docstring:
            doc_similarity = difflib.SequenceMatcher(None, func1.docstring, func2.docstring).ratio()
            similarity_score += doc_similarity
            factors += 1
        
        return similarity_score / factors if factors > 0 else 0.0
    
    def _calculate_syntactic_similarity(self, code1: str, code2: str) -> float:
        """Calculate syntactic similarity between code blocks."""
        # Extract tokens/patterns
        tokens1 = self._extract_code_tokens(code1)
        tokens2 = self._extract_code_tokens(code2)
        
        # Calculate Jaccard similarity
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_method_similarity(self, class1: Class, class2: Class) -> float:
        """Calculate method similarity between classes."""
        methods1 = {method.name: method for method in class1.methods}
        methods2 = {method.name: method for method in class2.methods}
        
        if not methods1 or not methods2:
            return 0.0
        
        # Calculate method name similarity
        names1 = set(methods1.keys())
        names2 = set(methods2.keys())
        
        intersection = len(names1.intersection(names2))
        union = len(names1.union(names2))
        
        name_similarity = intersection / union if union > 0 else 0.0
        
        # Calculate method count similarity
        count_similarity = 1.0 - abs(len(methods1) - len(methods2)) / max(len(methods1), len(methods2), 1)
        
        return (name_similarity + count_similarity) / 2
    
    def _calculate_attribute_similarity(self, class1: Class, class2: Class) -> float:
        """Calculate attribute similarity between classes."""
        attrs1 = set(class1.attributes)
        attrs2 = set(class2.attributes)
        
        if not attrs1 or not attrs2:
            return 0.0
        
        intersection = len(attrs1.intersection(attrs2))
        union = len(attrs1.union(attrs2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_inheritance_similarity(self, class1: Class, class2: Class) -> float:
        """Calculate inheritance similarity between classes."""
        bases1 = set(class1.base_classes)
        bases2 = set(class2.base_classes)
        
        if not bases1 or not bases2:
            return 0.0
        
        intersection = len(bases1.intersection(bases2))
        union = len(bases1.union(bases2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_import_similarity(self, module1: Module, module2: Module) -> float:
        """Calculate import similarity between modules."""
        imports1 = set(module1.imports)
        imports2 = set(module2.imports)
        
        if not imports1 or not imports2:
            return 0.0
        
        intersection = len(imports1.intersection(imports2))
        union = len(imports1.union(imports2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_module_structure_similarity(self, module1: Module, module2: Module) -> float:
        """Calculate structural similarity between modules."""
        similarity_score = 0.0
        factors = 0
        
        # Function count similarity
        func_count_sim = 1.0 - abs(len(module1.functions) - len(module2.functions)) / max(len(module1.functions), len(module2.functions), 1)
        similarity_score += func_count_sim
        factors += 1
        
        # Class count similarity
        class_count_sim = 1.0 - abs(len(module1.classes) - len(module2.classes)) / max(len(module1.classes), len(module2.classes), 1)
        similarity_score += class_count_sim
        factors += 1
        
        # LOC similarity
        loc_sim = 1.0 - abs(module1.lines_of_code - module2.lines_of_code) / max(module1.lines_of_code, module2.lines_of_code, 1)
        similarity_score += loc_sim
        factors += 1
        
        return similarity_score / factors if factors > 0 else 0.0
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison."""
        lines = code.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Remove comments if configured
            if self.config.ignore_comments:
                line = re.sub(r'#.*', '', line)
                line = re.sub(r'//.*', '', line)
                line = re.sub(r'/\*.*?\*/', '', line, flags=re.DOTALL)
            
            # Remove extra whitespace if configured
            if self.config.ignore_whitespace:
                line = ' '.join(line.split())
            
            # Replace variable names if configured
            if self.config.ignore_variable_names:
                line = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', 'VAR', line)
            
            if line.strip():
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def _extract_code_tokens(self, code: str) -> List[str]:
        """Extract tokens from code."""
        # Simple token extraction - can be enhanced with proper lexing
        tokens = []
        
        # Extract keywords and identifiers
        keyword_pattern = r'\b(def|class|if|else|for|while|try|except|import|from|return|break|continue|pass|yield|async|await|lambda|with|as|in|is|not|and|or|True|False|None)\b'
        tokens.extend(re.findall(keyword_pattern, code))
        
        # Extract operators
        operator_pattern = r'(\+|\-|\*|\/|\%|\*\*|\=|\+\=|\-\=|\*\=|\/\=|\=\=|\!\=|\>\=|\<\=|\<|\>|\&|\||\^|\~|\<\<|\>\>)'
        tokens.extend(re.findall(operator_pattern, code))
        
        # Extract brackets
        bracket_pattern = r'(\(|\)|\[|\]|\{|\})'
        tokens.extend(re.findall(bracket_pattern, code))
        
        return tokens
    
    def _determine_similarity_type(
        self, 
        structural: float, 
        semantic: float, 
        syntactic: float
    ) -> str:
        """Determine the primary type of similarity."""
        max_score = max(structural, semantic, syntactic)
        
        if max_score == structural:
            return "structural"
        elif max_score == semantic:
            return "semantic"
        else:
            return "syntactic"
    
    def find_duplicates(self, modules: List[Module], threshold: float = 0.9) -> List[Similarity]:
        """
        Find potential duplicate code.
        
        Args:
            modules: List of modules to analyze
            threshold: Similarity threshold for duplicates
            
        Returns:
            List of duplicate code similarities
        """
        duplicates = []
        
        for similarity in self.similarities:
            if similarity.score >= threshold:
                duplicates.append(similarity)
        
        return duplicates
    
    def get_similarity_clusters(self, threshold: float = 0.7) -> List[List[str]]:
        """
        Group similar items into clusters.
        
        Args:
            threshold: Similarity threshold for clustering
            
        Returns:
            List of clusters (each cluster is a list of item names)
        """
        # Build adjacency matrix
        items = set()
        for sim in self.similarities:
            items.add(sim.item1)
            items.add(sim.item2)
        
        items = list(items)
        adjacency = {item: set() for item in items}
        
        for sim in self.similarities:
            if sim.score >= threshold:
                adjacency[sim.item1].add(sim.item2)
                adjacency[sim.item2].add(sim.item1)
        
        # Find connected components (clusters)
        clusters = []
        visited = set()
        
        for item in items:
            if item not in visited:
                cluster = []
                self._dfs_cluster(item, adjacency, visited, cluster)
                if len(cluster) > 1:
                    clusters.append(cluster)
        
        return clusters
    
    def _dfs_cluster(
        self, 
        item: str, 
        adjacency: Dict[str, Set[str]], 
        visited: Set[str], 
        cluster: List[str]
    ) -> None:
        """DFS to find cluster members."""
        visited.add(item)
        cluster.append(item)
        
        for neighbor in adjacency[item]:
            if neighbor not in visited:
                self._dfs_cluster(neighbor, adjacency, visited, cluster)
    
    def get_similarity_report(self) -> Dict[str, Any]:
        """Generate a similarity analysis report."""
        if not self.similarities:
            return {"message": "No similarities found"}
        
        # Statistics
        total_similarities = len(self.similarities)
        high_similarity = len([s for s in self.similarities if s.score >= 0.8])
        medium_similarity = len([s for s in self.similarities if 0.6 <= s.score < 0.8])
        low_similarity = len([s for s in self.similarities if s.score < 0.6])
        
        # Type distribution
        type_counts = defaultdict(int)
        for sim in self.similarities:
            type_counts[sim.similarity_type] += 1
        
        # Top similarities
        top_similarities = sorted(self.similarities, key=lambda x: x.score, reverse=True)[:10]
        
        # Clusters
        clusters = self.get_similarity_clusters()
        
        return {
            "statistics": {
                "total_similarities": total_similarities,
                "high_similarity": high_similarity,
                "medium_similarity": medium_similarity,
                "low_similarity": low_similarity,
                "average_score": sum(s.score for s in self.similarities) / total_similarities
            },
            "type_distribution": dict(type_counts),
            "top_similarities": [
                {
                    "item1": sim.item1,
                    "item2": sim.item2,
                    "score": sim.score,
                    "type": sim.similarity_type
                }
                for sim in top_similarities
            ],
            "clusters": clusters,
            "duplicates": self.find_duplicates([])
        }
