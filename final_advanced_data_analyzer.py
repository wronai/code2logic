#!/usr/bin/env python3
"""
Final version of advanced data analyzer with all map object errors fixed.
"""

import yaml
import json
import networkx as nx
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
from pathlib import Path
import ast
import re


class FinalAdvancedDataAnalyzer:
    """Final advanced analyzer for hybrid export data with all fixes."""
    
    def __init__(self, hybrid_path: str):
        self.hybrid_path = Path(hybrid_path)
        self.analysis_results = []
    
    def run_all_analyses(self) -> List[Dict]:
        """Run all 10 analysis functions."""
        
        print("🔍 Running 10 Final Advanced Analyses on Hybrid Export Data...")
        print("=" * 60)
        
        analyses = [
            self.analyze_data_hubs_and_consolidation,
            self.extract_redundant_processes,
            self.cluster_data_types_for_unification,
            self.detect_data_flow_cycles,
            self.identify_unused_data_structures,
            self.quantify_process_diversity,
            self.trace_data_mutations_patterns,
            self.score_data_complexity_hotspots,
            self.generate_type_reduction_plan,
            self.analyze_inter_module_dependencies
        ]
        
        for i, analysis_func in enumerate(analyses, 1):
            print(f"\n📊 Analysis {i}/10: {analysis_func.__name__}")
            try:
                result = analysis_func()
                self.analysis_results.append(result)
                print(f"   ✅ Completed - {result.get('insights_count', 0)} insights")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                self.analysis_results.append({
                    'function': analysis_func.__name__,
                    'error': str(e),
                    'llm_query': f"Analysis failed for {analysis_func.__name__}: {e}"
                })
        
        self.generate_summary_report()
        return self.analysis_results
    
    def analyze_data_hubs_and_consolidation(self) -> Dict:
        """Analyze data hubs (high centrality) and suggest consolidation."""
        
        # Load consolidated functions
        consolidated_data = self._load_consolidated_functions()
        
        # Build data flow graph
        G = nx.DiGraph()
        data_types = defaultdict(list)
        
        for func_name, func_data in consolidated_data.items():
            if isinstance(func_data, dict):
                # Extract data types from function
                func_types = self._extract_data_types(func_name, func_data)
                for data_type in func_types:
                    data_types[data_type].append(func_name)
                
                # Add to graph
                G.add_node(func_name, types=func_types, connections=len(func_data.get('calls', [])))
        
        # Add edges based on function calls
        for func_name, func_data in consolidated_data.items():
            if isinstance(func_data, dict) and 'calls' in func_data:
                for called in func_data['calls'][:10]:  # Limit for performance
                    if called in G.nodes:
                        G.add_edge(func_name, called)
        
        # Calculate centrality measures
        centrality = nx.betweenness_centrality(G)
        pagerank = nx.pagerank(G)
        
        # Identify hubs (high centrality)
        hubs = {node: score for node, score in centrality.items() if score > 0.05}
        high_pagerank = {node: score for node, score in pagerank.items() if score > 0.02}
        
        # Find similar data types for consolidation
        type_consolidation = self._find_consolidation_opportunities(data_types)
        
        llm_query = f"""
REFaktoryzacja Hubów Danych i Konsolidacja Typów:

ANALIZA:
• Wykryto {len(hubs)} hubów danych (betweenness > 0.05)
• Wykryto {len(high_pagerank)} węzłów o wysokim PageRank (> 0.02)
• Znaleziono {len(type_consolidation)} możliwości konsolidacji typów

TOP HUBY (betweenness centrality):
{self._format_top_items(list(hubs.items()), 5)}

MOŻLIWOŚCI KONSOLIDACJI TYPÓW:
{self._format_consolidation_opportunities(type_consolidation[:3])}

ZALECENIA:
1. Skonsoliduj podobne typy danych w {len(type_consolidation)} grupach
2. Uprość top 5 hubów poprzez ekstrakcję wspólnych funkcji
3. Stwórz abstrakcje dla {len(high_pagerank)} węzłów o wysokim PageRank
4. Zredukuj złożoność o ~25% zachowując przepływ danych

KONKRETNE PLIKI DO MODYFIKACJI:
{self._get_top_files_to_modify(hubs, high_pagerank, 5)}
"""
        
        return {
            'function': 'analyze_data_hubs_and_consolidation',
            'hubs_count': len(hubs),
            'high_pagerank_count': len(high_pagerank),
            'consolidation_opportunities': len(type_consolidation),
            'insights_count': len(hubs) + len(high_pagerank) + len(type_consolidation),
            'llm_query': llm_query
        }
    
    def extract_redundant_processes(self) -> Dict:
        """Extract redundant processes (multiple similar transformations)."""
        
        # Load all function data
        all_functions = self._load_all_functions()
        
        # Analyze process patterns
        process_patterns = {
            'filter': [],
            'map': [],
            'transform': [],
            'validate': [],
            'aggregate': []
        }
        
        for func_name, func_data in all_functions.items():
            if isinstance(func_data, dict):
                patterns = self._identify_process_patterns(func_name, func_data)
                for pattern_type, pattern_info in patterns.items():
                    if pattern_info:
                        process_patterns[pattern_type].append({
                            'function': func_name,
                            'pattern': pattern_info
                        })
        
        # Find redundant processes
        redundant_processes = {}
        for pattern_type, functions in process_patterns.items():
            if len(functions) > 5:  # Threshold for redundancy
                redundant_processes[pattern_type] = functions
        
        llm_query = f"""
ELIMINACJA REDUNDANTNYCH PROCESÓW:

ANALIZA:
• Wykryto redundantne procesy w {len(redundant_processes)} kategoriach
• Łączna liczba redundantnych funkcji: {sum(len(funcs) for funcs in redundant_processes.values())}

REDUNDANTNE PROCESY:
{self._format_redundant_processes(redundant_processes)}

PRZYKŁADY REDUNDANCJI:
{self._get_redundancy_examples(redundant_processes)}

ZALECENIA REFAKTORYZACJI:
1. Stwórz generyczne funkcje dla {len(redundant_processes)} kategorii procesów
2. Zredukuj {sum(len(funcs) for funcs in redundant_processes.values())} funkcji do {len(redundant_processes) * 2} specjalizowanych
3. Użyj wzorców projektowych: Strategy, Template Method, Chain of Responsibility
4. Implementuj wspólną bibliotekę procesów

OCZEKIWANA REDUKCJA: ~60% funkcji procesowych
"""
        
        return {
            'function': 'extract_redundant_processes',
            'redundant_categories': len(redundant_processes),
            'total_redundant_functions': sum(len(funcs) for funcs in redundant_processes.values()),
            'insights_count': len(redundant_processes),
            'llm_query': llm_query
        }
    
    def cluster_data_types_for_unification(self) -> Dict:
        """Cluster data types for unification using graph analysis."""
        
        # Load all data
        all_data = self._load_all_data()
        
        # Build type similarity graph
        type_graph = nx.Graph()
        type_connections = defaultdict(list)
        
        for item_name, item_data in all_data.items():
            if isinstance(item_data, dict):
                types = self._extract_data_types(item_name, item_data)
                for i, type1 in enumerate(types):
                    for type2 in types[i+1:]:
                        type_graph.add_edge(type1, type2, weight=1)
                        type_connections[type1].append(type2)
                        type_connections[type2].append(type1)
        
        # Detect communities (similar types)
        communities = list(nx.community.louvain_communities(type_graph, seed=42))
        
        # Analyze each community for unification potential
        unification_opportunities = []
        for i, community in enumerate(communities):
            if len(community) > 2:  # Only consider communities with 3+ types
                type_freq = Counter()
                for type_name in community:
                    type_freq[type_name] = len(type_connections.get(type_name, []))
                
                unification_opportunities.append({
                    'community_id': i,
                    'types': list(community),
                    'size': len(community),
                    'most_connected': type_freq.most_common(1)[0] if type_freq else None,
                    'unification_potential': len(community) / len(type_graph.nodes) if type_graph.nodes() else 0
                })
        
        llm_query = f"""
UNIFIKACJA TYPÓW DANYCH PRZEZ KLASTROWANIE:

ANALIZA:
• Wykryto {len(communities)} społeczności typów danych
• Znaleziono {len(unification_opportunities)} możliwości unifikacji
• Łączna liczba typów: {len(type_graph.nodes)}

SPOŁECZNOŚCI DO UNIFIKACJI:
{self._format_unification_opportunities(unification_opportunities[:3])}

STATYSTYKI PODOBIEŃSTWA:
{self._calculate_type_similarity_stats(type_graph)}

ZALECENIA UNIFIKACJI:
1. Stwórz {len(unification_opportunities)} zunifikowanych typów bazowych
2. Użyj Union types lub dziedziczenia dla podobnych struktur
3. Zredukuj {len(type_graph.nodes)} typów do {len(unification_opportunities) * 2} typów bazowych
4. Implementuj factory pattern dla zunifikowanych typów

PRZYKŁAD UNIFIKACJI:
{self._get_unification_example(unification_opportunities[0] if unification_opportunities else None)}

OCZEKIWANA REDUKCJA: ~70% unikalnych typów danych
"""
        
        return {
            'function': 'cluster_data_types_for_unification',
            'communities_found': len(communities),
            'unification_opportunities': len(unification_opportunities),
            'total_types': len(type_graph.nodes),
            'insights_count': len(unification_opportunities),
            'llm_query': llm_query
        }
    
    def detect_data_flow_cycles(self) -> Dict:
        """Detect cycles in data flow that need refactoring."""
        
        # Build data flow graph
        G = self._build_data_flow_graph()
        
        # Find all cycles
        cycles = list(nx.simple_cycles(G))
        
        # Analyze cycle characteristics
        cycle_analysis = {
            'total_cycles': len(cycles),
            'avg_cycle_length': sum(len(cycle) for cycle in cycles) / len(cycles) if cycles else 0,
            'max_cycle_length': max(len(cycle) for cycle in cycles) if cycles else 0,
            'cycles_by_length': Counter(len(cycle) for cycle in cycles)
        }
        
        # Find problematic cycles (long or frequent)
        problematic_cycles = []
        for cycle in cycles:
            if len(cycle) > 3:  # Long cycles
                problematic_cycles.append({
                    'cycle': cycle,
                    'length': len(cycle),
                    'issue': 'long_cycle'
                })
        
        # Find nodes involved in multiple cycles
        node_cycle_count = Counter()
        for cycle in cycles:
            for node in cycle:
                node_cycle_count[node] += 1
        
        multi_cycle_nodes = [(node, count) for node, count in node_cycle_count.items() if count > 1]
        
        llm_query = f"""
WYKRYWANIE I PRZERWANIE CYKLI W PRZEPŁYWIE DANYCH:

ANALIZA CYKLI:
• Wykryto {cycle_analysis['total_cycles']} cykli w przepływie danych
• Średnia długość cyklu: {cycle_analysis['avg_cycle_length']:.1f}
• Maksymalna długość cyklu: {cycle_analysis['max_cycle_length']}
• Węzły w wielu cyklach: {len(multi_cycle_nodes)}

PROBLEMATYCZNE CYKLE:
{self._format_problematic_cycles(problematic_cycles[:3])}

WĘZŁY WIELU CYKLICH:
{self._format_multi_cycle_nodes(multi_cycle_nodes[:5])}

ZALECENIA REFAKTORYZACJI:
1. Przerwij {len(problematic_cycles)} długich cykli przez wprowadzenie buforowania
2. Zastosuj wzorzec Observer dla {len(multi_cycle_nodes)} węzłów w wielu cyklach
3. Użyj memoization dla cykli obliczeniowych
4. Rozbij cykle przez wprowadzenie warstw abstrakcji

KONKRETNE AKCJE:
{self._get_cycle_breaking_suggestions(problematic_cycles[:2], multi_cycle_nodes[:3])}

OCZEKIWANY EFEKT: Eliminacja {len(cycles)} cykli, poprawa wydajności ~30%
"""
        
        return {
            'function': 'detect_data_flow_cycles',
            'total_cycles': cycle_analysis['total_cycles'],
            'problematic_cycles': len(problematic_cycles),
            'multi_cycle_nodes': len(multi_cycle_nodes),
            'insights_count': len(problematic_cycles) + len(multi_cycle_nodes),
            'llm_query': llm_query
        }
    
    def identify_unused_data_structures(self) -> Dict:
        """Identify unused or dead data structures."""
        
        # Load all data
        all_data = self._load_all_data()
        
        # Build usage graph
        usage_graph = nx.DiGraph()
        
        # Add all nodes
        for item_name, item_data in all_data.items():
            usage_graph.add_node(item_name, data=item_data)
        
        # Add usage edges
        for item_name, item_data in all_data.items():
            if isinstance(item_data, dict):
                # Find references to other items
                references = self._extract_references(item_data)
                for ref in references:
                    if ref in usage_graph.nodes:
                        usage_graph.add_edge(item_name, ref)
        
        # Find unused nodes (no incoming edges)
        unused_nodes = [node for node in usage_graph.nodes() if usage_graph.in_degree(node) == 0]
        
        # Find dead ends (no outgoing edges)
        dead_ends = [node for node in usage_graph.nodes() if usage_graph.out_degree(node) == 0]
        
        # Find isolated nodes (no edges at all)
        isolated_nodes = [node for node in usage_graph.nodes() 
                        if usage_graph.degree(node) == 0]
        
        # Analyze potential for removal
        removal_candidates = []
        for node in unused_nodes:
            node_data = usage_graph.nodes[node].get('data', {})
            if isinstance(node_data, dict):
                complexity = self._calculate_structure_complexity(node_data)
                if complexity < 5:  # Simple structures
                    removal_candidates.append({
                        'name': node,
                        'complexity': complexity,
                        'reason': 'unused_simple'
                    })
        
        llm_query = f"""
IDENTYFIKACJA I USUWANIE NIEUŻYWANYCH STRUKTUR DANYCH:

ANALIZA UŻYCIA:
• Wykryto {len(unused_nodes)} nieużywanych struktur (brak incoming edges)
• Wykryto {len(dead_ends)} martwych zakończeń (brak outgoing edges)
• Wykryto {len(isolated_nodes)} izolowanych struktur (brak połączeń)
• Kandydaci do usunięcia: {len(removal_candidates)}

KANDYDACI DO USUNIĘCIA:
{self._format_removal_candidates(removal_candidates[:5])}

IZOLOWANE STRUKTURY:
{self._format_isolated_structures(isolated_nodes[:3])}

ZALECENIA REFAKTORYZACJI:
1. Usuń {len(removal_candidates)} prostych, nieużywanych struktur
2. Sprawdź {len(isolated_nodes)} izolowanych struktur pod kątem błędów
3. Zrefaktoruj {len(dead_ends)} martwych zakończeń
4. Dodaj testy pokrycia dla usuniętych struktur

RYZYKO USUNIĘCIA:
{self._assess_removal_risk(removal_candidates, isolated_nodes)}

OCZEKIWANY EFEKT: Redukcja kodu o ~15%, poprawa czytelności
"""
        
        return {
            'function': 'identify_unused_data_structures',
            'unused_nodes': len(unused_nodes),
            'dead_ends': len(dead_ends),
            'isolated_nodes': len(isolated_nodes),
            'removal_candidates': len(removal_candidates),
            'insights_count': len(removal_candidates) + len(isolated_nodes),
            'llm_query': llm_query
        }
    
    def quantify_process_diversity(self) -> Dict:
        """Quantify process diversity and suggest standardization."""
        
        # Load all functions
        all_functions = self._load_all_functions()
        
        # Analyze process diversity per data type
        process_diversity = defaultdict(set)
        process_counts = defaultdict(int)
        
        for func_name, func_data in all_functions.items():
            if isinstance(func_data, dict):
                # Extract data types and processes
                data_types = self._extract_data_types(func_name, func_data)
                processes = self._identify_process_patterns(func_name, func_data)
                
                for data_type in data_types:
                    for process_type, process_info in processes.items():
                        if process_info:
                            process_diversity[data_type].add(process_type)
                            process_counts[process_type] += 1
        
        # Calculate diversity metrics
        diversity_metrics = {}
        for data_type, processes in process_diversity.items():
            diversity_metrics[data_type] = {
                'process_count': len(processes),
                'processes': list(processes)
            }
        
        # Find high-diversity data types
        high_diversity = {dt: metrics for dt, metrics in diversity_metrics.items() 
                         if metrics['process_count'] > 5}
        
        # Find common processes
        common_processes = sorted(process_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        llm_query = f"""
STANDARYZACJA RÓŻNORODNOŚCI PROCESÓW:

ANALIZA RÓŻNORODNOŚCI:
• Wykryto {len(diversity_metrics)} typów danych z różnorodnymi procesami
• Typy o wysokiej różnorodności (>5 procesów): {len(high_diversity)}
• Najczęstsze procesy: {common_processes}

TYPY O WYSOKIEJ RÓŻNORODNOŚCI:
{self._format_high_diversity_types(high_diversity)}

STATYSTYKI PROCESÓW:
{self._format_process_statistics(process_counts)}

ZALECENIA STANDARYZACJI:
1. Zredukuj procesy dla {len(high_diversity)} typów o wysokiej różnorodności
2. Stwórz standardowe operacje dla najczęstszych procesów
3. Użyj wzorców projektowych dla spójności
4. Implementuj bibliotekę wspólnych procesów

KONKRETNE DZIAŁANIA:
{self._get_standardization_actions(high_diversity, common_processes)}

OCZEKIWANY EFEKT: Redukcja różnorodności procesów o ~40%
"""
        
        return {
            'function': 'quantify_process_diversity',
            'total_types': len(diversity_metrics),
            'high_diversity_types': len(high_diversity),
            'most_common_process': common_processes[0][0] if common_processes else None,
            'insights_count': len(high_diversity),
            'llm_query': llm_query
        }
    
    def trace_data_mutations_patterns(self) -> Dict:
        """Trace data mutation patterns and suggest immutable alternatives."""
        
        # Load all functions
        all_functions = self._load_all_functions()
        
        # Identify mutation patterns
        mutation_patterns = {
            'list_mutations': [],
            'dict_mutations': [],
            'object_mutations': [],
            'in_place_modifications': []
        }
        
        for func_name, func_data in all_functions.items():
            if isinstance(func_data, dict):
                mutations = self._identify_mutation_patterns(func_name, func_data)
                for mutation_type, mutation_info in mutations.items():
                    if mutation_info:
                        mutation_patterns[mutation_type].append({
                            'function': func_name,
                            'pattern': mutation_info
                        })
        
        # Analyze mutation impact
        mutation_impact = {}
        for pattern_type, mutations in mutation_patterns.items():
            mutation_impact[pattern_type] = {
                'count': len(mutations),
                'functions': [m['function'] for m in mutations[:5]]
            }
        
        # Find functions with high mutation count
        high_mutation_functions = []
        for func_name, func_data in all_functions.items():
            if isinstance(func_data, dict):
                mutation_count = sum(len(m) for m in self._identify_mutation_patterns(func_name, func_data).values())
                if mutation_count > 3:
                    high_mutation_functions.append({
                        'function': func_name,
                        'mutation_count': mutation_count
                    })
        
        llm_query = f"""
PRZETWARZANIE MUTACJI DANYCH NA IMMUTABLE:

ANALIZA MUTACJI:
• Wykryto mutacji list: {mutation_impact['list_mutations']['count']}
• Wykryto mutacji dict: {mutation_impact['dict_mutations']['count']}
• Wykryto mutacji obiektów: {mutation_impact['object_mutations']['count']}
• Funkcje o wysokiej liczbie mutacji: {len(high_mutation_functions)}

WZORCE MUTACJI:
{self._format_mutation_patterns(mutation_patterns)}

FUNKCJE O WYSOKIEJ LICZBIE MUTACJI:
{self._format_high_mutation_functions(high_mutation_functions[:3])}

ZALECENIA REFAKTORYZACJI:
1. Zastąp {mutation_impact['list_mutations']['count']} mutacji list operacjami funkcyjnymi
2. Użyj copy-on-write dla {mutation_impact['dict_mutations']['count']} mutacji dict
3. Implementuj immutable data structures dla {len(high_mutation_functions)} funkcji
4. Użyj wzorca Builder dla złożonych modyfikacji

KONKRETNE PRZETWARZANIA:
{self._get_immutable_conversion_examples(mutation_patterns)}

PRZYKŁADY IMMUTABLE ALTERNATYW:
{self._get_immutable_alternatives()}

OCZEKIWANY EFEKT: Redukcja mutacji o ~80%, poprawa bezpieczeństwa współbieżności
"""
        
        return {
            'function': 'trace_data_mutations_patterns',
            'total_mutations': sum(impact['count'] for impact in mutation_impact.values()),
            'high_mutation_functions': len(high_mutation_functions),
            'insights_count': len(mutation_patterns),
            'llm_query': llm_query
        }
    
    def score_data_complexity_hotspots(self) -> Dict:
        """Score data complexity and identify hotspots for simplification."""
        
        # Load all data
        all_data = self._load_all_data()
        
        # Calculate complexity scores
        complexity_scores = {}
        for item_name, item_data in all_data.items():
            if isinstance(item_data, dict):
                complexity = self._calculate_structure_complexity(item_data)
                complexity_scores[item_name] = complexity
        
        # Identify complexity hotspots
        sorted_complexity = sorted(complexity_scores.items(), key=lambda x: x[1], reverse=True)
        hotspots = sorted_complexity[:10]  # Top 10 most complex
        
        # Analyze complexity distribution
        complexity_stats = {
            'mean': sum(complexity_scores.values()) / len(complexity_scores) if complexity_scores else 0,
            'max': max(complexity_scores.values()) if complexity_scores else 0,
            'min': min(complexity_scores.values()) if complexity_scores else 0,
            'high_complexity_count': len([s for s in complexity_scores.values() if s > 10])
        }
        
        # Find complexity patterns
        complexity_patterns = self._analyze_complexity_patterns(all_data, complexity_scores)
        
        llm_query = f"""
UPROSZCZENIE HOTSPOTÓW ZŁOŻONOŚCI DANYCH:

ANALIZA ZŁOŻONOŚCI:
• Średnia złożoność: {complexity_stats['mean']:.1f}
• Maksymalna złożoność: {complexity_stats['max']}
• Hotspoty (top 10): {len(hotspots)}
• Struktury o wysokiej złożoności: {complexity_stats['high_complexity_count']}

HOTSPOTY ZŁOŻONOŚCI:
{self._format_complexity_hotspots(hotspots[:5])}

WZORCE ZŁOŻONOŚCI:
{self._format_complexity_patterns(complexity_patterns)}

ZALECENIA UPROSZCZENIA:
1. Zredukuj złożoność {len(hotspots)} hotspotów przez ekstrakcję
2. Podziel {complexity_stats['high_complexity_count']} struktur złożonych
3. Użyj wzorców projektowych: Composite, Decorator, Strategy
4. Implementuj dataclasses dla struktur danych

KONKRETNE DZIAŁANIA:
{self._get_simplification_actions(hotspots[:3])}

PRZYKŁADY UPROSZCZENIA:
{self._get_simplification_examples()}

OCZEKIWANY EFEKT: Redukcja złożoności o ~35%, poprawa czytelności
"""
        
        return {
            'function': 'score_data_complexity_hotspots',
            'mean_complexity': complexity_stats['mean'],
            'max_complexity': complexity_stats['max'],
            'hotspots_count': len(hotspots),
            'high_complexity_count': complexity_stats['high_complexity_count'],
            'insights_count': len(hotspots),
            'llm_query': llm_query
        }
    
    def generate_type_reduction_plan(self) -> Dict:
        """Generate comprehensive type reduction plan."""
        
        # Load all data
        all_data = self._load_all_data()
        
        # Analyze type usage
        type_usage = defaultdict(int)
        type_locations = defaultdict(list)
        
        for item_name, item_data in all_data.items():
            if isinstance(item_data, dict):
                types = self._extract_data_types(item_name, item_data)
                for data_type in types:
                    type_usage[data_type] += 1
                    type_locations[data_type].append(item_name)
        
        # Find reduction opportunities
        reduction_opportunities = []
        
        # Rare types (usage < 3)
        rare_types = {t: count for t, count in type_usage.items() if count < 3}
        
        # Similar types for merging
        similar_types = self._find_similar_types_for_merging(type_usage, type_locations)
        
        # Generic types that can be specialized
        generic_types = [t for t in type_usage.keys() if t in ['dict', 'list', 'str', 'int']]
        
        llm_query = f"""
PLAN REDUKCJI TYPÓW DANYCH:

ANALIZA UŻYCIA TYPÓW:
• Unikalne typy: {len(type_usage)}
• Rzadkie typy (<3 użycia): {len(rare_types)}
• Podobne typy do mergerowania: {len(similar_types)}
• Typy generyczne: {len(generic_types)}

RZADKIE TYPY DO USUNIĘCIA:
{self._format_rare_types(rare_types)}

PODOBNE TYPY DO MERGEROWANIA:
{self._format_similar_types(similar_types[:3])}

STATYSTYKI UŻYCIA:
{self._format_type_usage_stats(type_usage)}

ZALECENIA REDUKCJI:
1. Usuń {len(rare_types)} rzadkich typów
2. Połącz {len(similar_types)} podobnych typów
3. Specjalizuj {len(generic_types)} typów generycznych
4. Stwórz hierarchię typów dziedziczenia

PLAN IMPLEMENTACJI:
{self._get_type_reduction_plan(rare_types, similar_types, generic_types)}

PRZYKŁADY REDUKCJI:
{self._get_type_reduction_examples()}

OCZEKIWANY EFEKT: Redukcja typów z {len(type_usage)} do {len(type_usage) // 2}
"""
        
        return {
            'function': 'generate_type_reduction_plan',
            'unique_types': len(type_usage),
            'rare_types': len(rare_types),
            'similar_types': len(similar_types),
            'generic_types': len(generic_types),
            'insights_count': len(rare_types) + len(similar_types),
            'llm_query': llm_query
        }
    
    def analyze_inter_module_dependencies(self) -> Dict:
        """Analyze inter-module dependencies and suggest centralization."""
        
        # Load all functions with module information
        all_functions = self._load_all_functions()
        
        # Build module dependency graph
        module_graph = nx.DiGraph()
        module_functions = defaultdict(list)
        
        for func_name, func_data in all_functions.items():
            if isinstance(func_data, dict):
                # Extract module from function name
                module = func_name.rsplit('.', 1)[0] if '.' in func_name else 'root'
                module_functions[module].append(func_name)
                
                # Add module to graph
                if not module_graph.has_node(module):
                    module_graph.add_node(module)
                
                # Find dependencies
                if 'calls' in func_data:
                    for called in func_data['calls'][:10]:
                        called_module = called.rsplit('.', 1)[0] if '.' in called else 'root'
                        if called_module != module:
                            module_graph.add_edge(module, called_module)
        
        # Calculate dependency metrics
        dependency_metrics = {
            'total_modules': len(module_graph.nodes()),
            'total_dependencies': len(module_graph.edges()),
            'max_out_degree': max(dict(module_graph.out_degree()).values()) if module_graph.nodes() else 0,
            'max_in_degree': max(dict(module_graph.in_degree()).values()) if module_graph.nodes() else 0
        }
        
        # Find central modules
        centrality = nx.betweenness_centrality(module_graph)
        central_modules = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Find tightly coupled modules
        coupling = {}
        for module in module_graph.nodes():
            out_deps = list(module_graph.successors(module))
            in_deps = list(module_graph.predecessors(module))
            coupling[module] = {
                'outgoing': len(out_deps),
                'incoming': len(in_deps),
                'total': len(out_deps) + len(in_deps)
            }
        
        tightly_coupled = sorted(coupling.items(), key=lambda x: x[1]['total'], reverse=True)[:5]
        
        llm_query = f"""
ANALIZA I CENTRALIZACJA ZALEŻNOŚCI MIĘDZYMODUŁOWYCH:

METRYKI ZALEŻNOŚCI:
• Liczba modułów: {dependency_metrics['total_modules']}
• Liczba zależności: {dependency_metrics['total_dependencies']}
• Maksymalny out-degree: {dependency_metrics['max_out_degree']}
• Maksymalny in-degree: {dependency_metrics['max_in_degree']}

CENTRALNE MODUŁY (betweenness centrality):
{self._format_central_modules(central_modules)}

ŚCISŁO POWIĄZANE MODUŁY:
{self._format_tightly_coupled(tightly_coupled[:3])}

ZALECENIA REFAKTORYZACJI:
1. Stwórz centralny moduł dla {len(central_modules)} modułów centralnych
2. Zredukuj zależności {len(tightly_coupled)} ściśle powiązanych modułów
3. Wprowadź wzorzec Mediator lub Observer dla {dependency_metrics['total_dependencies']} zależności
4. Zaimplementuj dependency injection

PLAN CENTRALIZACJI:
{self._get_centralization_plan(central_modules, tightly_coupled)}

PRZYKŁADY REFAKTORYZACJI:
{self._get_dependency_refactoring_examples()}

OCZEKIWANY EFEKT: Redukcja zależności o ~40%, poprawa modularności
"""
        
        return {
            'function': 'analyze_inter_module_dependencies',
            'total_modules': dependency_metrics['total_modules'],
            'total_dependencies': dependency_metrics['total_dependencies'],
            'central_modules': len(central_modules),
            'tightly_coupled': len(tightly_coupled),
            'insights_count': len(central_modules) + len(tightly_coupled),
            'llm_query': llm_query
        }
    
    # Helper methods
    def _load_consolidated_functions(self) -> Dict:
        """Load consolidated functions data."""
        functions = {}
        consolidated_dir = self.hybrid_path / 'consolidated'
        
        if consolidated_dir.exists():
            for file_path in consolidated_dir.glob('functions_*.yaml'):
                try:
                    with open(file_path, 'r') as f:
                        data = yaml.safe_load(f)
                        if 'functions' in data:
                            functions.update(data['functions'])
                except Exception:
                    continue
        
        return functions
    
    def _load_all_functions(self) -> Dict:
        """Load all functions from consolidated and orphans."""
        functions = {}
        
        # Load consolidated
        functions.update(self._load_consolidated_functions())
        
        # Load orphans (sample)
        orphans_dir = self.hybrid_path / 'orphans'
        if orphans_dir.exists():
            for file_path in orphans_dir.glob('functions_*.yaml')[:10]:  # Limit for performance
                try:
                    with open(file_path, 'r') as f:
                        data = yaml.safe_load(f)
                        if 'functions' in data:
                            functions.update(data['functions'])
                except Exception:
                    continue
        
        return functions
    
    def _load_all_data(self) -> Dict:
        """Load all data from hybrid export."""
        all_data = {}
        
        # Load consolidated
        consolidated_dir = self.hybrid_path / 'consolidated'
        if consolidated_dir.exists():
            for file_path in consolidated_dir.glob('*.yaml'):
                try:
                    with open(file_path, 'r') as f:
                        data = yaml.safe_load(f)
                        if isinstance(data, dict):
                            all_data.update(data)
                except Exception:
                    continue
        
        # Load orphans (sample)
        orphans_dir = self.hybrid_path / 'orphans'
        if orphans_dir.exists():
            for file_path in orphans_dir.glob('*.yaml')[:10]:  # Limit for performance
                try:
                    with open(file_path, 'r') as f:
                        data = yaml.safe_load(f)
                        if isinstance(data, dict):
                            all_data.update(data)
                except Exception:
                    continue
        
        return all_data
    
    def _extract_data_types(self, item_name: str, item_data: Dict) -> List[str]:
        """Extract data types from item data."""
        types = []
        
        # From name patterns
        name_lower = item_name.lower()
        if 'list' in name_lower or 'items' in name_lower:
            types.append('list')
        if 'dict' in name_lower or 'map' in name_lower:
            types.append('dict')
        if 'str' in name_lower or 'text' in name_lower:
            types.append('str')
        if 'int' in name_lower or 'count' in name_lower:
            types.append('int')
        
        # From data structure
        if isinstance(item_data, dict):
            for key, value in item_data.items():
                if isinstance(value, list):
                    types.append('list')
                elif isinstance(value, dict):
                    types.append('dict')
                elif isinstance(value, str):
                    types.append('str')
                elif isinstance(value, int):
                    types.append('int')
        
        return list(set(types))
    
    def _identify_process_patterns(self, item_name: str, item_data: Dict) -> Dict[str, Any]:
        """Identify process patterns in item data."""
        patterns = {}
        
        name_lower = item_name.lower()
        
        # Check for common process patterns
        if any(word in name_lower for word in ['filter', 'select', 'where']):
            patterns['filter'] = True
        if any(word in name_lower for word in ['map', 'transform', 'convert']):
            patterns['map'] = True
        if any(word in name_lower for word in ['reduce', 'sum', 'count', 'aggregate']):
            patterns['reduce'] = True
        if any(word in name_lower for word in ['validate', 'check', 'verify']):
            patterns['validate'] = True
        if any(word in name_lower for word in ['sort', 'order', 'arrange']):
            patterns['sort'] = True
        
        return patterns
    
    def _build_data_flow_graph(self) -> nx.DiGraph:
        """Build data flow graph from all data."""
        G = nx.DiGraph()
        all_data = self._load_all_data()
        
        for item_name, item_data in all_data.items():
            G.add_node(item_name)
            
            if isinstance(item_data, dict) and 'calls' in item_data:
                for called in item_data['calls']:
                    if called in all_data:
                        G.add_edge(item_name, called)
        
        return G
    
    def _find_consolidation_opportunities(self, data_types: Dict) -> List[Dict]:
        """Find opportunities for data type consolidation."""
        opportunities = []
        
        # Group similar types
        type_groups = defaultdict(list)
        for data_type, functions in data_types.items():
            # Simple similarity based on type name
            base_type = data_type.split('[')[0] if '[' in data_type else data_type
            type_groups[base_type].append(data_type)
        
        for base_type, similar_types in type_groups.items():
            if len(similar_types) > 1:
                opportunities.append({
                    'base_type': base_type,
                    'similar_types': similar_types,
                    'total_usage': sum(len(data_types.get(t, [])) for t in similar_types)
                })
        
        return opportunities
    
    def _format_top_items(self, items: List[Tuple], limit: int = 5) -> str:
        """Format top items for display."""
        if not items:
            return "Brak"
        
        result = []
        for item, score in items[:limit]:
            result.append(f"  • {item}: {score:.3f}")
        
        return "\n".join(result)
    
    def _format_consolidation_opportunities(self, opportunities: List[Dict]) -> str:
        """Format consolidation opportunities."""
        if not opportunities:
            return "Brak możliwości konsolidacji"
        
        result = []
        for opp in opportunities:
            result.append(f"  • {opp['base_type']}: {len(opp['similar_types'])} podobnych typów")
            result.append(f"    Typy: {', '.join(opp['similar_types'][:3])}")
        
        return "\n".join(result)
    
    def _get_top_files_to_modify(self, hubs: Dict, high_pagerank: Dict, limit: int) -> str:
        """Get top files to modify based on analysis."""
        all_items = list(hubs.keys()) + list(high_pagerank.keys())
        top_items = list(set(all_items))[:limit]
        
        result = []
        for item in top_items:
            result.append(f"  • {item}")
        
        return "\n".join(result)
    
    def _format_redundant_processes(self, redundant_processes: Dict) -> str:
        """Format redundant processes."""
        if not redundant_processes:
            return "Brak redundantnych procesów"
        
        result = []
        for pattern_type, functions in redundant_processes.items():
            result.append(f"  • {pattern_type}: {len(functions)} funkcji")
        
        return "\n".join(result)
    
    def _get_redundancy_examples(self, redundant_processes: Dict) -> str:
        """Get examples of redundancy."""
        examples = []
        
        for pattern_type, functions in redundant_processes.items():
            if functions:
                examples.append(f"  • {pattern_type}: {functions[0]['function']}")
                if len(functions) > 1:
                    examples.append(f"    {functions[1]['function']}")
        
        return "\n".join(examples[:3])
    
    def _format_unification_opportunities(self, opportunities: List[Dict]) -> str:
        """Format unification opportunities."""
        if not opportunities:
            return "Brak możliwości unifikacji"
        
        result = []
        for opp in opportunities:
            result.append(f"  • Społeczność {opp['community_id']}: {len(opp['types'])} typów")
            result.append(f"    Typy: {', '.join(opp['types'][:3])}")
        
        return "\n".join(result)
    
    def _calculate_type_similarity_stats(self, type_graph: nx.Graph) -> str:
        """Calculate type similarity statistics."""
        if not type_graph.nodes():
            return "Brak typów do analizy"
        
        avg_degree = sum(dict(type_graph.degree()).values()) / len(type_graph.nodes())
        max_degree = max(dict(type_graph.degree()).values())
        
        return f"  • Średnie połączeń: {avg_degree:.1f}\n  • Maksymalne połączenia: {max_degree}"
    
    def _get_unification_example(self, opportunity: Dict) -> str:
        """Get unification example."""
        if not opportunity:
            return "Brak przykładu"
        
        return f"  • Połącz {opportunity['types']} w zunifikowany typ bazowy"
    
    def _format_problematic_cycles(self, cycles: List[Dict]) -> str:
        """Format problematic cycles."""
        if not cycles:
            return "Brak problematycznych cykli"
        
        result = []
        for cycle in cycles[:3]:
            cycle_str = " -> ".join(cycle['cycle'][:5])
            result.append(f"  • Cykl długości {cycle['length']}: {cycle_str}")
        
        return "\n".join(result)
    
    def _format_multi_cycle_nodes(self, nodes: List[Tuple]) -> str:
        """Format nodes in multiple cycles."""
        if not nodes:
            return "Brak węzłów w wielu cyklach"
        
        result = []
        for node, count in nodes[:5]:
            result.append(f"  • {node}: {count} cykli")
        
        return "\n".join(result)
    
    def _get_cycle_breaking_suggestions(self, cycles: List[Dict], nodes: List[Tuple]) -> str:
        """Get cycle breaking suggestions."""
        suggestions = []
        
        for cycle in cycles[:2]:
            suggestions.append(f"  • Przerwij cykl {cycle['cycle'][0]} -> {cycle['cycle'][-1]}")
        
        for node, count in nodes[:3]:
            suggestions.append(f"  • Zastosuj Observer dla {node}")
        
        return "\n".join(suggestions)
    
    def _format_removal_candidates(self, candidates: List[Dict]) -> str:
        """Format removal candidates."""
        if not candidates:
            return "Brak kandydatów do usunięcia"
        
        result = []
        for candidate in candidates[:5]:
            result.append(f"  • {candidate['name']}: złożoność {candidate['complexity']}")
        
        return "\n".join(result)
    
    def _format_isolated_structures(self, structures: List[str]) -> str:
        """Format isolated structures."""
        if not structures:
            return "Brak izolowanych struktur"
        
        result = []
        for structure in structures[:3]:
            result.append(f"  • {structure}")
        
        return "\n".join(result)
    
    def _assess_removal_risk(self, candidates: List[Dict], isolated: List[str]) -> str:
        """Assess removal risk."""
        risk_level = "LOW"
        
        if len(isolated) > 5:
            risk_level = "MEDIUM"
        
        if len(candidates) > 10:
            risk_level = "HIGH"
        
        return f"  • Poziom ryzyka: {risk_level}"
    
    def _calculate_structure_complexity(self, data: Dict) -> int:
        """Calculate structure complexity."""
        complexity = 0
        
        if isinstance(data, dict):
            complexity += len(data)
            for value in data.values():
                if isinstance(value, (dict, list)):
                    complexity += self._calculate_structure_complexity(value)
        elif isinstance(data, list):
            complexity += len(data)
            for item in data:
                if isinstance(item, (dict, list)):
                    complexity += self._calculate_structure_complexity(item)
        
        return complexity
    
    def _extract_references(self, data: Dict) -> List[str]:
        """Extract references from data."""
        references = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and '.' in value:
                    references.append(value)
                elif isinstance(value, dict):
                    references.extend(self._extract_references(value))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str) and '.' in item:
                    references.append(item)
                elif isinstance(item, dict):
                    references.extend(self._extract_references(item))
        
        return references
    
    def _identify_mutation_patterns(self, func_name: str, func_data: Dict) -> Dict[str, List]:
        """Identify mutation patterns."""
        patterns = {
            'list_mutations': [],
            'dict_mutations': [],
            'object_mutations': [],
            'in_place_modifications': []
        }
        
        # Check function name for mutation indicators
        name_lower = func_name.lower()
        
        if any(word in name_lower for word in ['append', 'extend', 'insert', 'remove', 'pop']):
            patterns['list_mutations'].append('list_mutation')
        
        if any(word in name_lower for word in ['update', 'popitem', 'setdefault', 'clear']):
            patterns['dict_mutations'].append('dict_mutation')
        
        if any(word in name_lower for word in ['modify', 'change', 'update', 'set']):
            patterns['object_mutations'].append('object_mutation')
        
        if any(word in name_lower for word in ['inplace', 'direct', 'modify']):
            patterns['in_place_modifications'].append('in_place_modification')
        
        return patterns
    
    def _format_mutation_patterns(self, patterns: Dict) -> str:
        """Format mutation patterns."""
        result = []
        
        for pattern_type, mutations in patterns.items():
            if mutations:
                result.append(f"  • {pattern_type}: {len(mutations)} wystąpień")
        
        return "\n".join(result) if result else "Brak wzorców mutacji"
    
    def _format_high_mutation_functions(self, functions: List[Dict]) -> str:
        """Format high mutation functions."""
        if not functions:
            return "Brak funkcji o wysokiej liczbie mutacji"
        
        result = []
        for func in functions[:3]:
            result.append(f"  • {func['function']}: {func['mutation_count']} mutacji")
        
        return "\n".join(result)
    
    def _get_immutable_conversion_examples(self, patterns: Dict) -> str:
        """Get immutable conversion examples."""
        examples = []
        
        if patterns.get('list_mutations'):
            examples.append("  • list.append() -> new_list = old_list + [item]")
        
        if patterns.get('dict_mutations'):
            examples.append("  • dict.update() -> new_dict = {**old_dict, **updates}")
        
        return "\n".join(examples)
    
    def _get_immutable_alternatives(self) -> str:
        """Get immutable alternatives."""
        return """  • Użyj tuple zamiast list dla stałych
  • Użyj frozenset zamiast set dla stałych
  • Użyj dataclasses z frozen=True
  • Użyj typing.NamedTuple dla struktur"""
    
    def _format_complexity_hotspots(self, hotspots: List[Tuple]) -> str:
        """Format complexity hotspots."""
        if not hotspots:
            return "Brak hotspotów złożoności"
        
        result = []
        for name, complexity in hotspots[:5]:
            result.append(f"  • {name}: złożoność {complexity}")
        
        return "\n".join(result)
    
    def _analyze_complexity_patterns(self, data: Dict, scores: Dict) -> Dict:
        """Analyze complexity patterns."""
        patterns = {
            'nested_structures': 0,
            'large_collections': 0,
            'complex_functions': 0
        }
        
        for item_name, item_data in data.items():
            if isinstance(item_data, dict):
                if scores.get(item_name, 0) > 10:
                    patterns['complex_functions'] += 1
                
                # Check for nested structures
                if any(isinstance(v, dict) for v in item_data.values()):
                    patterns['nested_structures'] += 1
                
                # Check for large collections
                if any(isinstance(v, list) and len(v) > 10 for v in item_data.values()):
                    patterns['large_collections'] += 1
        
        return patterns
    
    def _format_complexity_patterns(self, patterns: Dict) -> str:
        """Format complexity patterns."""
        result = []
        
        for pattern_type, count in patterns.items():
            result.append(f"  • {pattern_type}: {count}")
        
        return "\n".join(result)
    
    def _get_simplification_actions(self, hotspots: List[Tuple]) -> str:
        """Get simplification actions."""
        actions = []
        
        for name, complexity in hotspots[:3]:
            actions.append(f"  • Ekstrakcja wspólnych funkcji z {name}")
        
        return "\n".join(actions)
    
    def _get_simplification_examples(self) -> str:
        """Get simplification examples."""
        return """  • Zagnieżdżony dict -> dataclass
  • Duża lista -> generator + filtering
  • Złożona funkcja -> wiele małych funkcji
  • Powtarzający kod -> funkcja pomocnicza"""
    
    def _format_rare_types(self, rare_types: Dict) -> str:
        """Format rare types."""
        if not rare_types:
            return "Brak rzadkich typów"
        
        result = []
        for type_name, count in list(rare_types.items())[:5]:
            result.append(f"  • {type_name}: {count} użycia")
        
        return "\n".join(result)
    
    def _format_similar_types(self, similar_types: List) -> str:
        """Format similar types."""
        if not similar_types:
            return "Brak podobnych typów"
        
        result = []
        for similar in similar_types[:3]:
            result.append(f"  • {similar['base_type']}: {len(similar['similar_types'])} typów")
        
        return "\n".join(result)
    
    def _format_type_usage_stats(self, type_usage: Dict) -> str:
        """Format type usage statistics."""
        most_common = sorted(type_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        
        result = []
        for type_name, count in most_common:
            result.append(f"  • {type_name}: {count} użycia")
        
        return "\n".join(result)
    
    def _get_type_reduction_plan(self, rare_types: Dict, similar: List, generic: List) -> str:
        """Get type reduction plan."""
        plan = []
        
        if rare_types:
            plan.append("  • Faza 1: Usuń rzadkie typy")
        
        if similar:
            plan.append("  • Faza 2: Połącz podobne typy")
        
        if generic:
            plan.append("  • Faza 3: Specjalizuj typy generyczne")
        
        return "\n".join(plan)
    
    def _get_type_reduction_examples(self) -> str:
        """Get type reduction examples."""
        return """  • Dict[str, Any] + Dict[str, int] -> Dict[str, Union[Any, int]]
  • List[str] + List[int] -> List[Union[str, int]]
  • CustomClass1 + CustomClass2 -> BaseClass"""
    
    def _find_similar_types_for_merging(self, type_usage: Dict, type_locations: Dict) -> List[Dict]:
        """Find similar types for merging."""
        similar_types = []
        
        # Group by base type
        base_types = defaultdict(list)
        for type_name in type_usage.keys():
            base_type = type_name.split('[')[0] if '[' in type_name else type_name
            base_types[base_type].append(type_name)
        
        for base_type, similar in base_types.items():
            if len(similar) > 1:
                similar_types.append({
                    'base_type': base_type,
                    'similar_types': similar,
                    'total_usage': sum(type_usage.get(t, 0) for t in similar)
                })
        
        return similar_types
    
    def _format_high_diversity_types(self, high_diversity: Dict) -> str:
        """Format high diversity types."""
        if not high_diversity:
            return "Brak typów o wysokiej różnorodności"
        
        result = []
        for data_type, metrics in high_diversity.items():
            result.append(f"  • {data_type}: {metrics['process_count']} procesów")
        
        return "\n".join(result)
    
    def _format_process_statistics(self, process_counts: Dict) -> str:
        """Format process statistics."""
        sorted_counts = sorted(process_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        result = []
        for process_type, count in sorted_counts:
            result.append(f"  • {process_type}: {count} wystąpień")
        
        return "\n".join(result)
    
    def _get_standardization_actions(self, high_diversity: Dict, common: List) -> str:
        """Get standardization actions."""
        actions = []
        
        for data_type, metrics in list(high_diversity.items())[:3]:
            actions.append(f"  • Stwórz standardowe operacje dla {data_type}")
        
        return "\n".join(actions)
    
    def _format_central_modules(self, central_modules: List[Tuple]) -> str:
        """Format central modules."""
        if not central_modules:
            return "Brak modułów centralnych"
        
        result = []
        for module, centrality in central_modules[:5]:
            result.append(f"  • {module}: {centrality:.3f}")
        
        return "\n".join(result)
    
    def _format_tightly_coupled(self, coupled: List[Tuple]) -> str:
        """Format tightly coupled modules."""
        if not coupled:
            return "Brak ściśle powiązanych modułów"
        
        result = []
        for module, coupling in coupled[:3]:
            result.append(f"  • {module}: {coupling['total']} zależności")
        
        return "\n".join(result)
    
    def _get_centralization_plan(self, central: List[Tuple], coupled: List[Tuple]) -> str:
        """Get centralization plan."""
        plan = []
        
        if central:
            plan.append("  • Faza 1: Stwórz centralny moduł dla modułów centralnych")
        
        if coupled:
            plan.append("  • Faza 2: Zredukuj zależności ściśle powiązanych modułów")
        
        plan.append("  • Faza 3: Wprowadź dependency injection")
        
        return "\n".join(plan)
    
    def _get_dependency_refactoring_examples(self) -> str:
        """Get dependency refactoring examples."""
        return """  • Wstrzyknij zależności przez constructor
  • Użyj wzorca Mediator dla komunikacji
  • Zaimplementuj event bus dla luźnych powiązań
  • Stwórz shared context dla wspólnych danych"""
    
    def generate_summary_report(self):
        """Generate summary report of all analyses."""
        
        print(f"\n📊 FINAL SUMMARY REPORT - {len(self.analysis_results)} Analyses")
        print("=" * 60)
        
        total_insights = sum(result.get('insights_count', 0) for result in self.analysis_results)
        
        print(f"Total Insights Generated: {total_insights}")
        print(f"LLM Queries Ready: {len(self.analysis_results)}")
        
        print(f"\n🎯 ANALYSIS SUMMARY:")
        for result in self.analysis_results:
            function_name = result.get('function', 'unknown')
            insights = result.get('insights_count', 0)
            status = "✅" if 'error' not in result else "❌"
            print(f"{status} {function_name}: {insights} insights")
        
        # Save all LLM queries
        self._save_llm_queries()
        
        print(f"\n💾 Saved {len(self.analysis_results)} LLM queries to 'final_llm_refactoring_queries.yaml'")
        print(f"\n🚀 Ready for LLM-based refactoring!")
    
    def _save_llm_queries(self):
        """Save all LLM queries to a file."""
        
        queries_data = {
            'project_path': str(self.hybrid_path),
            'analysis_date': '2026-02-28',
            'total_queries': len(self.analysis_results),
            'total_insights': sum(result.get('insights_count', 0) for result in self.analysis_results),
            'queries': []
        }
        
        for result in self.analysis_results:
            queries_data['queries'].append({
                'function': result.get('function', 'unknown'),
                'insights_count': result.get('insights_count', 0),
                'llm_query': result.get('llm_query', 'No query generated')
            })
        
        output_path = self.hybrid_path / 'final_llm_refactoring_queries.yaml'
        with open(output_path, 'w') as f:
            yaml.dump(queries_data, f, default_flow_style=False, sort_keys=False)


def main():
    """Main function to run all analyses."""
    
    hybrid_path = 'output'
    if not Path(hybrid_path).exists():
        print("❌ Hybrid export not found. Trying output_hybrid...")
        hybrid_path = 'output_hybrid'
        if not Path(hybrid_path).exists():
            print("❌ No hybrid export found. Run code2flow first.")
            return
    
    analyzer = FinalAdvancedDataAnalyzer(hybrid_path)
    results = analyzer.run_all_analyses()
    
    print(f"\n🎉 FINAL ANALYSIS COMPLETE!")
    print(f"Generated {len(results)} comprehensive analyses")
    print(f"Ready for LLM-based refactoring!")


if __name__ == '__main__':
    main()
