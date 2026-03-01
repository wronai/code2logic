# API Reference

## Core Classes

### UltimateAdvancedDataAnalyzer

Main analysis engine for processing hybrid export data and generating insights.

```python
class UltimateAdvancedDataAnalyzer:
    def __init__(self, hybrid_path: str)
    def run_all_analyses(self) -> List[Dict]
    def analyze_data_hubs_and_consolidation(self) -> Dict
    def extract_redundant_processes(self) -> Dict
    def cluster_data_types_for_unification(self) -> Dict
    def detect_data_flow_cycles(self) -> Dict
    def identify_unused_data_structures(self) -> Dict
    def quantify_process_diversity(self) -> Dict
    def trace_data_mutations_patterns(self) -> Dict
    def score_data_complexity_hotspots(self) -> Dict
    def generate_type_reduction_plan(self) -> Dict
    def analyze_inter_module_dependencies(self) -> Dict
```

**Example Usage:**
```python
analyzer = UltimateAdvancedDataAnalyzer("output_hybrid")
results = analyzer.run_all_analyses()
print(f"Generated {len(results)} analyses")
```

**Parameters:**
- `hybrid_path`: Path to hybrid export directory

**Returns:**
- List of analysis results with insights and LLM queries

---

### LLMRefactoringExecutor

Executes LLM-based refactoring using generated queries.

```python
class LLMRefactoringExecutor:
    def __init__(self, queries_file: str)
    def load_queries(self)
    def execute_refactoring(self) -> List[Dict]
    def _execute_single_refactoring(self, query_data: Dict) -> Dict
    def _parse_llm_query(self, llm_query: str) -> List[Dict]
    def _generate_refactoring_actions(self, function_name: str, actionable_items: List[Dict]) -> List[Dict]
```

**Example Usage:**
```python
executor = LLMRefactoringExecutor("llm_refactoring_queries.yaml")
executor.load_queries()
results = executor.execute_refactoring()
```

**Parameters:**
- `queries_file`: Path to LLM queries YAML file

**Returns:**
- List of refactoring execution results

---

### RefactoringImplementationExecutor

Executes actual refactoring implementation with backup and validation.

```python
class RefactoringImplementationExecutor:
    def __init__(self, base_path: str)
    def execute_implementation(self)
    def _create_backup(self)
    def _load_refactoring_plan(self) -> Dict
    def _execute_phase_1(self, refactoring_plan: Dict)
    def _execute_phase_2(self, refactoring_plan: Dict)
    def _execute_data_hubs_consolidation(self, result: Dict)
```

**Example Usage:**
```python
executor = RefactoringImplementationExecutor(".")
executor.execute_implementation()
```

**Parameters:**
- `base_path`: Base path for project files

**Features:**
- Automatic backup creation
- Phased implementation
- File generation with templates

---

### RefactoringValidator

Validates and tests the complete refactoring pipeline.

```python
class RefactoringValidator:
    def __init__(self, base_path: str)
    def run_complete_validation(self)
    def _validate_generated_files(self)
    def _test_refactored_components(self)
    def _validate_implementation_completeness(self)
    def _generate_validation_report(self)
```

**Example Usage:**
```python
validator = RefactoringValidator(".")
validator.run_complete_validation()
```

**Parameters:**
- `base_path`: Base path for validation

**Returns:**
- Validation report with quality metrics

---

## Configuration Classes

### AnalysisConfig

Configuration for analysis parameters and thresholds.

```python
@dataclass
class AnalysisConfig:
    max_depth: int = 10
    complexity_threshold: int = 10
    redundancy_threshold: int = 5
    include_tests: bool = False
    exclude_patterns: List[str] = field(default_factory=lambda: ["*_test.py", "test_*.py"])
    sampling_enabled: bool = False
    sample_size: int = 1000
    random_seed: int = 42
```

**Example Usage:**
```python
config = AnalysisConfig(
    max_depth=15,
    complexity_threshold=12,
    include_patterns=["*.py"]
)
analyzer = UltimateAdvancedDataAnalyzer("output_hybrid")
analyzer.config = config
```

---

### VisualizationConfig

Configuration for visualization parameters.

```python
@dataclass
class VisualizationConfig:
    layout_algorithm: str = "force_directed"
    node_size_range: Tuple[int, int] = (5, 50)
    edge_width_range: Tuple[int, int] = (1, 10)
    color_scheme: str = "viridis"
    show_labels: bool = True
    enable_zoom: bool = True
    enable_search: bool = True
    export_format: str = "png"
```

**Example Usage:**
```python
viz_config = VisualizationConfig(
    layout="circular",
    color_scheme="plasma",
    node_size_range=(10, 100)
)
```

---

### LLMConfig

Configuration for LLM-based refactoring.

```python
@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4000
    risk_tolerance: str = "medium"
    preserve_comments: bool = True
    generate_tests: bool = True
    backup_before_changes: bool = True
```

**Example Usage:**
```python
llm_config = LLMConfig(
    provider="local",
    model="qwen2.5:3b",
    temperature=0.5
)
executor = LLMRefactoringExecutor("queries.yaml")
executor.config = llm_config
```

---

## Data Structures

### AnalysisResult

Standard structure for analysis function results.

```python
@dataclass
class AnalysisResult:
    function: str
    insights_count: int
    status: str  # "completed" or "failed"
    llm_query: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: Optional[float] = None
```

---

### RefactoringAction

Structure for individual refactoring actions.

```python
@dataclass
class RefactoringAction:
    type: str  # "consolidation", "removal", "simplification", etc.
    description: str
    priority: str  # "high", "medium", "low"
    function: str
    estimated_effort: str  # "low", "medium", "high"
    files_affected: List[str]
    code_template: Optional[str] = None
    test_required: bool = False
```

---

### ImplementationPlan

Structure for phased implementation plans.

```python
@dataclass
class ImplementationPlan:
    function: str
    total_actions: int
    total_phases: int
    estimated_total_days: int
    phases: List[Dict[str, Any]]
    risk_level: str
    dependencies: List[str] = field(default_factory=list)
```

---

## Utility Functions

### File Operations

```python
def load_yaml_file(file_path: Path) -> Dict[str, Any]
def save_yaml_file(data: Dict[str, Any], file_path: Path) -> None
def create_backup_directory(base_path: Path) -> Path
def validate_python_syntax(file_path: Path) -> bool
```

### Graph Analysis

```python
def calculate_centrality_metrics(graph: nx.Graph) -> Dict[str, float]
def detect_communities(graph: nx.Graph) -> List[Set[str]]
def find_cycles(graph: nx.DiGraph) -> List[List[str]]
def calculate_complexity_score(data: Dict[str, Any]) -> int
```

### Code Generation

```python
def generate_dataclass_template(class_name: str, fields: List[str]) -> str
def generate_factory_pattern(class_name: str) -> str
def generate_observer_pattern(subject_name: str) -> str
def generate_strategy_pattern(interface_name: str) -> str
```

---

## Error Handling

### Custom Exceptions

```python
class AnalysisError(Exception):
    """Raised when analysis fails."""
    pass

class RefactoringError(Exception):
    """Raised when refactoring fails."""
    pass

class ValidationError(Exception):
    """Raised when validation fails."""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass
```

### Error Recovery

```python
def handle_analysis_error(error: Exception, function_name: str) -> Dict[str, Any]
def recover_from_failed_refactoring(error: Exception, action: RefactoringAction) -> bool
def validate_configuration(config: AnalysisConfig) -> List[str]
```

---

## Constants and Enums

### AnalysisTypes

```python
class AnalysisType(Enum):
    DATA_HUBS = "analyze_data_hubs_and_consolidation"
    REDUNDANT_PROCESSES = "extract_redundant_processes"
    TYPE_CLUSTERING = "cluster_data_types_for_unification"
    CYCLE_DETECTION = "detect_data_flow_cycles"
    UNUSED_STRUCTURES = "identify_unused_data_structures"
    PROCESS_DIVERSITY = "quantify_process_diversity"
    MUTATION_PATTERNS = "trace_data_mutations_patterns"
    COMPLEXITY_HOTSPOTS = "score_data_complexity_hotspots"
    TYPE_REDUCTION = "generate_type_reduction_plan"
    DEPENDENCIES = "analyze_inter_module_dependencies"
```

### RefactoringTypes

```python
class RefactoringType(Enum):
    CONSOLIDATION = "consolidation"
    REMOVAL = "removal"
    SIMPLIFICATION = "simplification"
    CREATION = "creation"
    REFACTORING = "refactoring"
    PATTERN_APPLICATION = "pattern_application"
```

### QualityLevels

```python
class QualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
```

---

## Integration Examples

### Complete Pipeline

```python
def run_complete_pipeline(base_path: str, hybrid_path: str):
    """Run the complete analysis and refactoring pipeline."""
    
    # 1. Analysis
    analyzer = UltimateAdvancedDataAnalyzer(hybrid_path)
    analysis_results = analyzer.run_all_analyses()
    
    # 2. LLM Refactoring
    executor = LLMRefactoringExecutor("llm_refactoring_queries.yaml")
    refactoring_results = executor.execute_refactoring()
    
    # 3. Implementation
    impl_executor = RefactoringImplementationExecutor(base_path)
    impl_executor.execute_implementation()
    
    # 4. Validation
    validator = RefactoringValidator(base_path)
    validation_results = validator.run_complete_validation()
    
    return {
        'analysis': analysis_results,
        'refactoring': refactoring_results,
        'implementation': impl_executor.implementation_log,
        'validation': validation_results
    }
```

### Custom Analysis

```python
def custom_analysis_example():
    """Example of custom analysis configuration."""
    
    # Configure analysis
    config = AnalysisConfig(
        max_depth=20,
        complexity_threshold=15,
        sampling_enabled=True,
        sample_size=500
    )
    
    # Run specific analyses
    analyzer = UltimateAdvancedDataAnalyzer("output_hybrid")
    
    # Run only specific functions
    results = []
    results.append(analyzer.analyze_data_hubs_and_consolidation())
    results.append(analyzer.detect_data_flow_cycles())
    results.append(analyzer.score_data_complexity_hotspots())
    
    return results
```

### Batch Processing

```python
def batch_process_multiple_projects(project_paths: List[str]):
    """Process multiple projects in batch."""
    
    all_results = {}
    
    for project_path in project_paths:
        try:
            results = run_complete_pipeline(project_path, f"{project_path}/output_hybrid")
            all_results[project_path] = results
        except Exception as e:
            all_results[project_path] = {'error': str(e)}
    
    return all_results
```

---

## Performance Considerations

### Memory Management

```python
# For large codebases, use sampling
config = AnalysisConfig(
    sampling_enabled=True,
    sample_size=1000,
    random_seed=42
)

# Process in chunks for very large datasets
def process_in_chunks(data: List[Any], chunk_size: int = 1000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor

def parallel_analysis(analysis_functions: List[Callable], data: Dict):
    """Run analysis functions in parallel."""
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(func, data) for func in analysis_functions]
        results = [future.result() for future in futures]
    
    return results
```

---

## Testing

### Unit Tests

```python
import pytest
from ultimate_advanced_data_analyzer import UltimateAdvancedDataAnalyzer

class TestAdvancedDataAnalyzer:
    def test_analyze_data_hubs(self):
        analyzer = UltimateAdvancedDataAnalyzer("test_data")
        result = analyzer.analyze_data_hubs_and_consolidation()
        
        assert result['function'] == 'analyze_data_hubs_and_consolidation'
        assert 'insights_count' in result
        assert 'llm_query' in result
    
    def test_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            UltimateAdvancedDataAnalyzer("nonexistent_path")
```

### Integration Tests

```python
def test_complete_pipeline():
    """Test the complete pipeline with sample data."""
    
    # Setup test data
    setup_test_environment()
    
    # Run pipeline
    results = run_complete_pipeline("test_project", "test_project/output_hybrid")
    
    # Validate results
    assert 'analysis' in results
    assert 'refactoring' in results
    assert 'implementation' in results
    assert 'validation' in results
    
    # Cleanup
    cleanup_test_environment()
```

---

This API reference provides comprehensive documentation for all classes, methods, and data structures in the Advanced Data Analysis & Refactoring Pipeline.
