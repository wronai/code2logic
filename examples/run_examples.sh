#!/bin/bash
# =============================================================================
# Code2Logic Examples Runner
# =============================================================================
# Usage: ./run_examples.sh <command> [options]
#
# Commands:
#   analyze      - Analyze a project and generate output
#   compare      - Compare two projects for duplicates
#   generate     - Generate code from CSV in another language
#   refactor     - Suggest refactoring using LLM
#   deduplicate  - Find and suggest duplicate removal
#   translate    - Translate code logic to another language
#   document     - Generate documentation from analysis
#   gherkin      - Generate Gherkin/BDD test features
#   tests        - Generate test files for various frameworks
#   batch        - Batch analyze multiple projects
#   pipeline     - Run full LLM pipeline
#   api-demo     - Demonstrate Python API usage
#
# Examples:
#   ./run_examples.sh analyze /path/to/project -f csv
#   ./run_examples.sh compare /project1 /project2
#   ./run_examples.sh generate analysis.csv --lang typescript
#   ./run_examples.sh refactor /path/to/project
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
LITELLM_HOST="${LITELLM_HOST:-http://localhost:4000}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
MODEL="${MODEL:-qwen2.5-coder:7b}"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_ollama() {
    if ! curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
        log_error "Ollama not running at $OLLAMA_HOST"
        log_info "Start Ollama: ollama serve"
        exit 1
    fi
    log_info "Ollama connected at $OLLAMA_HOST"
}

check_model() {
    local model="$1"
    if ! curl -s "$OLLAMA_HOST/api/tags" | grep -q "$model"; then
        log_warning "Model $model not found. Pulling..."
        ollama pull "$model"
    fi
}

ensure_output_dir() {
    mkdir -p "$OUTPUT_DIR"
}

# =============================================================================
# Commands
# =============================================================================

cmd_analyze() {
    local project_path="$1"
    shift
    
    if [ -z "$project_path" ]; then
        log_error "Project path required"
        echo "Usage: $0 analyze /path/to/project [-f format] [-d detail] [-o output]"
        exit 1
    fi
    
    log_info "Analyzing project: $project_path"
    ensure_output_dir
    
    # Default options
    local format="csv"
    local detail="standard"
    local output="$OUTPUT_DIR/analysis.csv"
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--format) format="$2"; shift 2 ;;
            -d|--detail) detail="$2"; shift 2 ;;
            -o|--output) output="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    # Update output extension based on format
    case $format in
        json) output="${output%.csv}.json" ;;
        yaml) output="${output%.csv}.yaml" ;;
        csv) output="${output%.json}.csv" ;;
    esac
    
    code2logic "$project_path" -f "$format" -d "$detail" -o "$output"
    
    log_success "Analysis saved to: $output"
    echo ""
    echo "Statistics:"
    wc -l "$output" | awk '{print "  Lines: "$1}'
    du -h "$output" | awk '{print "  Size: "$1}'
}

cmd_compare() {
    local project1="$1"
    local project2="$2"
    
    if [ -z "$project1" ] || [ -z "$project2" ]; then
        log_error "Two project paths required"
        echo "Usage: $0 compare /path/to/project1 /path/to/project2"
        exit 1
    fi
    
    log_info "Comparing projects..."
    ensure_output_dir
    
    # Analyze both projects
    code2logic "$project1" -f csv -d full -o "$OUTPUT_DIR/project1.csv"
    code2logic "$project2" -f csv -d full -o "$OUTPUT_DIR/project2.csv"
    
    log_info "Running comparison..."
    
    # Use Python script for comparison
    python3 - "$OUTPUT_DIR/project1.csv" "$OUTPUT_DIR/project2.csv" << 'PYTHON_COMPARE'
import sys
import csv

def load_csv(path):
    with open(path, 'r') as f:
        return list(csv.DictReader(f))

def compare_projects(file1, file2):
    data1 = load_csv(file1)
    data2 = load_csv(file2)
    
    # Extract hashes
    hashes1 = {r.get('hash', ''): r for r in data1 if r.get('hash')}
    hashes2 = {r.get('hash', ''): r for r in data2 if r.get('hash')}
    
    # Find matches
    common = set(hashes1.keys()) & set(hashes2.keys())
    only1 = set(hashes1.keys()) - set(hashes2.keys())
    only2 = set(hashes2.keys()) - set(hashes1.keys())
    
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Project 1 elements: {len(data1)}")
    print(f"Project 2 elements: {len(data2)}")
    print(f"\nIdentical elements: {len(common)}")
    print(f"Only in Project 1: {len(only1)}")
    print(f"Only in Project 2: {len(only2)}")
    
    if common:
        print(f"\n--- Identical Elements (top 10) ---")
        for i, h in enumerate(list(common)[:10]):
            r = hashes1[h]
            print(f"  {r['type']}: {r['name']} ({r['path']})")
    
    # Find similar by signature
    sigs1 = {r.get('signature', ''): r for r in data1 if r.get('signature')}
    sigs2 = {r.get('signature', ''): r for r in data2 if r.get('signature')}
    similar_sigs = set(sigs1.keys()) & set(sigs2.keys())
    
    print(f"\nSimilar signatures: {len(similar_sigs)}")
    
    return {
        'identical': len(common),
        'only_project1': len(only1),
        'only_project2': len(only2),
        'similar_signatures': len(similar_sigs)
    }

if __name__ == '__main__':
    compare_projects(sys.argv[1], sys.argv[2])
PYTHON_COMPARE
    
    log_success "Comparison complete"
}

cmd_generate() {
    local csv_file="$1"
    local target_lang="${2:-typescript}"
    
    if [ -z "$csv_file" ]; then
        log_error "CSV file required"
        echo "Usage: $0 generate analysis.csv [--lang typescript|python|go|rust]"
        exit 1
    fi
    
    check_ollama
    check_model "$MODEL"
    
    log_info "Generating $target_lang code from: $csv_file"
    ensure_output_dir
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --lang) target_lang="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    python3 - "$csv_file" "$target_lang" "$OLLAMA_HOST" "$MODEL" << 'PYTHON_GENERATE'
import sys
import csv
import json
import httpx

def generate_code(csv_file, target_lang, ollama_host, model):
    # Load CSV
    with open(csv_file, 'r') as f:
        data = list(csv.DictReader(f))
    
    # Group by path/class
    groups = {}
    for row in data:
        key = row.get('path', 'unknown')
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    
    print(f"Generating {target_lang} code for {len(groups)} modules...")
    
    results = []
    for path, elements in list(groups.items())[:5]:  # Limit to 5 modules
        # Build context
        context = f"Module: {path}\n\nElements:\n"
        for e in elements[:10]:
            context += f"- {e.get('type', 'unknown')}: {e.get('name', '')} {e.get('signature', '')}\n"
            if e.get('intent'):
                context += f"  Intent: {e['intent']}\n"
        
        prompt = f"""Generate {target_lang} code based on this specification:

{context}

Generate clean, idiomatic {target_lang} code with:
1. Type annotations
2. Docstrings/comments
3. Error handling

Output only the code, no explanations."""

        # Call Ollama
        try:
            response = httpx.post(
                f"{ollama_host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120.0
            )
            result = response.json()
            code = result.get('response', '')
            
            print(f"\n{'='*60}")
            print(f"Generated: {path} -> {target_lang}")
            print('='*60)
            print(code[:500] + "..." if len(code) > 500 else code)
            
            results.append({
                'source': path,
                'target_lang': target_lang,
                'code': code
            })
        except Exception as e:
            print(f"Error generating {path}: {e}")
    
    # Save results
    output_file = csv_file.replace('.csv', f'_{target_lang}_generated.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    generate_code(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
PYTHON_GENERATE
    
    log_success "Code generation complete"
}

cmd_refactor() {
    local project_path="$1"
    
    if [ -z "$project_path" ]; then
        log_error "Project path required"
        echo "Usage: $0 refactor /path/to/project"
        exit 1
    fi
    
    check_ollama
    check_model "$MODEL"
    
    log_info "Analyzing project for refactoring suggestions..."
    ensure_output_dir
    
    # First analyze
    code2logic "$project_path" -f csv -d full -o "$OUTPUT_DIR/refactor_analysis.csv"
    
    python3 - "$OUTPUT_DIR/refactor_analysis.csv" "$OLLAMA_HOST" "$MODEL" << 'PYTHON_REFACTOR'
import sys
import csv
import httpx
from collections import defaultdict

def analyze_for_refactoring(csv_file, ollama_host, model):
    with open(csv_file, 'r') as f:
        data = list(csv.DictReader(f))
    
    issues = []
    
    # 1. Find high complexity
    high_complexity = [r for r in data if int(r.get('complexity', 0)) > 10]
    for r in high_complexity:
        issues.append({
            'type': 'high_complexity',
            'path': r['path'],
            'name': r['name'],
            'complexity': r['complexity'],
            'suggestion': 'Consider breaking down into smaller functions'
        })
    
    # 2. Find duplicates by hash
    hash_counts = defaultdict(list)
    for r in data:
        if r.get('hash'):
            hash_counts[r['hash']].append(r)
    
    duplicates = {h: rs for h, rs in hash_counts.items() if len(rs) > 1}
    for h, rs in duplicates.items():
        issues.append({
            'type': 'duplicate',
            'elements': [f"{r['path']}::{r['name']}" for r in rs],
            'suggestion': 'Consider extracting to shared utility'
        })
    
    # 3. Find similar signatures (potential DRY violations)
    sig_counts = defaultdict(list)
    for r in data:
        if r.get('signature') and r.get('type') in ('function', 'method'):
            sig_counts[r['signature']].append(r)
    
    similar = {s: rs for s, rs in sig_counts.items() if len(rs) > 2}
    for s, rs in list(similar.items())[:5]:
        issues.append({
            'type': 'similar_signature',
            'signature': s,
            'count': len(rs),
            'suggestion': 'Consider creating generic implementation'
        })
    
    # 4. Find long files
    path_lines = defaultdict(int)
    for r in data:
        path_lines[r['path']] += int(r.get('lines', 0))
    
    long_files = [(p, l) for p, l in path_lines.items() if l > 500]
    for p, l in long_files:
        issues.append({
            'type': 'long_file',
            'path': p,
            'lines': l,
            'suggestion': 'Consider splitting into multiple modules'
        })
    
    print(f"\n{'='*60}")
    print("REFACTORING ANALYSIS")
    print(f"{'='*60}")
    print(f"Total elements analyzed: {len(data)}")
    print(f"Issues found: {len(issues)}")
    
    for issue in issues[:20]:
        print(f"\n[{issue['type'].upper()}]")
        for k, v in issue.items():
            if k != 'type':
                print(f"  {k}: {v}")
    
    # Get LLM suggestions for top issues
    if issues[:3]:
        print(f"\n{'='*60}")
        print("LLM SUGGESTIONS")
        print(f"{'='*60}")
        
        context = "\n".join([
            f"Issue {i+1}: {issue['type']} - {issue.get('suggestion', '')}"
            for i, issue in enumerate(issues[:5])
        ])
        
        prompt = f"""Analyze these code issues and provide specific refactoring suggestions:

{context}

Provide concrete, actionable suggestions for each issue. Be specific about:
1. What pattern to apply
2. How to implement the change
3. Potential risks"""

        try:
            response = httpx.post(
                f"{ollama_host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120.0
            )
            result = response.json()
            print(result.get('response', 'No response'))
        except Exception as e:
            print(f"LLM error: {e}")

if __name__ == '__main__':
    analyze_for_refactoring(sys.argv[1], sys.argv[2], sys.argv[3])
PYTHON_REFACTOR
    
    log_success "Refactoring analysis complete"
}

cmd_deduplicate() {
    local project_path="$1"
    
    if [ -z "$project_path" ]; then
        log_error "Project path required"
        echo "Usage: $0 deduplicate /path/to/project"
        exit 1
    fi
    
    log_info "Finding duplicates in: $project_path"
    ensure_output_dir
    
    code2logic "$project_path" -f csv -d full -o "$OUTPUT_DIR/dedup_analysis.csv"
    
    python3 - "$OUTPUT_DIR/dedup_analysis.csv" << 'PYTHON_DEDUP'
import sys
import csv
from collections import defaultdict

def find_duplicates(csv_file):
    with open(csv_file, 'r') as f:
        data = list(csv.DictReader(f))
    
    print(f"\n{'='*60}")
    print("DUPLICATE DETECTION")
    print(f"{'='*60}")
    
    # By hash (exact duplicates)
    hash_groups = defaultdict(list)
    for r in data:
        if r.get('hash'):
            hash_groups[r['hash']].append(r)
    
    exact_dups = {h: rs for h, rs in hash_groups.items() if len(rs) > 1}
    
    print(f"\n1. EXACT DUPLICATES (same signature): {len(exact_dups)}")
    for h, rs in list(exact_dups.items())[:10]:
        print(f"\n  Hash: {h}")
        for r in rs:
            print(f"    - {r['path']}::{r['name']}")
    
    # By intent (semantic duplicates)
    intent_groups = defaultdict(list)
    for r in data:
        if r.get('intent'):
            # Normalize intent
            intent_key = r['intent'].lower().strip()[:50]
            intent_groups[intent_key].append(r)
    
    semantic_dups = {i: rs for i, rs in intent_groups.items() if len(rs) > 1}
    
    print(f"\n2. SEMANTIC DUPLICATES (similar intent): {len(semantic_dups)}")
    for intent, rs in list(semantic_dups.items())[:10]:
        print(f"\n  Intent: {intent}...")
        for r in rs:
            print(f"    - {r['path']}::{r['name']}")
    
    # By category+domain (potential consolidation)
    cat_domain_groups = defaultdict(list)
    for r in data:
        key = f"{r.get('category', '')}:{r.get('domain', '')}"
        if key != ':':
            cat_domain_groups[key].append(r)
    
    consolidation = [(k, rs) for k, rs in cat_domain_groups.items() if len(rs) > 5]
    
    print(f"\n3. CONSOLIDATION CANDIDATES (same category+domain):")
    for key, rs in sorted(consolidation, key=lambda x: -len(x[1]))[:10]:
        print(f"\n  {key}: {len(rs)} functions")
    
    # Summary
    total_dups = sum(len(rs) - 1 for rs in exact_dups.values())
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total elements: {len(data)}")
    print(f"Exact duplicate functions: {total_dups}")
    print(f"Semantic duplicate groups: {len(semantic_dups)}")
    print(f"Potential consolidation areas: {len(consolidation)}")
    
if __name__ == '__main__':
    find_duplicates(sys.argv[1])
PYTHON_DEDUP
    
    log_success "Duplicate detection complete"
}

cmd_translate() {
    local csv_file="$1"
    local source_lang="$2"
    local target_lang="$3"
    
    if [ -z "$csv_file" ] || [ -z "$target_lang" ]; then
        log_error "CSV file and target language required"
        echo "Usage: $0 translate analysis.csv python typescript"
        exit 1
    fi
    
    check_ollama
    check_model "$MODEL"
    
    log_info "Translating from $source_lang to $target_lang..."
    
    python3 - "$csv_file" "$source_lang" "$target_lang" "$OLLAMA_HOST" "$MODEL" << 'PYTHON_TRANSLATE'
import sys
import csv
import httpx

def translate_code(csv_file, source_lang, target_lang, ollama_host, model):
    with open(csv_file, 'r') as f:
        data = list(csv.DictReader(f))
    
    # Filter by source language
    source_elements = [r for r in data if r.get('language', '').lower() == source_lang.lower()]
    
    print(f"Found {len(source_elements)} {source_lang} elements")
    print(f"Translating to {target_lang}...\n")
    
    for r in source_elements[:5]:
        prompt = f"""Translate this {source_lang} function specification to {target_lang}:

Function: {r.get('name', '')}
Signature: {r.get('signature', '')}
Intent: {r.get('intent', '')}
Category: {r.get('category', '')}

Generate idiomatic {target_lang} code with proper:
1. Type annotations
2. Error handling
3. Documentation

Output only code."""

        try:
            response = httpx.post(
                f"{ollama_host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60.0
            )
            result = response.json()
            code = result.get('response', '')
            
            print(f"{'='*40}")
            print(f"{r['name']} ({source_lang} -> {target_lang})")
            print('='*40)
            print(code[:400])
            print()
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    translate_code(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
PYTHON_TRANSLATE
    
    log_success "Translation complete"
}

cmd_document() {
    local project_path="$1"
    
    if [ -z "$project_path" ]; then
        log_error "Project path required"
        echo "Usage: $0 document /path/to/project"
        exit 1
    fi
    
    log_info "Generating documentation for: $project_path"
    ensure_output_dir
    
    mkdir -p "$OUTPUT_DIR/docs"
    
    # Generate in multiple formats
    code2logic "$project_path" -f markdown -d standard -o "$OUTPUT_DIR/docs/README.md"
    code2logic "$project_path" -f yaml -d standard -o "$OUTPUT_DIR/docs/api.yaml"
    code2logic "$project_path" -f json --flat -d full -o "$OUTPUT_DIR/docs/api.json"
    code2logic "$project_path" -f gherkin -d standard -o "$OUTPUT_DIR/docs/tests.feature"
    
    log_success "Documentation generated in $OUTPUT_DIR/docs/"
    ls -la "$OUTPUT_DIR/docs/"
}

cmd_gherkin() {
    local project_path="$1"
    shift || true
    
    if [ -z "$project_path" ]; then
        log_error "Project path required"
        echo "Usage: $0 gherkin /path/to/project [--steps] [--lang en|pl]"
        exit 1
    fi
    
    log_info "Generating Gherkin features for: $project_path"
    ensure_output_dir
    
    local extra_args=""
    if [[ "$*" == *"--steps"* ]]; then
        extra_args="$extra_args --steps"
    fi
    
    local lang="en"
    if [[ "$*" == *"--lang"* ]]; then
        lang=$(echo "$*" | sed -n 's/.*--lang \([^ ]*\).*/\1/p')
        extra_args="$extra_args --lang $lang"
    fi
    
    python3 "$(dirname "$0")/generate_gherkin.py" "$project_path" --output "$OUTPUT_DIR" --compare $extra_args
    
    log_success "Gherkin generation complete"
}

cmd_tests() {
    local project_path="$1"
    local framework="${2:-pytest}"
    
    if [ -z "$project_path" ]; then
        log_error "Project path required"
        echo "Usage: $0 tests /path/to/project [framework]"
        echo "Frameworks: pytest, bdd, jest, go"
        exit 1
    fi
    
    log_info "Generating $framework tests for: $project_path"
    ensure_output_dir
    
    python3 "$(dirname "$0")/generate_tests.py" "$project_path" --framework "$framework" --output "$OUTPUT_DIR/tests"
    
    log_success "Test generation complete"
}

cmd_batch() {
    log_info "Running batch analysis..."
    
    bash "$(dirname "$0")/batch_analysis.sh" "$@"
}

cmd_pipeline() {
    local project_path="$1"
    
    if [ -z "$project_path" ]; then
        log_error "Project path required"
        echo "Usage: $0 pipeline /path/to/project"
        exit 1
    fi
    
    log_info "Running full LLM pipeline for: $project_path"
    ensure_output_dir
    
    python3 "$(dirname "$0")/llm_pipeline.py" "$project_path" --output "$OUTPUT_DIR"
    
    log_success "Pipeline complete"
}

cmd_api_demo() {
    log_info "Running API usage demonstration..."
    
    python3 "$(dirname "$0")/api_usage.py"
    
    log_success "API demo complete"
}

# =============================================================================
# Main
# =============================================================================

show_help() {
    echo "Code2Logic Examples Runner"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  analyze      Analyze a project and generate output"
    echo "  compare      Compare two projects for duplicates"
    echo "  generate     Generate code from CSV in another language"
    echo "  refactor     Suggest refactoring using LLM"
    echo "  deduplicate  Find and suggest duplicate removal"
    echo "  translate    Translate code logic to another language"
    echo "  document     Generate documentation from analysis"
    echo "  gherkin      Generate Gherkin/BDD test features (50x token savings)"
    echo "  tests        Generate test files (pytest, jest, go)"
    echo "  batch        Batch analyze multiple projects"
    echo "  pipeline     Run full LLM pipeline"
    echo "  api-demo     Demonstrate Python API usage"
    echo ""
    echo "Environment variables:"
    echo "  OLLAMA_HOST  Ollama server (default: http://localhost:11434)"
    echo "  MODEL        Model to use (default: qwen2.5-coder:7b)"
    echo "  OUTPUT_DIR   Output directory (default: ./output)"
    echo ""
    echo "Examples:"
    echo "  $0 analyze /path/to/project -f csv"
    echo "  $0 compare /project1 /project2"
    echo "  $0 gherkin /path/to/project --steps"
    echo "  $0 tests /path/to/project pytest"
    echo "  $0 pipeline /path/to/project"
    echo "  $0 batch /projects/dir"
}

main() {
    local cmd="${1:-help}"
    shift || true
    
    case "$cmd" in
        analyze)     cmd_analyze "$@" ;;
        compare)     cmd_compare "$@" ;;
        generate)    cmd_generate "$@" ;;
        refactor)    cmd_refactor "$@" ;;
        deduplicate) cmd_deduplicate "$@" ;;
        translate)   cmd_translate "$@" ;;
        document)    cmd_document "$@" ;;
        gherkin)     cmd_gherkin "$@" ;;
        tests)       cmd_tests "$@" ;;
        batch)       cmd_batch "$@" ;;
        pipeline)    cmd_pipeline "$@" ;;
        api-demo)    cmd_api_demo "$@" ;;
        help|--help|-h) show_help ;;
        *)
            log_error "Unknown command: $cmd"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
