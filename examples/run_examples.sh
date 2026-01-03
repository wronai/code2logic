#!/bin/bash

# Main script to run all code2logic examples
# This script demonstrates various features of the code2logic package

set -e

echo "=== code2logic Examples ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if code2logic is installed
check_installation() {
    print_status "Checking code2logic installation..."
    
    if python -c "import code2logic" 2>/dev/null; then
        print_success "code2logic is installed"
    else
        print_error "code2logic is not installed. Please install it first:"
        echo "pip install -e ."
        exit 1
    fi
}

# Create sample project directory
create_sample_project() {
    print_status "Creating sample project..."
    
    SAMPLE_DIR="./sample_project"
    rm -rf "$SAMPLE_DIR"
    mkdir -p "$SAMPLE_DIR"
    
    # Create sample Python files
    cat > "$SAMPLE_DIR/main.py" << 'EOF'
import os
import sys
from typing import List, Dict
from utils import helper_function

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract two numbers."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def get_history(self) -> List[str]:
        """Get calculation history."""
        return self.history.copy()

def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def main():
    """Main function."""
    calc = Calculator()
    
    # Perform some calculations
    result1 = calc.add(10, 5)
    result2 = calc.multiply(result1, 2)
    result3 = calc.subtract(result2, 3)
    
    print(f"Results: {result1}, {result2}, {result3}")
    print("History:", calc.get_history())
    
    # Test helper function
    helper_result = helper_function("test")
    print(f"Helper result: {helper_result}")

if __name__ == "__main__":
    main()
EOF

    cat > "$SAMPLE_DIR/utils.py" << 'EOF'
import json
import re
from typing import Any, Dict, List

def helper_function(data: str) -> Dict[str, Any]:
    """A helper function that processes data."""
    processed_data = {
        "original": data,
        "length": len(data),
        "uppercase": data.upper(),
        "lowercase": data.lower(),
        "words": data.split()
    }
    return processed_data

class DataProcessor:
    """A class for processing various data types."""
    
    def __init__(self):
        self.processed_items = []
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text data."""
        result = {
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split()),
            "has_numbers": bool(re.search(r'\d', text)),
            "has_special_chars": bool(re.search(r'[^a-zA-Z0-9\s]', text))
        }
        self.processed_items.append(result)
        return result
    
    def process_numbers(self, numbers: List[float]) -> Dict[str, float]:
        """Process numeric data."""
        if not numbers:
            return {}
        
        result = {
            "count": len(numbers),
            "sum": sum(numbers),
            "average": sum(numbers) / len(numbers),
            "min": min(numbers),
            "max": max(numbers)
        }
        self.processed_items.append(result)
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        return {
            "total_items": len(self.processed_items),
            "item_types": list(set(type(item).__name__ for item in self.processed_items))
        }

def validate_email(email: str) -> bool:
    """Validate email address."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def format_json(data: Dict[str, Any]) -> str:
    """Format data as JSON string."""
    return json.dumps(data, indent=2)
EOF

    cat > "$SAMPLE_DIR/__init__.py" << 'EOF'
"""Sample project package."""

__version__ = "1.0.0"
__author__ = "Sample Author"
EOF

    print_success "Sample project created in $SAMPLE_DIR"
}

# Run basic analysis example
run_basic_analysis() {
    print_status "Running basic analysis example..."
    
    OUTPUT_DIR="./examples_output"
    mkdir -p "$OUTPUT_DIR"
    
    # Analyze sample project
    python -m code2logic.cli ./sample_project --output "$OUTPUT_DIR/basic_analysis" --format json
    
    if [ -f "$OUTPUT_DIR/basic_analysis.json" ]; then
        print_success "Basic analysis completed: $OUTPUT_DIR/basic_analysis.json"
        
        # Show summary
        print_status "Analysis summary:"
        python -c "
import json
with open('$OUTPUT_DIR/basic_analysis.json', 'r') as f:
    data = json.load(f)
stats = data['project']['statistics']
print(f'Modules: {stats[\"total_modules\"]}')
print(f'Functions: {stats[\"total_functions\"]}')
print(f'Classes: {stats[\"total_classes\"]}')
print(f'Dependencies: {stats[\"total_dependencies\"]}')
print(f'Lines of Code: {stats[\"total_lines_of_code\"]}')
"
    else
        print_error "Basic analysis failed"
    fi
}

# Run multiple formats example
run_multiple_formats() {
    print_status "Running multiple formats example..."
    
    OUTPUT_DIR="./examples_output"
    
    # Generate all formats
    python -m code2logic.cli ./sample_project --output "$OUTPUT_DIR/multi_format" --format all
    
    # Check generated files
    formats=("json" "yaml" "csv" "markdown" "compact")
    
    for format in "${formats[@]}"; do
        if [ "$format" = "csv" ]; then
            # CSV generates multiple files
            if ls "$OUTPUT_DIR/multi_format."*".csv" 1> /dev/null 2>&1; then
                print_success "$format format generated"
            else
                print_warning "$format format not found"
            fi
        else
            if [ -f "$OUTPUT_DIR/multi_format.$format" ]; then
                print_success "$format format generated"
            else
                print_warning "$format format not found"
            fi
        fi
    done
}

# Run dependency analysis
run_dependency_analysis() {
    print_status "Running dependency analysis example..."
    
    python examples/compare_projects.py ./sample_project ./sample_project --output "$OUTPUT_DIR/dependency_analysis.json"
    
    if [ -f "$OUTPUT_DIR/dependency_analysis.json" ]; then
        print_success "Dependency analysis completed"
    else
        print_warning "Dependency analysis failed"
    fi
}

# Run LLM examples (if available)
run_llm_examples() {
    print_status "Running LLM examples..."
    
    # Check if Ollama is available
    if command -v ollama &> /dev/null; then
        print_status "Ollama found, running LLM examples..."
        
        # Run code generation example
        python examples/generate_code.py --prompt "Create a function that validates email addresses" --output "$OUTPUT_DIR/generated_code.py"
        
        if [ -f "$OUTPUT_DIR/generated_code.py" ]; then
            print_success "Code generation example completed"
        else
            print_warning "Code generation example failed"
        fi
        
        # Run refactoring suggestions
        python examples/refactor_suggestions.py --target ./sample_project/main.py --output "$OUTPUT_DIR/refactor_suggestions.json"
        
        if [ -f "$OUTPUT_DIR/refactor_suggestions.json" ]; then
            print_success "Refactoring suggestions completed"
        else
            print_warning "Refactoring suggestions failed"
        fi
    else
        print_warning "Ollama not found. Skipping LLM examples."
        print_status "To run LLM examples, install Ollama:"
        echo "curl -fsSL https://ollama.ai/install.sh | sh"
    fi
}

# Run MCP server example
run_mcp_example() {
    print_status "Running MCP server example..."
    
    # Check if MCP is available
    if python -c "import mcp" 2>/dev/null; then
        print_status "Starting MCP server for 10 seconds..."
        
        # Start server in background
        python -m code2logic.mcp_server --mcp --mcp-port 8080 &
        SERVER_PID=$!
        
        # Wait a bit
        sleep 2
        
        # Test server (simple curl test)
        if command -v curl &> /dev/null; then
            print_status "Testing MCP server..."
            # Note: This is a basic test - actual MCP protocol would be more complex
            if curl -s http://localhost:8080 > /dev/null; then
                print_success "MCP server is responding"
            else
                print_warning "MCP server test failed"
            fi
        fi
        
        # Stop server
        kill $SERVER_PID 2>/dev/null || true
        print_success "MCP server example completed"
    else
        print_warning "MCP not installed. Skipping MCP server example."
        print_status "To run MCP examples, install MCP:"
        echo "pip install mcp"
    fi
}

# Clean up function
cleanup() {
    print_status "Cleaning up..."
    rm -rf ./sample_project
    print_success "Cleanup completed"
}

# Show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all          Run all examples"
    echo "  --basic        Run basic analysis only"
    echo "  --formats      Run multiple formats example"
    echo "  --dependencies Run dependency analysis"
    echo "  --llm          Run LLM examples (requires Ollama)"
    echo "  --mcp          Run MCP server example"
    echo "  --cleanup      Clean up sample files"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all              # Run all examples"
    echo "  $0 --basic            # Run basic analysis only"
    echo "  $0 --llm              # Run LLM examples"
    echo "  $0 --cleanup          # Clean up"
}

# Main execution
main() {
    echo ""
    
    case "${1:-all}" in
        --all)
            check_installation
            create_sample_project
            run_basic_analysis
            run_multiple_formats
            run_dependency_analysis
            run_llm_examples
            run_mcp_example
            ;;
        --basic)
            check_installation
            create_sample_project
            run_basic_analysis
            ;;
        --formats)
            check_installation
            create_sample_project
            run_multiple_formats
            ;;
        --dependencies)
            check_installation
            create_sample_project
            run_dependency_analysis
            ;;
        --llm)
            check_installation
            create_sample_project
            run_llm_examples
            ;;
        --mcp)
            check_installation
            run_mcp_example
            ;;
        --cleanup)
            cleanup
            ;;
        --help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    
    echo ""
    print_success "Examples completed!"
    echo "Output files are in: ./examples_output/"
    echo ""
    echo "To clean up sample files, run:"
    echo "$0 --cleanup"
}

# Run main function with all arguments
main "$@"
