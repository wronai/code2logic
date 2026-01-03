#!/usr/bin/env python3
"""
Generate code using Ollama LLM integration.

This example demonstrates how to use code2logic's LLM integration
to generate code based on natural language prompts.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code2logic.llm import LLMInterface, LLMConfig
from code2logic.models import Function, Class, Module


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_sample_function() -> Function:
    """Create a sample function for demonstration."""
    return Function(
        name="calculate_area",
        parameters=["width", "height"],
        lines_of_code=3,
        complexity=1,
        docstring="Calculate the area of a rectangle.",
        code="def calculate_area(width, height):\n    return width * height"
    )


def create_sample_class() -> Class:
    """Create a sample class for demonstration."""
    return Class(
        name="Rectangle",
        methods=[
            Function(
                name="__init__",
                parameters=["width", "height"],
                lines_of_code=3,
                complexity=1,
                docstring="Initialize rectangle."
            ),
            Function(
                name="area",
                parameters=[],
                lines_of_code=2,
                complexity=1,
                docstring="Calculate area."
            )
        ],
        lines_of_code=10,
        docstring="A rectangle class."
    )


def generate_code_from_prompt(llm: LLMInterface, prompt: str) -> str:
    """Generate code from a natural language prompt."""
    print(f"🤖 Generating code for prompt: '{prompt}'")
    
    # Create a more detailed prompt for code generation
    detailed_prompt = f"""
Generate Python code for the following request. Provide only the code without explanations:

{prompt}

Requirements:
- Use proper Python syntax
- Include type hints where appropriate
- Add docstrings for functions and classes
- Follow PEP 8 style guidelines
- Make the code production-ready

Example format:
```python
def function_name(param1: type1, param2: type2) -> return_type:
    \"\"\"Function description.\"\"\"
    # implementation
    pass
```
"""
    
    try:
        response = llm._call_llm(detailed_prompt)
        
        # Extract code from response
        code = extract_code_from_response(response)
        
        print("✅ Code generation completed")
        return code
        
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
        return f"# Code generation failed: {e}"


def extract_code_from_response(response: str) -> str:
    """Extract code block from LLM response."""
    lines = response.split('\n')
    code_lines = []
    in_code_block = False
    
    for line in lines:
        if line.strip().startswith('```python'):
            in_code_block = True
            continue
        elif line.strip().startswith('```') and in_code_block:
            in_code_block = False
            continue
        elif in_code_block:
            code_lines.append(line)
        elif not in_code_block and line.strip() and not line.startswith('#'):
            # If no code blocks found, assume entire response is code
            code_lines.append(line)
    
    return '\n'.join(code_lines) if code_lines else response


def improve_existing_code(llm: LLMInterface, function: Function) -> str:
    """Improve existing function code using LLM."""
    print(f"🔧 Improving function: {function.name}")
    
    prompt = f"""
Improve the following Python function. Make it more robust, efficient, and well-documented:

Current function:
```python
{function.code}
```

Provide an improved version with:
1. Better error handling
2. Input validation
3. Comprehensive docstring
4. Type hints
5. Performance optimizations if applicable
6. Edge case handling

Return only the improved function code.
"""
    
    try:
        response = llm._call_llm(prompt)
        improved_code = extract_code_from_response(response)
        
        print("✅ Function improvement completed")
        return improved_code
        
    except Exception as e:
        print(f"❌ Function improvement failed: {e}")
        return function.code


def generate_tests(llm: LLMInterface, function: Function) -> str:
    """Generate unit tests for a function."""
    print(f"🧪 Generating tests for function: {function.name}")
    
    prompt = f"""
Generate comprehensive unit tests for the following Python function using pytest:

Function:
```python
{function.code}
```

Requirements:
1. Test normal operation cases
2. Test edge cases and boundary conditions
3. Test error handling
4. Use pytest fixtures where appropriate
5. Include descriptive test names
6. Add parameterized tests for multiple inputs
7. Mock external dependencies if any

Provide complete, runnable test code.
"""
    
    try:
        response = llm._call_llm(prompt)
        test_code = extract_code_from_response(response)
        
        print("✅ Test generation completed")
        return test_code
        
    except Exception as e:
        print(f"❌ Test generation failed: {e}")
        return f"# Test generation failed: {e}"


def generate_documentation(llm: LLMInterface, target) -> str:
    """Generate documentation for a code element."""
    target_name = getattr(target, 'name', 'unknown')
    target_type = type(target).__name__
    
    print(f"📚 Generating documentation for {target_type}: {target_name}")
    
    if isinstance(target, Function):
        content = target.code
    elif isinstance(target, Class):
        content = f"class {target.name}:\n    # Methods: {len(target.methods)}"
    else:
        content = str(target)
    
    prompt = f"""
Generate comprehensive documentation for the following {target_type.lower()}:

{target_type}:
```python
{content}
```

Create documentation that includes:
1. Purpose and functionality
2. Parameters and return values (if applicable)
3. Usage examples
4. Important notes and warnings
5. Related components
6. Best practices for usage

Format as Markdown with proper headings and code examples.
"""
    
    try:
        response = llm._call_llm(prompt)
        
        print("✅ Documentation generation completed")
        return response
        
    except Exception as e:
        print(f"❌ Documentation generation failed: {e}")
        return f"# Documentation generation failed: {e}"


def explain_code(llm: LLMInterface, code: str, detail_level: str = "medium") -> str:
    """Explain what the code does."""
    print(f"🔍 Explaining code (detail level: {detail_level})")
    
    prompt = f"""
Explain the following Python code. Provide a {detail_level} level of detail:

Code:
```python
{code}
```

Focus on:
1. What the code does
2. How it works
3. Key algorithms or patterns used
4. Important design decisions
5. Potential improvements or issues

Provide a clear, educational explanation.
"""
    
    try:
        response = llm._call_llm(prompt)
        
        print("✅ Code explanation completed")
        return response
        
    except Exception as e:
        print(f"❌ Code explanation failed: {e}")
        return f"# Code explanation failed: {e}"


def save_output(content: str, output_path: str) -> None:
    """Save content to file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"💾 Output saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save output: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate code using Ollama LLM integration'
    )
    
    parser.add_argument(
        '--prompt', '-p',
        help='Natural language prompt for code generation'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./generated_code.py',
        help='Output file path'
    )
    
    parser.add_argument(
        '--improve',
        action='store_true',
        help='Improve existing code (requires --target)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Generate tests for existing code (requires --target)'
    )
    
    parser.add_argument(
        '--document',
        action='store_true',
        help='Generate documentation for existing code (requires --target)'
    )
    
    parser.add_argument(
        '--explain',
        action='store_true',
        help='Explain existing code (requires --target)'
    )
    
    parser.add_argument(
        '--target', '-t',
        help='Target file or function to improve/test/document'
    )
    
    parser.add_argument(
        '--detail-level',
        choices=['low', 'medium', 'high'],
        default='medium',
        help='Detail level for explanations'
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
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Initialize LLM interface
    config = LLMConfig(
        provider=args.provider,
        model=args.model,
        temperature=0.7,
        max_tokens=2000
    )
    
    try:
        llm = LLMInterface(config)
        print(f"🚀 Initialized {args.provider} LLM with model: {args.model}")
        
        # Check LLM health
        health = llm.health_check()
        if not health['available']:
            print(f"❌ LLM service not available: {health.get('error', 'Unknown error')}")
            sys.exit(1)
        
        print("✅ LLM service is healthy")
        
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        print("Make sure Ollama is installed and running:")
        print("  curl -fsSL https://ollama.ai/install.sh | sh")
        print("  ollama serve")
        print("  ollama pull codellama")
        sys.exit(1)
    
    # Execute requested action
    if args.prompt:
        # Generate code from prompt
        code = generate_code_from_prompt(llm, args.prompt)
        save_output(code, args.output)
        
    elif args.improve:
        # Improve existing code
        if not args.target:
            print("❌ --target required for --improve")
            sys.exit(1)
        
        # Read target file
        try:
            with open(args.target, 'r') as f:
                code = f.read()
            
            # Create function object
            func = Function(name="target_function", code=code)
            improved_code = improve_existing_code(llm, func)
            
            # Save improved code
            output_path = args.output.replace('.py', '_improved.py')
            save_output(improved_code, output_path)
            
        except Exception as e:
            print(f"❌ Failed to read target file: {e}")
            sys.exit(1)
    
    elif args.test:
        # Generate tests
        if not args.target:
            print("❌ --target required for --test")
            sys.exit(1)
        
        # Read target file
        try:
            with open(args.target, 'r') as f:
                code = f.read()
            
            # Create function object
            func = Function(name="target_function", code=code)
            test_code = generate_tests(llm, func)
            
            # Save test code
            output_path = args.output.replace('.py', '_test.py')
            save_output(test_code, output_path)
            
        except Exception as e:
            print(f"❌ Failed to read target file: {e}")
            sys.exit(1)
    
    elif args.document:
        # Generate documentation
        if not args.target:
            print("❌ --target required for --document")
            sys.exit(1)
        
        # Read target file
        try:
            with open(args.target, 'r') as f:
                code = f.read()
            
            # Create function object
            func = Function(name="target_function", code=code)
            documentation = generate_documentation(llm, func)
            
            # Save documentation
            output_path = args.output.replace('.py', '.md')
            save_output(documentation, output_path)
            
        except Exception as e:
            print(f"❌ Failed to read target file: {e}")
            sys.exit(1)
    
    elif args.explain:
        # Explain code
        if not args.target:
            print("❌ --target required for --explain")
            sys.exit(1)
        
        # Read target file
        try:
            with open(args.target, 'r') as f:
                code = f.read()
            
            explanation = explain_code(llm, code, args.detail_level)
            
            # Save explanation
            output_path = args.output.replace('.py', '_explanation.md')
            save_output(explanation, output_path)
            
        except Exception as e:
            print(f"❌ Failed to read target file: {e}")
            sys.exit(1)
    
    else:
        # Demo mode - show examples
        print("🎯 Running demo examples...")
        
        # Example 1: Generate code from prompt
        print("\n" + "="*50)
        print("EXAMPLE 1: Generate code from prompt")
        print("="*50)
        
        sample_func = create_sample_function()
        improved_code = improve_existing_code(llm, sample_func)
        
        output_path = args.output.replace('.py', '_demo_improved.py')
        save_output(improved_code, output_path)
        
        # Example 2: Generate tests
        print("\n" + "="*50)
        print("EXAMPLE 2: Generate unit tests")
        print("="*50)
        
        test_code = generate_tests(llm, sample_func)
        
        output_path = args.output.replace('.py', '_demo_test.py')
        save_output(test_code, output_path)
        
        # Example 3: Generate documentation
        print("\n" + "="*50)
        print("EXAMPLE 3: Generate documentation")
        print("="*50)
        
        documentation = generate_documentation(llm, sample_func)
        
        output_path = args.output.replace('.py', '_demo_documentation.md')
        save_output(documentation, output_path)
        
        # Example 4: Explain code
        print("\n" + "="*50)
        print("EXAMPLE 4: Explain code")
        print("="*50)
        
        explanation = explain_code(llm, sample_func.code, args.detail_level)
        
        output_path = args.output.replace('.py', '_demo_explanation.md')
        save_output(explanation, output_path)
        
        print("\n🎉 Demo completed! Check the generated files.")
    
    print("\n✅ Code generation example completed successfully!")


if __name__ == '__main__':
    main()
