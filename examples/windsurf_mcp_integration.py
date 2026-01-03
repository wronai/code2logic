#!/usr/bin/env python3
"""
Example: Windsurf MCP Integration with LiteLLM.

Demonstrates how to use code2logic with Windsurf IDE through:
1. MCP Server - For Claude Desktop / Windsurf integration
2. LiteLLM - For using multiple LLM providers

Setup for Windsurf:
-------------------
1. Add to your Windsurf/Claude Desktop config:

   ~/.config/windsurf/mcp_config.json (or claude_desktop_config.json):
   {
     "mcpServers": {
       "code2logic": {
         "command": "python",
         "args": ["-m", "code2logic.mcp_server"],
         "env": {
           "PYTHONPATH": "/path/to/code2logic"
         }
       }
     }
   }

2. Or use the standalone MCP server:
   python -m code2logic.mcp_server

LiteLLM Integration:
-------------------
LiteLLM allows using code2logic analysis with any LLM provider:
- OpenAI (gpt-4, gpt-3.5-turbo)
- Anthropic (claude-3-opus, claude-3-sonnet)
- Ollama (local models)
- Groq, Together AI, etc.

Usage:
    python windsurf_mcp_integration.py /path/to/project
    python windsurf_mcp_integration.py /path/to/project --provider openai
    python windsurf_mcp_integration.py /path/to/project --test-mcp
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

from code2logic import analyze_project, CSVGenerator, GherkinGenerator


def check_mcp_server() -> bool:
    """Check if MCP server can be started."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from code2logic.mcp_server import run_server; print('ok')"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return "ok" in result.stdout
    except Exception:
        return False


def test_mcp_call(tool_name: str, arguments: dict) -> Dict[str, Any]:
    """Test MCP tool call directly."""
    from code2logic.mcp_server import call_tool
    
    try:
        result = call_tool(tool_name, arguments)
        return {"success": True, "result": result[:500] + "..." if len(result) > 500 else result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_windsurf_config() -> str:
    """Generate Windsurf MCP configuration."""
    config = {
        "mcpServers": {
            "code2logic": {
                "command": sys.executable,
                "args": ["-m", "code2logic.mcp_server"],
                "env": {
                    "PYTHONPATH": str(Path(__file__).parent.parent)
                }
            }
        }
    }
    return json.dumps(config, indent=2)


def analyze_with_litellm(
    project_path: str,
    model: str = "ollama/qwen2.5-coder:7b",
    task: str = "refactor"
) -> str:
    """Analyze project and get LLM suggestions using LiteLLM."""
    if not LITELLM_AVAILABLE:
        return "Error: LiteLLM not installed. Run: pip install litellm"
    
    # Analyze project
    project = analyze_project(project_path)
    
    # Generate compact analysis
    gen = GherkinGenerator()
    analysis = gen.generate(project, detail='minimal')
    
    # Truncate if too long
    max_tokens = 3000
    if len(analysis) > max_tokens * 4:
        analysis = analysis[:max_tokens * 4] + "\n... (truncated)"
    
    # Build prompt based on task
    prompts = {
        "refactor": f"""Analyze this codebase and suggest 3 specific refactoring improvements:

{analysis}

Focus on:
1. Code duplication
2. Long functions/files
3. Architecture issues

Provide actionable suggestions with estimated effort.""",

        "review": f"""Review this codebase for potential issues:

{analysis}

Check for:
1. Security concerns
2. Performance issues
3. Best practice violations

Be specific and actionable.""",

        "document": f"""Based on this code analysis, generate API documentation:

{analysis}

Include:
1. Module overview
2. Key functions and their purpose
3. Usage examples"""
    }
    
    prompt = prompts.get(task, prompts["refactor"])
    
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert code reviewer and architect."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def main():
    """Main entry point."""
    if len(sys.argv) < 2 or '--help' in sys.argv:
        print(__doc__)
        print("\nOptions:")
        print("  --test-mcp         Test MCP server functionality")
        print("  --generate-config  Generate Windsurf MCP config")
        print("  --provider PROV    LiteLLM provider (ollama/openai/anthropic)")
        print("  --task TASK        Analysis task (refactor/review/document)")
        sys.exit(0)
    
    # Generate config
    if '--generate-config' in sys.argv:
        print("Windsurf MCP Configuration:")
        print("="*50)
        print(generate_windsurf_config())
        print("\nSave this to ~/.config/windsurf/mcp_config.json")
        sys.exit(0)
    
    project_path = sys.argv[1] if not sys.argv[1].startswith('--') else '.'
    
    print("="*60)
    print("WINDSURF MCP INTEGRATION")
    print("="*60)
    
    # Check components
    print("\nComponent Status:")
    print(f"  MCP Server:  {'✓' if check_mcp_server() else '✗'}")
    print(f"  LiteLLM:     {'✓' if LITELLM_AVAILABLE else '✗'}")
    print(f"  HTTPX:       {'✓' if HTTPX_AVAILABLE else '✗'}")
    
    # Test MCP
    if '--test-mcp' in sys.argv:
        print("\n" + "-"*40)
        print("MCP Tool Tests")
        print("-"*40)
        
        tools = [
            ("analyze_project", {"path": project_path, "format": "csv", "detail": "minimal"}),
            ("find_duplicates", {"path": project_path}),
            ("suggest_refactoring", {"path": project_path}),
        ]
        
        for tool_name, args in tools:
            result = test_mcp_call(tool_name, args)
            status = "✓" if result["success"] else "✗"
            print(f"\n{status} {tool_name}")
            if result["success"]:
                print(f"  Result preview: {result['result'][:200]}...")
            else:
                print(f"  Error: {result['error']}")
        
        sys.exit(0)
    
    # LiteLLM analysis
    model = "ollama/qwen2.5-coder:7b"
    task = "refactor"
    
    if '--provider' in sys.argv:
        idx = sys.argv.index('--provider')
        provider = sys.argv[idx + 1]
        models = {
            'ollama': 'ollama/qwen2.5-coder:7b',
            'openai': 'gpt-4',
            'anthropic': 'claude-3-sonnet-20240229',
        }
        model = models.get(provider, model)
    
    if '--task' in sys.argv:
        idx = sys.argv.index('--task')
        task = sys.argv[idx + 1]
    
    print(f"\nAnalyzing: {project_path}")
    print(f"Model: {model}")
    print(f"Task: {task}")
    
    if LITELLM_AVAILABLE:
        print("\n" + "-"*40)
        print("LLM Analysis")
        print("-"*40)
        
        result = analyze_with_litellm(project_path, model, task)
        print(result)
    else:
        print("\n⚠️  LiteLLM not installed.")
        print("   Install with: pip install litellm")
        print("\n   Showing analysis without LLM...")
        
        project = analyze_project(project_path)
        gen = CSVGenerator()
        print(gen.generate(project, detail='minimal')[:2000])
    
    # Show config instructions
    print("\n" + "="*60)
    print("WINDSURF SETUP INSTRUCTIONS")
    print("="*60)
    print("""
1. Generate MCP config:
   python windsurf_mcp_integration.py --generate-config

2. Add to Windsurf config file:
   - Linux: ~/.config/windsurf/mcp_config.json
   - macOS: ~/Library/Application Support/windsurf/mcp_config.json
   - Windows: %APPDATA%/windsurf/mcp_config.json

3. Restart Windsurf

4. Use in Windsurf:
   - "Analyze my project at /path/to/project"
   - "Find duplicates in the codebase"
   - "Suggest refactoring for this code"
""")


if __name__ == '__main__':
    main()
