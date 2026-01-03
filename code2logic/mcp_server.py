"""
MCP (Model Context Protocol) server for Claude Desktop integration.

This module provides an MCP server that allows Claude Desktop
to interact with code2logic for intelligent code analysis.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from .analyzer import ProjectAnalyzer
from .models import Project, Module, Function, Class
from .llm import LLMInterface, LLMConfig
from .intent import IntentAnalyzer

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP server for Claude Desktop integration."""
    
    def __init__(self, port: int = 8080):
        """
        Initialize MCP server.
        
        Args:
            port: Port to run the server on
        """
        if not MCP_AVAILABLE:
            raise ImportError("MCP package is required. Install with: pip install mcp")
        
        self.port = port
        self.server = Server("code2logic")
        self.analyzer: Optional[ProjectAnalyzer] = None
        self.llm: Optional[LLMInterface] = None
        self.intent_analyzer = IntentAnalyzer()
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self) -> None:
        """Register MCP tools."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="analyze_project",
                    description="Analyze a code project and extract structure",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_path": {
                                "type": "string",
                                "description": "Path to the project directory"
                            }
                        },
                        "required": ["project_path"]
                    }
                ),
                Tool(
                    name="get_project_summary",
                    description="Get a summary of the analyzed project",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="list_modules",
                    description="List all modules in the analyzed project",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="get_module_details",
                    description="Get detailed information about a specific module",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "module_name": {
                                "type": "string",
                                "description": "Name of the module"
                            }
                        },
                        "required": ["module_name"]
                    }
                ),
                Tool(
                    name="analyze_dependencies",
                    description="Analyze project dependencies",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="detect_code_smells",
                    description="Detect code smells and issues",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="suggest_refactoring",
                    description="Get refactoring suggestions for a target",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "Target module, class, or function name"
                            }
                        },
                        "required": ["target"]
                    }
                ),
                Tool(
                    name="generate_documentation",
                    description="Generate documentation for a target",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "Target module, class, or function name"
                            },
                            "style": {
                                "type": "string",
                                "enum": ["google", "numpy", "sphinx"],
                                "default": "google"
                            }
                        },
                        "required": ["target"]
                    }
                ),
                Tool(
                    name="explain_code",
                    description="Explain code functionality",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "Target module, class, or function name"
                            },
                            "detail_level": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                                "default": "medium"
                            }
                        },
                        "required": ["target"]
                    }
                ),
                Tool(
                    name="analyze_intent",
                    description="Analyze user intent from natural language query",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="export_project",
                    description="Export project analysis to various formats",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "format": {
                                "type": "string",
                                "enum": ["json", "yaml", "csv", "markdown"],
                                "default": "json"
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Output file path"
                            }
                        },
                        "required": []
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "analyze_project":
                    result = await self._analyze_project(arguments)
                elif name == "get_project_summary":
                    result = await self._get_project_summary()
                elif name == "list_modules":
                    result = await self._list_modules()
                elif name == "get_module_details":
                    result = await self._get_module_details(arguments)
                elif name == "analyze_dependencies":
                    result = await self._analyze_dependencies()
                elif name == "detect_code_smells":
                    result = await self._detect_code_smells()
                elif name == "suggest_refactoring":
                    result = await self._suggest_refactoring(arguments)
                elif name == "generate_documentation":
                    result = await self._generate_documentation(arguments)
                elif name == "explain_code":
                    result = await self._explain_code(arguments)
                elif name == "analyze_intent":
                    result = await self._analyze_intent(arguments)
                elif name == "export_project":
                    result = await self._export_project(arguments)
                else:
                    result = {"error": f"Unknown tool: {name}"}
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            except Exception as e:
                logger.error(f"Tool call failed: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "error": str(e)
                }))]
    
    async def _analyze_project(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a project."""
        project_path = arguments["project_path"]
        
        # Validate path
        path = Path(project_path)
        if not path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
        
        # Create analyzer
        self.analyzer = ProjectAnalyzer(str(path))
        
        # Analyze project
        project = self.analyzer.analyze()
        
        # Initialize LLM if available
        try:
            self.llm = LLMInterface()
        except Exception:
            logger.warning("LLM not available")
        
        return {
            "status": "success",
            "project": {
                "name": project.name,
                "path": project.path,
                "modules": len(project.modules),
                "functions": sum(len(m.functions) for m in project.modules),
                "classes": sum(len(m.classes) for m in project.modules),
                "dependencies": len(project.dependencies)
            }
        }
    
    async def _get_project_summary(self) -> Dict[str, Any]:
        """Get project summary."""
        if not self.analyzer or not self.analyzer.project:
            raise ValueError("No project analyzed yet")
        
        project = self.analyzer.project
        
        return {
            "name": project.name,
            "path": project.path,
            "statistics": {
                "modules": len(project.modules),
                "functions": sum(len(m.functions) for m in project.modules),
                "classes": sum(len(m.classes) for m in project.modules),
                "dependencies": len(project.dependencies),
                "lines_of_code": sum(m.lines_of_code for m in project.modules)
            },
            "metadata": project.metadata
        }
    
    async def _list_modules(self) -> Dict[str, Any]:
        """List all modules."""
        if not self.analyzer or not self.analyzer.project:
            raise ValueError("No project analyzed yet")
        
        modules = []
        for module in self.analyzer.project.modules:
            modules.append({
                "name": module.name,
                "path": module.path,
                "lines_of_code": module.lines_of_code,
                "functions": len(module.functions),
                "classes": len(module.classes),
                "imports": len(module.imports)
            })
        
        return {"modules": modules}
    
    async def _get_module_details(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get module details."""
        if not self.analyzer or not self.analyzer.project:
            raise ValueError("No project analyzed yet")
        
        module_name = arguments["module_name"]
        
        for module in self.analyzer.project.modules:
            if module.name == module_name:
                return {
                    "name": module.name,
                    "path": module.path,
                    "lines_of_code": module.lines_of_code,
                    "imports": module.imports,
                    "functions": [
                        {
                            "name": func.name,
                            "lines_of_code": func.lines_of_code,
                            "complexity": func.complexity,
                            "parameters": func.parameters,
                            "has_docstring": func.docstring is not None
                        }
                        for func in module.functions
                    ],
                    "classes": [
                        {
                            "name": cls.name,
                            "lines_of_code": cls.lines_of_code,
                            "base_classes": cls.base_classes,
                            "methods": len(cls.methods)
                        }
                        for cls in module.classes
                    ]
                }
        
        raise ValueError(f"Module not found: {module_name}")
    
    async def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependencies."""
        if not self.analyzer or not self.analyzer.project:
            raise ValueError("No project analyzed yet")
        
        dependencies = []
        for dep in self.analyzer.project.dependencies:
            dependencies.append({
                "source": dep.source,
                "target": dep.target,
                "type": dep.type,
                "strength": dep.strength
            })
        
        return {"dependencies": dependencies}
    
    async def _detect_code_smells(self) -> Dict[str, Any]:
        """Detect code smells."""
        if not self.analyzer or not self.analyzer.project:
            raise ValueError("No project analyzed yet")
        
        smells = self.intent_analyzer.detect_code_smells(self.analyzer.project)
        
        return {"code_smells": smells}
    
    async def _suggest_refactoring(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest refactoring."""
        if not self.analyzer or not self.analyzer.project:
            raise ValueError("No project analyzed yet")
        
        target = arguments["target"]
        
        # Get basic suggestions
        suggestions = self.intent_analyzer.suggest_refactoring(
            target, self.analyzer.project
        )
        
        result = {"suggestions": suggestions}
        
        # Try to get LLM suggestions if available
        if self.llm:
            try:
                target_obj = self._find_target_object(target)
                if target_obj:
                    llm_suggestions = self.llm.suggest_refactoring(
                        target_obj, self.analyzer.project
                    )
                    result["llm_suggestions"] = llm_suggestions
            except Exception as e:
                logger.warning(f"LLM suggestions failed: {e}")
        
        return result
    
    async def _generate_documentation(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate documentation."""
        if not self.analyzer or not self.analyzer.project:
            raise ValueError("No project analyzed yet")
        
        target = arguments["target"]
        style = arguments.get("style", "google")
        
        if not self.llm:
            raise ValueError("LLM not available for documentation generation")
        
        target_obj = self._find_target_object(target)
        if not target_obj:
            raise ValueError(f"Target not found: {target}")
        
        documentation = self.llm.generate_documentation(target_obj, style)
        
        return {"documentation": documentation}
    
    async def _explain_code(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Explain code."""
        if not self.analyzer or not self.analyzer.project:
            raise ValueError("No project analyzed yet")
        
        target = arguments["target"]
        detail_level = arguments.get("detail_level", "medium")
        
        if not self.llm:
            raise ValueError("LLM not available for code explanation")
        
        target_obj = self._find_target_object(target)
        if not target_obj:
            raise ValueError(f"Target not found: {target}")
        
        if isinstance(target_obj, (Function, Class)):
            code = target_obj.code if hasattr(target_obj, 'code') else str(target_obj)
        else:
            code = f"Module: {target_obj.name}"
        
        explanation = self.llm.explain_code(code, detail_level)
        
        return {"explanation": explanation}
    
    async def _analyze_intent(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user intent."""
        if not self.analyzer or not self.analyzer.project:
            raise ValueError("No project analyzed yet")
        
        query = arguments["query"]
        
        intents = self.intent_analyzer.analyze_intent(query, self.analyzer.project)
        
        return {
            "intents": [
                {
                    "type": intent.type.value,
                    "confidence": intent.confidence,
                    "target": intent.target,
                    "description": intent.description,
                    "suggestions": intent.suggestions
                }
                for intent in intents
            ]
        }
    
    async def _export_project(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Export project."""
        if not self.analyzer or not self.analyzer.project:
            raise ValueError("No project analyzed yet")
        
        format_type = arguments.get("format", "json")
        output_path = arguments.get("output_path", f"output.{format_type}")
        
        from .generators import get_generator
        
        generator = get_generator(format_type)
        self.analyzer.generate_output(generator, output_path)
        
        return {
            "status": "success",
            "format": format_type,
            "output_path": output_path
        }
    
    def _find_target_object(self, target: str) -> Optional[Any]:
        """Find target object in the project."""
        if not self.analyzer or not self.analyzer.project:
            return None
        
        parts = target.split('.')
        if len(parts) == 1:
            # Module name
            for module in self.analyzer.project.modules:
                if module.name == parts[0]:
                    return module
        elif len(parts) == 2:
            # Module.Class or Module.Function
            module_name, item_name = parts
            for module in self.analyzer.project.modules:
                if module.name == module_name:
                    for cls in module.classes:
                        if cls.name == item_name:
                            return cls
                    for func in module.functions:
                        if func.name == item_name:
                            return func
        
        return None
    
    def start(self) -> None:
        """Start the MCP server."""
        logger.info(f"Starting MCP server on port {self.port}")
        
        # Run the server
        import asyncio
        
        async def run_server():
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
        
        asyncio.run(run_server())
    
    def stop(self) -> None:
        """Stop the MCP server."""
        logger.info("Stopping MCP server")


def create_server(port: int = 8080) -> MCPServer:
    """Create and return an MCP server instance."""
    return MCPServer(port)


if __name__ == "__main__":
    # Run server directly
    server = create_server()
    server.start()
