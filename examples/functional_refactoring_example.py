"""
Example Functional Refactoring for nlp2cmd generation module.

This shows how to split the 1202-line template_generator.py (100 methods)
into functional domains.
"""

# ============================================================================
# BEFORE: Monolithic template_generator.py (1202 lines, 100 methods)
# ============================================================================

class TemplateGenerator:
    """Original - handles EVERYTHING: loading, matching, rendering, shell, docker, sql..."""
    
    def __init__(self, ...):
        self._load_templates_from_json()
        self._load_defaults_from_json()
    
    def generate(self, intent: str, entities: dict, context: dict) -> TemplateResult:
        # 100+ lines of mixed logic
        if intent.startswith('shell_'):
            entities = self._prepare_shell_entities(intent, entities, context)
        elif intent.startswith('docker_'):
            entities = self._prepare_docker_entities(intent, entities)
        elif intent.startswith('sql_'):
            entities = self._prepare_sql_entities(intent, entities)
        # ... 10+ more domains
        
        template = self._find_alternative_template(intent, entities)
        rendered = self._render_template(template, entities)
        return TemplateResult(rendered)
    
    # 100 private methods for different domains...
    def _prepare_shell_entities(self, ...): ...
    def _prepare_docker_entities(self, ...): ...
    def _prepare_sql_entities(self, ...): ...
    def _prepare_kubernetes_entities(self, ...): ...
    def _apply_shell_find_flags(self, ...): ...
    def _build_shell_find_name_flag(self, ...): ...
    def _build_shell_find_size_flag(self, ...): ...
    # ... 95 more methods


# ============================================================================
# AFTER: Functional Domain Separation
# ============================================================================

# domain/command_generation/__init__.py
"""
Command Generation Domain - coordinates all sub-domains.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Protocol


@dataclass
class CommandContext:
    """Context for command generation."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    environment: Dict[str, str] = None
    history: list = None


@dataclass  
class CommandResult:
    """Result of command generation."""
    command: str
    confidence: float
    explanation: Optional[str] = None
    alternatives: list = None


# domain/command_generation/entities/preparer.py
"""
Entity Preparation - converts raw entities to command-specific format.
"""
from typing import Dict, Any, Protocol
from abc import ABC, abstractmethod


class EntityPreparer(Protocol):
    """Protocol for domain-specific entity preparation."""
    
    def supports(self, intent: str) -> bool: ...
    def prepare(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]: ...


class ShellEntityPreparer:
    """Prepares entities for shell commands."""
    
    SUPPORTED_PREFIXES = ('shell_', 'find_', 'grep_', 'awk_')
    
    def supports(self, intent: str) -> bool:
        return any(intent.startswith(p) for p in self.SUPPORTED_PREFIXES)
    
    def prepare(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(entities)
        
        # Apply shell-specific defaults
        self._apply_path_defaults(intent, entities, result, context)
        self._apply_pattern_defaults(entities, result)
        self._apply_find_flags(intent, entities, result)
        
        return result
    
    def _apply_path_defaults(self, intent: str, entities: Dict, result: Dict, context: Dict) -> None:
        """Apply shell path defaults."""
        if 'path' not in result:
            result['path'] = context.get('current_directory', '.')
    
    def _apply_pattern_defaults(self, entities: Dict, result: Dict) -> None:
        """Apply pattern defaults for file operations."""
        if 'pattern' in entities and '*' not in entities['pattern']:
            result['pattern'] = f"*{entities['pattern']}*"
    
    def _apply_find_flags(self, intent: str, entities: Dict, result: Dict) -> None:
        """Build find command flags."""
        flags = []
        
        if 'target_type' in entities:
            flags.append(f"-type {entities['target_type']}")
        
        if 'pattern' in entities:
            flags.append(f"-name '{entities['pattern']}'")
            
        if 'size' in entities:
            flags.append(f"-size {entities['size']}")
            
        if 'mtime' in entities:
            flags.append(f"-mtime {entities['mtime']}")
        
        result['find_flags'] = ' '.join(flags)


class DockerEntityPreparer:
    """Prepares entities for docker commands."""
    
    def supports(self, intent: str) -> bool:
        return intent.startswith('docker_')
    
    def prepare(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(entities)
        
        # Apply docker-specific logic
        if 'container' in entities and not entities['container'].startswith(('container_', 'id:')):
            result['container'] = self._resolve_container_name(entities['container'])
        
        return result
    
    def _resolve_container_name(self, name: str) -> str:
        """Resolve container name to ID if needed."""
        # Simplified - in real implementation would query docker
        return name


class SQLEntityPreparer:
    """Prepares entities for SQL commands."""
    
    def supports(self, intent: str) -> bool:
        return intent.startswith('sql_')
    
    def prepare(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(entities)
        
        # Validate and sanitize SQL entities
        if 'table' in entities:
            result['table'] = self._sanitize_identifier(entities['table'])
        
        if 'columns' in entities:
            result['columns'] = self._sanitize_columns(entities['columns'])
        
        return result
    
    def _sanitize_identifier(self, identifier: str) -> str:
        """Sanitize SQL identifier."""
        # Remove dangerous characters
        return ''.join(c for c in identifier if c.isalnum() or c == '_')
    
    def _sanitize_columns(self, columns: Any) -> str:
        """Sanitize column list."""
        if isinstance(columns, str):
            columns = [c.strip() for c in columns.split(',')]
        return ', '.join(self._sanitize_identifier(c) for c in columns)


class KubernetesEntityPreparer:
    """Prepares entities for kubernetes commands."""
    
    def supports(self, intent: str) -> bool:
        return intent.startswith(('kubectl_', 'kubernetes_', 'k8s_'))
    
    def prepare(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(entities)
        
        # Set default namespace if not specified
        if 'namespace' not in result:
            result['namespace'] = context.get('kubernetes_namespace', 'default')
        
        # Set default context
        if 'context' not in result:
            result['context'] = context.get('kubernetes_context')
        
        return result


class EntityPreparationPipeline:
    """Coordinates entity preparation across domains."""
    
    def __init__(self):
        self._preparers = [
            ShellEntityPreparer(),
            DockerEntityPreparer(),
            SQLEntityPreparer(),
            KubernetesEntityPreparer(),
            # Easy to add more domains
        ]
    
    def prepare(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Route to appropriate preparer based on intent."""
        for preparer in self._preparers:
            if preparer.supports(intent):
                return preparer.prepare(intent, entities, context)
        
        # No specific preparer found - return as-is
        return entities


# domain/command_generation/templates/loader.py
"""
Template Loading - handles template storage and retrieval.
"""
import json
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class Template:
    """Command template."""
    intent: str
    pattern: str
    template: str
    description: Optional[str] = None
    aliases: Optional[Dict[str, str]] = None


class TemplateLoader:
    """Loads templates from various sources."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        self._templates_dir = templates_dir or Path(__file__).parent / 'data'
        self._templates: Dict[str, Template] = {}
        self._defaults: Dict[str, Any] = {}
    
    def load(self) -> None:
        """Load all templates and defaults."""
        self._load_templates()
        self._load_defaults()
    
    def _load_templates(self) -> None:
        """Load templates from JSON files."""
        templates_file = self._templates_dir / 'templates.json'
        if templates_file.exists():
            data = json.loads(templates_file.read_text())
            for intent, template_data in data.items():
                self._templates[intent] = Template(
                    intent=intent,
                    pattern=template_data.get('pattern', ''),
                    template=template_data.get('template', ''),
                    description=template_data.get('description'),
                    aliases=template_data.get('aliases')
                )
    
    def _load_defaults(self) -> None:
        """Load default values from JSON."""
        defaults_file = self._templates_dir / 'defaults.json'
        if defaults_file.exists():
            self._defaults = json.loads(defaults_file.read_text())
    
    def get_template(self, intent: str) -> Optional[Template]:
        """Get template by intent."""
        return self._templates.get(intent)
    
    def get_default(self, key: str, fallback: Any = None) -> Any:
        """Get default value."""
        return self._defaults.get(key, fallback)
    
    def find_alternative_template(self, intent: str, entities: Dict[str, Any]) -> Optional[Template]:
        """Find alternative template based on entities."""
        # Simplified - real implementation would use fuzzy matching
        return self._templates.get(intent)


# domain/command_generation/templates/renderer.py
"""
Template Rendering - handles template string interpolation.
"""
import re
from typing import Dict, Any, Optional
from string import Template as StringTemplate


class TemplateRenderer:
    """Renders templates with entity substitution."""
    
    def render(self, template: str, entities: Dict[str, Any]) -> str:
        """Render template with entities."""
        # Handle simple ${variable} substitution
        try:
            t = StringTemplate(template)
            return t.safe_substitute(entities)
        except Exception:
            # Fallback to manual substitution
            return self._manual_render(template, entities)
    
    def _manual_render(self, template: str, entities: Dict[str, Any]) -> str:
        """Manual template rendering for complex cases."""
        result = template
        for key, value in entities.items():
            placeholder = f"${{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result
    
    def render_with_conditionals(self, template: str, entities: Dict[str, Any]) -> str:
        """Render template with conditional blocks."""
        # Handle ${if:condition}...${endif} blocks
        result = template
        
        # Simple regex-based conditional removal
        pattern = r'\$\{if:(\w+)\}(.*?)\$\{endif\}'
        
        def replace_conditional(match):
            condition_var = match.group(1)
            content = match.group(2)
            
            if entities.get(condition_var):
                return content
            return ''
        
        result = re.sub(pattern, replace_conditional, result, flags=re.DOTALL)
        
        # Now do regular substitution
        return self.render(result, entities)


# domain/command_generation/generator.py
"""
Main Command Generator - orchestrates the generation process.
"""
from typing import Optional
from .entities.preparer import EntityPreparationPipeline
from .templates.loader import TemplateLoader
from .templates.renderer import TemplateRenderer


class CommandGenerator:
    """Generates commands from natural language intents."""
    
    def __init__(
        self,
        template_loader: Optional[TemplateLoader] = None,
        entity_preparer: Optional[EntityPreparationPipeline] = None,
        template_renderer: Optional[TemplateRenderer] = None,
    ):
        self._template_loader = template_loader or TemplateLoader()
        self._entity_preparer = entity_preparer or EntityPreparationPipeline()
        self._template_renderer = template_renderer or TemplateRenderer()
        
        # Load templates on init
        self._template_loader.load()
    
    def generate(
        self,
        intent: str,
        entities: dict,
        context: Optional[dict] = None
    ) -> CommandResult:
        """
        Generate command from intent and entities.
        
        Args:
            intent: The command intent (e.g., 'shell_find_files')
            entities: Extracted entities from NL query
            context: Optional execution context
            
        Returns:
            CommandResult with generated command
        """
        context = context or {}
        
        # Step 1: Prepare entities for the domain
        prepared_entities = self._entity_preparer.prepare(intent, entities, context)
        
        # Step 2: Load appropriate template
        template = self._template_loader.find_alternative_template(intent, prepared_entities)
        if not template:
            return CommandResult(
                command="",
                confidence=0.0,
                explanation=f"No template found for intent: {intent}"
            )
        
        # Step 3: Render template with entities
        command = self._template_renderer.render_with_conditionals(
            template.template,
            prepared_entities
        )
        
        return CommandResult(
            command=command,
            confidence=0.9,  # Could be calculated from template match quality
            explanation=template.description
        )


# infrastructure/caching/evolutionary_cache.py
"""
Evolutionary Cache - stores and evolves command templates based on usage.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path


@dataclass
class CacheEntry:
    """Single cache entry with evolution metadata."""
    intent: str
    entities: Dict[str, Any]
    result: str
    success_count: int = 0
    failure_count: int = 0
    last_used: Optional[datetime] = None
    evolution_score: float = 0.0


class EvolutionaryCache:
    """
    Cache that evolves based on usage patterns.
    
    Unlike simple LRU cache, this tracks success/failure rates
    and prioritizes entries that work well.
    """
    
    def __init__(self, cache_file: Optional[Path] = None, max_size: int = 1000):
        self._cache_file = cache_file
        self._max_size = max_size
        self._entries: Dict[str, CacheEntry] = {}
        self._load()
    
    def _load(self) -> None:
        """Load cache from disk."""
        if self._cache_file and self._cache_file.exists():
            try:
                data = json.loads(self._cache_file.read_text())
                for key, entry_data in data.items():
                    self._entries[key] = CacheEntry(**entry_data)
            except Exception:
                pass  # Start with empty cache
    
    def _save(self) -> None:
        """Save cache to disk."""
        if self._cache_file:
            data = {k: asdict(v) for k, v in self._entries.items()}
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(json.dumps(data, default=str, indent=2))
    
    def get(self, intent: str, entities: Dict[str, Any]) -> Optional[str]:
        """Get cached result if available."""
        key = self._make_key(intent, entities)
        entry = self._entries.get(key)
        
        if entry:
            entry.last_used = datetime.now()
            return entry.result
        return None
    
    def put(self, intent: str, entities: Dict[str, Any], result: str) -> None:
        """Store result in cache."""
        key = self._make_key(intent, entities)
        
        if key in self._entries:
            self._entries[key].result = result
            self._entries[key].last_used = datetime.now()
        else:
            # Evict if at capacity
            if len(self._entries) >= self._max_size:
                self._evict_worst()
            
            self._entries[key] = CacheEntry(
                intent=intent,
                entities=entities,
                result=result,
                last_used=datetime.now()
            )
        
        self._save()
    
    def report_success(self, intent: str, entities: Dict[str, Any]) -> None:
        """Report successful command execution."""
        key = self._make_key(intent, entities)
        if key in self._entries:
            self._entries[key].success_count += 1
            self._entries[key].evolution_score = self._calculate_score(
                self._entries[key]
            )
            self._save()
    
    def report_failure(self, intent: str, entities: Dict[str, Any]) -> None:
        """Report failed command execution."""
        key = self._make_key(intent, entities)
        if key in self._entries:
            self._entries[key].failure_count += 1
            self._entries[key].evolution_score = self._calculate_score(
                self._entries[key]
            )
            self._save()
    
    def _make_key(self, intent: str, entities: Dict[str, Any]) -> str:
        """Create cache key from intent and entities."""
        entity_str = json.dumps(entities, sort_keys=True)
        return f"{intent}:{hash(entity_str)}"
    
    def _calculate_score(self, entry: CacheEntry) -> float:
        """Calculate evolution score based on success rate and recency."""
        total = entry.success_count + entry.failure_count
        if total == 0:
            return 0.5  # Neutral score for new entries
        
        success_rate = entry.success_count / total
        
        # Boost recently used entries
        recency_boost = 0.0
        if entry.last_used:
            days_since = (datetime.now() - entry.last_used).days
            recency_boost = max(0, 0.1 - days_since * 0.01)
        
        return success_rate + recency_boost
    
    def _evict_worst(self) -> None:
        """Remove lowest-scored entry when cache is full."""
        if not self._entries:
            return
        
        worst_key = min(self._entries.keys(),
                       key=lambda k: self._entries[k].evolution_score)
        del self._entries[worst_key]


# interfaces/cli/generate_command.py
"""
CLI interface for command generation.
"""
import click
from domain.command_generation.generator import CommandGenerator, CommandContext
from infrastructure.caching.evolutionary_cache import EvolutionaryCache
from pathlib import Path


@click.command()
@click.argument('query')
@click.option('--intent', help='Force specific intent')
@click.option('--dry-run', is_flag=True, help='Show command without executing')
@click.option('--cache-dir', default='~/.nlp2cmd/cache', help='Cache directory')
def generate(query: str, intent: Optional[str], dry_run: bool, cache_dir: str):
    """Generate command from natural language query."""
    
    # Initialize components
    cache = EvolutionaryCache(cache_file=Path(cache_dir) / 'commands.json')
    generator = CommandGenerator()
    
    # Check cache first
    if not intent:
        cached = cache.get(query, {})
        if cached:
            click.echo(f"Cached: {cached}")
            return
    
    # Generate command
    context = CommandContext(
        environment=dict(os.environ),
    )
    
    result = generator.generate(
        intent=intent or 'auto',
        entities={'query': query},
        context=context
    )
    
    if result.command:
        click.echo(result.command)
        
        if not dry_run:
            # Store in cache for future
            cache.put(query, {}, result.command)
    else:
        click.echo(f"Error: {result.explanation}", err=True)


# ============================================================================
# BENEFITS OF FUNCTIONAL SEPARATION
# ============================================================================

"""
BEFORE (Monolithic):
- template_generator.py: 1202 lines, 100 methods
- evolutionary_cache.py: 1048 lines (separate but tightly coupled)
- fuzzy_schema_matcher.py: 560 lines
- semantic_matcher_optimized.py: 750 lines

AFTER (Functional Domains):
- domain/command_generation/
  - __init__.py: 50 lines (domain exports)
  - generator.py: 100 lines (orchestration)
  - entities/
    - preparer.py: 200 lines (entity preparation)
    - shell_preparer.py: 150 lines (shell-specific)
    - docker_preparer.py: 80 lines (docker-specific)
  - templates/
    - loader.py: 100 lines (template loading)
    - renderer.py: 80 lines (template rendering)

- infrastructure/caching/
  - evolutionary_cache.py: 150 lines (caching logic)

- interfaces/cli/
  - generate_command.py: 60 lines (CLI interface)

TOTAL: ~870 lines vs 3560 lines = 4x less code per concept

BENEFITS:
1. Separation of Concerns - each module has one responsibility
2. Testability - can test entity preparers independently
3. Extensibility - add new domain (e.g., AWS) by adding one preparer class
4. Readability - 150-line files vs 1200-line files
5. Reusability - preparers can be used in other contexts
6. Maintainability - changes isolated to specific domain
7. Team Scaling - different developers can own different domains
"""
