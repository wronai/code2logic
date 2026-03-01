# System Architecture Analysis

## Overview

- **Project**: /home/tom/github/wronai/nlp2cmd/src/nlp2cmd
- **Analysis Mode**: static
- **Total Functions**: 1473
- **Total Classes**: 398
- **Modules**: 197
- **Entry Points**: 1366

## Architecture by Module

### thermodynamic
- **Functions**: 44
- **Classes**: 11
- **File**: `__init__.py`

### schemas
- **Functions**: 40
- **Classes**: 2
- **File**: `__init__.py`

### web_schema.form_data_loader
- **Functions**: 38
- **Classes**: 1
- **File**: `form_data_loader.py`

### automation.step_validator
- **Functions**: 34
- **Classes**: 3
- **File**: `step_validator.py`

### validators
- **Functions**: 26
- **Classes**: 8
- **File**: `__init__.py`

### web_schema.browser_config
- **Functions**: 26
- **Classes**: 2
- **File**: `browser_config.py`

### history.tracker
- **Functions**: 26
- **Classes**: 3
- **File**: `tracker.py`

### automation.password_store
- **Functions**: 25
- **Classes**: 7
- **File**: `password_store.py`

### generation.data_loader
- **Functions**: 23
- **Classes**: 3
- **File**: `data_loader.py`

### parsing.toon_parser
- **Functions**: 23
- **Classes**: 3
- **File**: `toon_parser.py`

### cli.commands.doctor
- **Functions**: 23
- **Classes**: 3
- **File**: `doctor.py`

### automation.vector_store
- **Functions**: 22
- **Classes**: 2
- **File**: `vector_store.py`

### automation.mouse_controller
- **Functions**: 22
- **Classes**: 2
- **File**: `mouse_controller.py`

### generation.thermodynamic_components
- **Functions**: 22
- **Classes**: 5
- **File**: `thermodynamic_components.py`

### core.toon_integration
- **Functions**: 22
- **Classes**: 1
- **File**: `toon_integration.py`

### web_schema.history
- **Functions**: 22
- **Classes**: 2
- **File**: `history.py`

### thermodynamic.energy_models
- **Functions**: 20
- **Classes**: 6
- **File**: `energy_models.py`

### adapters.shell_generators
- **Functions**: 20
- **Classes**: 8
- **File**: `shell_generators.py`

### generation.semantic_matcher_optimized
- **Functions**: 19
- **Classes**: 3
- **File**: `semantic_matcher_optimized.py`

### cli.cache
- **Functions**: 19
- **Classes**: 8
- **File**: `cache.py`

## Key Entry Points

Main execution flows into the system:

### pipeline_runner_plans.PlanExecutionMixin.execute_action_plan
> Execute an ActionPlan step by step using Playwright.

Args:
    plan: ActionPlan instance with steps to execute
    dry_run: If True, only show the pl
- **Calls**: Console, frozenset, console.print, console.print, enumerate, None.strip, RunnerResult, any

### pipeline_runner_plans.execute_action_plan
> Execute an ActionPlan step by step using Playwright.

Args:
    plan: ActionPlan instance with steps to execute
    dry_run: If True, only show the pl
- **Calls**: Console, frozenset, console.print, console.print, enumerate, None.strip, RunnerResult, any

### cli.commands.run.handle_run_mode
> Handle --run option: generate and execute command with error recovery.

Features:
- Generate command from natural language
- Execute with real-time ou
- **Calls**: _verbose_log, _verbose_log, _verbose_log, FormDataLoader, loader.get_nlp_keywords, loader.get_nlp_keywords, FormDataLoader.dedupe_selectors, loader.get_nlp_keywords

### adapters.canvas.CanvasAdapter.execute_drawing_plan
> Execute a canvas drawing plan on a Playwright page.

IMPROVED: Added detailed diagnostic logging for each step.
- **Calls**: plan.get, MouseController, enumerate, json.loads, step.get, step.get, pipeline_runner_utils._MarkdownConsoleWrapper.print, pipeline_runner_utils._MarkdownConsoleWrapper.print

### adapters.canvas.execute_drawing_plan
> Execute a canvas drawing plan on a Playwright page.

IMPROVED: Added detailed diagnostic logging for each step.
- **Calls**: plan.get, MouseController, enumerate, json.loads, step.get, step.get, pipeline_runner_utils._MarkdownConsoleWrapper.print, pipeline_runner_utils._MarkdownConsoleWrapper.print

### execution.runner.ExecutionRunner.run_command
> Execute a shell command with real-time output.

Args:
    command: Shell command to execute
    cwd: Working directory
    env: Environment variables

- **Calls**: time.time, self.print_markdown_block, ExecutionResult, self.execution_history.append, subprocess.Popen, None.join, None.join, subprocess.run

### execution.runner.run_command
> Execute a shell command with real-time output.

Args:
    command: Shell command to execute
    cwd: Working directory
    env: Environment variables

- **Calls**: time.time, self.print_markdown_block, ExecutionResult, self.execution_history.append, subprocess.Popen, None.join, None.join, subprocess.run

### web_schema.form_handler.FormHandler.detect_form_fields
> Detect all form fields on a page.

Args:
    page: Playwright page object

Returns:
    List of FormField objects
- **Calls**: page.query_selector_all, self._print_yaml, page.query_selector_all, self._print_yaml, page.query_selector_all, self._print_yaml, page.query_selector_all, self._print_yaml

### web_schema.form_handler.detect_form_fields
> Detect all form fields on a page.

Args:
    page: Playwright page object

Returns:
    List of FormField objects
- **Calls**: page.query_selector_all, self._print_yaml, page.query_selector_all, self._print_yaml, page.query_selector_all, self._print_yaml, page.query_selector_all, self._print_yaml

### web_schema.site_explorer.SiteExplorer.find_form
> Find a form on the website matching the intent.

Args:
    url: Starting URL (homepage)
    intent: Type of form to find (contact, search, newsletter,
- **Calls**: time.perf_counter, set, self._find_best_form_candidate, ExplorationResult, None.start, p.chromium.launch, browser.new_context, context.new_page

### web_schema.site_explorer.find_form
> Find a form on the website matching the intent.

Args:
    url: Starting URL (homepage)
    intent: Type of form to find (contact, search, newsletter,
- **Calls**: time.perf_counter, set, self._find_best_form_candidate, ExplorationResult, None.start, p.chromium.launch, browser.new_context, context.new_page

### adapters.browser.BrowserAdapter.generate
- **Calls**: str, _debug, isinstance, _debug, self._has_fill_form_action, self._should_explore_for_forms, self._should_explore_for_content, self._has_type_action

### adapters.browser.generate
- **Calls**: str, _debug, isinstance, _debug, self._has_fill_form_action, self._should_explore_for_forms, self._should_explore_for_content, self._has_type_action

### web_schema.site_explorer.SiteExplorer.find_content
> Find content on the website (articles, products, docs, etc.).

Args:
    url: Starting URL (homepage)
    content_type: Type of content to find (artic
- **Calls**: time.perf_counter, self._resolve_platform_url, self._try_github_api, set, self._explore_recursive, _debug, self._find_best_content_candidate, ExplorationResult

### web_schema.site_explorer.find_content
> Find content on the website (articles, products, docs, etc.).

Args:
    url: Starting URL (homepage)
    content_type: Type of content to find (artic
- **Calls**: time.perf_counter, self._resolve_platform_url, self._try_github_api, set, self._explore_recursive, _debug, self._find_best_content_candidate, ExplorationResult

### generation.evolutionary_cache.EvolutionaryCache.lookup
> 4-tier lookup: cache → template → regex → LLM teacher.
Returns LookupResult with command and timing.
- **Calls**: time.perf_counter, generation.evolutionary_cache.fingerprint, generation.evolutionary_cache.fuzzy_fingerprint, LookupResult, self.stats.get, None.lower, None.isoformat, self.save

### generation.evolutionary_cache.lookup
> 4-tier lookup: cache → template → regex → LLM teacher.
Returns LookupResult with command and timing.
- **Calls**: time.perf_counter, generation.evolutionary_cache.fingerprint, generation.evolutionary_cache.fuzzy_fingerprint, LookupResult, self.stats.get, None.lower, None.isoformat, self.save

### validators.DockerValidator.validate
> Validate Docker command or Dockerfile.
- **Calls**: None.strip, content_stripped.lower, content_stripped.split, enumerate, content_lower.startswith, content_lower.startswith, self._iter_publish_ports, content_lower.startswith

### feedback.FeedbackAnalyzer.analyze
> Analyze transformation result and generate feedback.

Args:
    original_input: Original natural language input
    generated_output: Generated comman
- **Calls**: list, list, isinstance, str, output_str.strip, self._calculate_confidence, isinstance, FeedbackResult

### feedback.analyze
> Analyze transformation result and generate feedback.

Args:
    original_input: Original natural language input
    generated_output: Generated comman
- **Calls**: list, list, isinstance, str, output_str.strip, self._calculate_confidence, isinstance, FeedbackResult

### schema_extraction.script_extractors.ShellScriptExtractor.extract_from_source
- **Calls**: source_code.splitlines, None.join, self._re_getopts.finditer, set, self._re_long_opt_value.finditer, sorted, self._re_short_opt.finditer, _dedupe_params

### storage.versioned_store.demonstrate_version_management
> Demonstrate version management for command schemas.
- **Calls**: pipeline_runner_utils._MarkdownConsoleWrapper.print, pipeline_runner_utils._MarkdownConsoleWrapper.print, pipeline_runner_utils._MarkdownConsoleWrapper.print, VersionedSchemaStore, ExtractedSchema, ExtractedSchema, pipeline_runner_utils._MarkdownConsoleWrapper.print, store.store_schema_version

### execution.runner.ExecutionRunner.run_with_recovery
> Execute command with automatic error recovery and resource discovery.

When a command fails due to missing resources (files, directories,
endpoints), 
- **Calls**: range, LLMValidator, LLMRepair, self.run_command, attempts.append, exploration.resource_discovery.get_resource_discovery_manager, self.confirm_execution, ExecutionResult

### service.cli.add_service_command
> Add service command to the main CLI group.
- **Calls**: main_group.command, click.option, click.option, click.option, click.option, click.option, click.option, click.option

### generation.pipeline.RuleBasedPipeline.process
> Process natural language text through the pipeline.

Args:
    text: Natural language input
    
Returns:
    PipelineResult with generated command an
- **Calls**: time.time, PipelineResult, None.lower, self.metrics.record_result, any, bool, bool, any

### validators.KubernetesValidator.validate
> Validate kubectl command.
- **Calls**: None.strip, content_stripped.lower, content_lower.startswith, content_stripped.split, enumerate, ValidationResult, ValidationResult, t.lower

### utils.yaml_compat.safe_load
> Parse a minimal subset of YAML into Python objects.
- **Calls**: None.splitlines, raw.lstrip, stripped.startswith, cleaned.append, _Frame, content.startswith, content.split, key.strip

### schema_extraction.script_extractors.MakefileExtractor.extract_from_source
- **Calls**: source_code.splitlines, None.join, set, ExtractedSchema, line.strip, s.startswith, self._re_var.match, self._re_phony.match

### schema_extraction.script_extractors.extract_from_source
- **Calls**: source_code.splitlines, None.join, set, ExtractedSchema, line.strip, s.startswith, self._re_var.match, self._re_phony.match

### cli.history.show_stats
> Show command execution statistics.
- **Calls**: history_group.command, click.option, history.tracker.get_global_history, history.get_schema_usage_stats, console.print, console.print, console.print, history.get_stats

## Process Flows

Key execution flows identified:

### Flow 1: execute_action_plan
```
execute_action_plan
```

### Flow 2: execute_action_plan
```
execute_action_plan
```

### Flow 3: handle_run_mode
```
handle_run_mode
```

### Flow 4: execute_drawing_plan
```
execute_drawing_plan
```

### Flow 5: execute_drawing_plan
```
execute_drawing_plan
```

### Flow 6: run_command
```
run_command
```

### Flow 7: run_command
```
run_command
```

### Flow 8: detect_form_fields
```
detect_form_fields
```

### Flow 9: detect_form_fields
```
detect_form_fields
```

### Flow 10: find_form
```
find_form
```

## Key Classes

### generation.template_generator.TemplateGenerator
> Generate DSL commands from templates.

Uses predefined templates filled with extracted entities.
Fal
- **Methods**: 100
- **Key Methods**: generation.template_generator.TemplateGenerator.__init__, generation.template_generator.TemplateGenerator._load_defaults_from_json, generation.template_generator.TemplateGenerator._load_templates_from_json, generation.template_generator.TemplateGenerator._get_default, generation.template_generator.TemplateGenerator.generate, generation.template_generator.TemplateGenerator._find_alternative_template, generation.template_generator.TemplateGenerator._get_intent_aliases, generation.template_generator.TemplateGenerator._prepare_entities, generation.template_generator.TemplateGenerator._prepare_sql_entities, generation.template_generator.TemplateGenerator._prepare_shell_entities

### web_schema.form_data_loader.FormDataLoader
> Loads form field data from multiple sources:
1. .env file (for sensitive data like email, name, phon
- **Methods**: 45
- **Key Methods**: web_schema.form_data_loader.FormDataLoader.__init__, web_schema.form_data_loader.FormDataLoader._dedupe_preserve_order, web_schema.form_data_loader.FormDataLoader.dedupe_selectors, web_schema.form_data_loader.FormDataLoader._parse_domain, web_schema.form_data_loader.FormDataLoader._safe_domain_filename, web_schema.form_data_loader.FormDataLoader._user_sites_dir, web_schema.form_data_loader.FormDataLoader._project_sites_dir, web_schema.form_data_loader.FormDataLoader._site_profile_paths, web_schema.form_data_loader.FormDataLoader.get_site_profile_write_path, web_schema.form_data_loader.FormDataLoader._load_site_profile_payload

### schemas.SchemaRegistry
> Registry for file format schemas with validation and repair capabilities.
- **Methods**: 37
- **Key Methods**: schemas.SchemaRegistry.__init__, schemas.SchemaRegistry._register_builtin_schemas, schemas.SchemaRegistry.register, schemas.SchemaRegistry.get, schemas.SchemaRegistry.has_schema, schemas.SchemaRegistry.list_schemas, schemas.SchemaRegistry.unregister, schemas.SchemaRegistry.find_schema_for_file, schemas.SchemaRegistry.find_schema_by_mime_type, schemas.SchemaRegistry.find_extension_conflicts

### core.toon_integration.ToonDataManager
> Unified data manager using TOON format
- **Methods**: 27
- **Key Methods**: core.toon_integration.ToonDataManager.__init__, core.toon_integration.ToonDataManager._ensure_loaded, core.toon_integration.ToonDataManager.get_all_commands, core.toon_integration.ToonDataManager.get_shell_commands, core.toon_integration.ToonDataManager.get_browser_commands, core.toon_integration.ToonDataManager.get_command_by_name, core.toon_integration.ToonDataManager.search_commands, core.toon_integration.ToonDataManager.get_config, core.toon_integration.ToonDataManager.get_llm_config, core.toon_integration.ToonDataManager.get_test_commands

### web_schema.site_explorer.SiteExplorer
> Explores website to find forms, contact pages, and other content.

Usage:
    explorer = SiteExplore
- **Methods**: 27
- **Key Methods**: web_schema.site_explorer.SiteExplorer.__init__, web_schema.site_explorer.SiteExplorer._setup_resource_blocking, web_schema.site_explorer.SiteExplorer._resolve_platform_url, web_schema.site_explorer.SiteExplorer._goto_with_retry, web_schema.site_explorer.SiteExplorer._try_github_api, web_schema.site_explorer.SiteExplorer._detect_docs_framework, web_schema.site_explorer.SiteExplorer._record_timing, web_schema.site_explorer.SiteExplorer.get_timing_stats, web_schema.site_explorer.SiteExplorer._fallback_static_scrape, web_schema.site_explorer.SiteExplorer.find_content

### core.core_transform.NLP2CMD
> Main class for Natural Language to Command transformation.

This class orchestrates the transformati
- **Methods**: 23
- **Key Methods**: core.core_transform.NLP2CMD.__init__, core.core_transform.NLP2CMD.transform, core.core_transform.NLP2CMD.transform_ir, core.core_transform.NLP2CMD._normalize_entities, core.core_transform.NLP2CMD._normalize_entities_sql, core.core_transform.NLP2CMD._normalize_entities_shell, core.core_transform.NLP2CMD._normalize_entities_docker, core.core_transform.NLP2CMD._normalize_entities_kubernetes, core.core_transform.NLP2CMD._normalize_entities_dql, core.core_transform.NLP2CMD._normalize_shell_entities

### adapters.browser.BrowserAdapter
> Minimal adapter that turns NL into dom_dql.v1 navigation (Playwright).
- **Methods**: 22
- **Key Methods**: adapters.browser.BrowserAdapter.__init__, adapters.browser.BrowserAdapter._extract_url, adapters.browser.BrowserAdapter._extract_type_text, adapters.browser.BrowserAdapter._has_type_action, adapters.browser.BrowserAdapter._should_explore_for_content, adapters.browser.BrowserAdapter._should_explore_for_forms, adapters.browser.BrowserAdapter._has_fill_form_action, adapters.browser.BrowserAdapter._has_press_enter, adapters.browser.BrowserAdapter._has_form_action, adapters.browser.BrowserAdapter._has_submit_action
- **Inherits**: BaseDSLAdapter

### adapters.kubernetes.KubernetesAdapter
> Kubernetes adapter for kubectl commands and manifests.

Transforms natural language into kubectl com
- **Methods**: 22
- **Key Methods**: adapters.kubernetes.KubernetesAdapter.__init__, adapters.kubernetes.KubernetesAdapter._parse_cluster_context, adapters.kubernetes.KubernetesAdapter._normalize_resource, adapters.kubernetes.KubernetesAdapter.generate, adapters.kubernetes.KubernetesAdapter._generate_get, adapters.kubernetes.KubernetesAdapter._generate_describe, adapters.kubernetes.KubernetesAdapter._generate_apply, adapters.kubernetes.KubernetesAdapter._generate_delete, adapters.kubernetes.KubernetesAdapter._generate_scale, adapters.kubernetes.KubernetesAdapter._generate_logs
- **Inherits**: BaseDSLAdapter

### automation.action_planner.ActionPlanner
> Decomposes complex NL commands into ActionPlan via rules or LLM.

Costs:
- Rule match (known service
- **Methods**: 21
- **Key Methods**: automation.action_planner.ActionPlanner.__init__, automation.action_planner.ActionPlanner.decompose, automation.action_planner.ActionPlanner.decompose_sync, automation.action_planner.ActionPlanner._try_rule_decomposition, automation.action_planner.ActionPlanner._resolve_service, automation.action_planner.ActionPlanner._wants_new_tab, automation.action_planner.ActionPlanner._wants_existing_firefox, automation.action_planner.ActionPlanner._wants_create_key, automation.action_planner.ActionPlanner._build_navigation_steps, automation.action_planner.ActionPlanner._build_session_check_steps

### generation.semantic_matcher_optimized.OptimizedSemanticMatcher
> Optimized semantic similarity matcher using sentence embeddings.

Features:
- Handles typos and para
- **Methods**: 20
- **Key Methods**: generation.semantic_matcher_optimized.OptimizedSemanticMatcher.__init__, generation.semantic_matcher_optimized.OptimizedSemanticMatcher._preload_models, generation.semantic_matcher_optimized.OptimizedSemanticMatcher._get_model, generation.semantic_matcher_optimized.OptimizedSemanticMatcher._get_polish_model, generation.semantic_matcher_optimized.OptimizedSemanticMatcher._load_model, generation.semantic_matcher_optimized.OptimizedSemanticMatcher.add_intent, generation.semantic_matcher_optimized.OptimizedSemanticMatcher.add_intents_batch, generation.semantic_matcher_optimized.OptimizedSemanticMatcher._encode_text, generation.semantic_matcher_optimized.OptimizedSemanticMatcher._encode_batch, generation.semantic_matcher_optimized.OptimizedSemanticMatcher._encode_with_cache

### generation.evolutionary_cache.EvolutionaryCache
> Manages the .nlp2cmd/ learned schema cache.

Usage:
    cache = EvolutionaryCache()
    result = cac
- **Methods**: 20
- **Key Methods**: generation.evolutionary_cache.EvolutionaryCache.__init__, generation.evolutionary_cache.EvolutionaryCache._ensure_dir, generation.evolutionary_cache.EvolutionaryCache._load, generation.evolutionary_cache.EvolutionaryCache.save, generation.evolutionary_cache.EvolutionaryCache.lookup, generation.evolutionary_cache.EvolutionaryCache._ask_teacher, generation.evolutionary_cache.EvolutionaryCache._clean, generation.evolutionary_cache.EvolutionaryCache._try_template_pipeline, generation.evolutionary_cache.EvolutionaryCache._try_english_pipeline, generation.evolutionary_cache.EvolutionaryCache._try_polish_template

### parsing.toon_parser.ToonParser
> Unified TOON format parser with hierarchical access
- **Methods**: 20
- **Key Methods**: parsing.toon_parser.ToonParser.__init__, parsing.toon_parser.ToonParser.parse_file, parsing.toon_parser.ToonParser.parse_content, parsing.toon_parser.ToonParser._parse_lines, parsing.toon_parser.ToonParser._parse_array_node, parsing.toon_parser.ToonParser._parse_object_node, parsing.toon_parser.ToonParser._parse_key_value, parsing.toon_parser.ToonParser._parse_value, parsing.toon_parser.ToonParser._extract_categories, parsing.toon_parser.ToonParser.get_category

### automation.step_validator.StepValidator
> Validates pre/post conditions for ActionPlan steps.

Checks clipboard state, DOM elements, environme
- **Methods**: 19
- **Key Methods**: automation.step_validator.StepValidator.__init__, automation.step_validator.StepValidator.metrics, automation.step_validator.StepValidator.start_step, automation.step_validator.StepValidator.finish_step, automation.step_validator.StepValidator.get_clipboard, automation.step_validator.StepValidator.set_clipboard, automation.step_validator.StepValidator.snapshot_clipboard, automation.step_validator.StepValidator.clipboard_changed, automation.step_validator.StepValidator.validate_pre_navigate, automation.step_validator.StepValidator.validate_pre_check_session

### automation.mouse_controller.MouseController
> Advanced mouse control via Playwright with human-like movements.

Supports:
- Click, double-click, r
- **Methods**: 19
- **Key Methods**: automation.mouse_controller.MouseController.__init__, automation.mouse_controller.MouseController._jitter, automation.mouse_controller.MouseController._human_delay, automation.mouse_controller.MouseController.click, automation.mouse_controller.MouseController.double_click, automation.mouse_controller.MouseController.right_click, automation.mouse_controller.MouseController.move_to, automation.mouse_controller.MouseController.drag, automation.mouse_controller.MouseController._compute_bezier, automation.mouse_controller.MouseController.bezier_move

### generation.fuzzy_schema_matcher.FuzzySchemaMatcher
> Language-agnostic fuzzy matcher using JSON schemas.

Works with any language by using character-leve
- **Methods**: 19
- **Key Methods**: generation.fuzzy_schema_matcher.FuzzySchemaMatcher.__init__, generation.fuzzy_schema_matcher.FuzzySchemaMatcher.load_schema, generation.fuzzy_schema_matcher.FuzzySchemaMatcher.add_phrase, generation.fuzzy_schema_matcher.FuzzySchemaMatcher.add_phrases_from_dict, generation.fuzzy_schema_matcher.FuzzySchemaMatcher._build_index, generation.fuzzy_schema_matcher.FuzzySchemaMatcher._index_phrase, generation.fuzzy_schema_matcher.FuzzySchemaMatcher._normalize, generation.fuzzy_schema_matcher.FuzzySchemaMatcher._remove_spaces, generation.fuzzy_schema_matcher.FuzzySchemaMatcher._get_ngrams, generation.fuzzy_schema_matcher.FuzzySchemaMatcher._ngram_similarity

### adapters.dynamic.DynamicAdapter
> Dynamic adapter that uses extracted schemas instead of hardcoded patterns.

This adapter can work wi
- **Methods**: 19
- **Key Methods**: adapters.dynamic.DynamicAdapter.__init__, adapters.dynamic.DynamicAdapter.check_safety, adapters.dynamic.DynamicAdapter._load_common_commands, adapters.dynamic.DynamicAdapter.register_schema_source, adapters.dynamic.DynamicAdapter.generate, adapters.dynamic.DynamicAdapter._find_matching_commands, adapters.dynamic.DynamicAdapter._generate_from_schema, adapters.dynamic.DynamicAdapter._generate_make_command, adapters.dynamic.DynamicAdapter._generate_web_dql, adapters.dynamic.DynamicAdapter._generate_from_template
- **Inherits**: BaseDSLAdapter

### adapters.desktop.DesktopAdapter
> Adapter for desktop GUI automation via VNC/noVNC + xdotool/wmctrl.
- **Methods**: 19
- **Key Methods**: adapters.desktop.DesktopAdapter.__init__, adapters.desktop.DesktopAdapter.generate, adapters.desktop.DesktopAdapter._build_actions, adapters.desktop.DesktopAdapter._build_email_actions, adapters.desktop.DesktopAdapter._build_email_compose, adapters.desktop.DesktopAdapter._detect_followup_actions, adapters.desktop.DesktopAdapter.detect_intent, adapters.desktop.DesktopAdapter._extract_app_name, adapters.desktop.DesktopAdapter._extract_quoted_text, adapters.desktop.DesktopAdapter._extract_shortcut
- **Inherits**: BaseAdapter

### generation.pipeline.RuleBasedPipeline
> Complete rule-based NL → DSL pipeline.

Combines intent detection, entity extraction, and template g
- **Methods**: 18
- **Key Methods**: generation.pipeline.RuleBasedPipeline.__init__, generation.pipeline.RuleBasedPipeline.complex_detector, generation.pipeline.RuleBasedPipeline.action_planner, generation.pipeline.RuleBasedPipeline.evolutionary_cache, generation.pipeline.RuleBasedPipeline.enhanced_detector, generation.pipeline.RuleBasedPipeline.process, generation.pipeline.RuleBasedPipeline.process_steps, generation.pipeline.RuleBasedPipeline._process_with_detection, generation.pipeline.RuleBasedPipeline._split_sentences, generation.pipeline.RuleBasedPipeline._persist_result

### web_schema.browser_config.BrowserConfigLoader
> Single source of truth for browser automation config.

Loads from ``data/browser_config/*.yaml`` wit
- **Methods**: 18
- **Key Methods**: web_schema.browser_config.BrowserConfigLoader.__init__, web_schema.browser_config.BrowserConfigLoader._ensure_loaded, web_schema.browser_config.BrowserConfigLoader.get_dismiss_selectors, web_schema.browser_config.BrowserConfigLoader.get_submit_selectors, web_schema.browser_config.BrowserConfigLoader.get_type_selectors, web_schema.browser_config.BrowserConfigLoader.get_contact_page_link_selectors, web_schema.browser_config.BrowserConfigLoader.get_common_contact_paths, web_schema.browser_config.BrowserConfigLoader.get_contact_url_keywords, web_schema.browser_config.BrowserConfigLoader.get_contact_page_keywords, web_schema.browser_config.BrowserConfigLoader.get_junk_field_types

### adapters.docker.DockerAdapter
> Docker adapter for CLI and Compose operations.

Transforms natural language into Docker commands
wit
- **Methods**: 18
- **Key Methods**: adapters.docker.DockerAdapter.__init__, adapters.docker.DockerAdapter._parse_compose_context, adapters.docker.DockerAdapter.generate, adapters.docker.DockerAdapter._generate_run, adapters.docker.DockerAdapter._generate_stop, adapters.docker.DockerAdapter._generate_remove, adapters.docker.DockerAdapter._generate_build, adapters.docker.DockerAdapter._generate_pull, adapters.docker.DockerAdapter._generate_compose_up, adapters.docker.DockerAdapter._generate_compose_down
- **Inherits**: BaseDSLAdapter

## Data Transformation Functions

Key functions that process and transform data:

### schema_driven.SchemaDrivenNLP2CMD.transform
- **Output to**: self._select_action, self._extract_params, self._render_dsl, str, ActionIR

### schema_driven.transform
- **Output to**: self._select_action, self._extract_params, self._render_dsl, str, ActionIR

### monitoring.resources.ResourceMonitor.format_metrics
> Format metrics for display.
- **Output to**: None.join, lines.append

### monitoring.resources.format_last_metrics
> Format metrics from last execution for display.
- **Output to**: get_last_metrics, _monitor.format_metrics

### monitoring.resources.format_metrics
> Format metrics for display.
- **Output to**: None.join, lines.append

### monitoring.resources._PsutilStub.Process
- **Output to**: _ProcessStub, _MemInfo, _CpuTimes

### monitoring.resources.Process
- **Output to**: _ProcessStub, _MemInfo, _CpuTimes

### monitoring.token_costs.TokenCostEstimator.format_estimate
> Format token cost estimate for display.
- **Output to**: None.join, lines.append

### monitoring.token_costs.parse_metrics_string
> Parse metrics string like '⏱️ Time: 2.6ms | 💻 CPU: 0.0% | 🧠 RAM: 53.5MB (0.1%) | ⚡ Energy: 0.022mJ'
- **Output to**: None.strip, None.strip, None.strip, None.strip, float

### monitoring.token_costs.format_estimate
> Format token cost estimate for display.
- **Output to**: None.join, lines.append

### schemas.FileFormatSchema.validate
> Validate content using this schema.
- **Output to**: self.validator

### schemas.FileFormatSchema.parse
> Parse content using this schema.
- **Output to**: self.parser

### schemas.FileFormatSchema.self_validate
> Validate the schema itself.
- **Output to**: errors.append, errors.append, errors.append, errors.append, errors.append

### schemas.SchemaRegistry.validate_integrity
> Validate registry integrity.
- **Output to**: self._schemas.items, self.find_extension_conflicts, isinstance, ValueError

### schemas.SchemaRegistry.detect_format
> Detect file format from path.
- **Output to**: str, self._schemas.values, self._detect_by_content, generation.thermodynamic_components._NumpyStub.max, self._match_pattern

### schemas.SchemaRegistry.validate
> Validate content against schema.
- **Output to**: self.get, schema.validator

### schemas.validate
> Validate content against schema.
- **Output to**: self.get, schema.validator

### schemas.parse
> Parse content using this schema.
- **Output to**: self.parser

### schemas.self_validate
> Validate the schema itself.
- **Output to**: errors.append, errors.append, errors.append, errors.append, errors.append

### schemas.validate_integrity
> Validate registry integrity.
- **Output to**: self._schemas.items, self.find_extension_conflicts, isinstance, ValueError

### schemas.detect_format
> Detect file format from path.
- **Output to**: str, self._schemas.values, self._detect_by_content, generation.thermodynamic_components._NumpyStub.max, self._match_pattern

### automation.step_validator.StepValidator.validate_pre_navigate
> Validate before navigation step.
- **Output to**: params.get, ValidationResult, ValidationResult, url.startswith, ValidationResult

### automation.step_validator.StepValidator.validate_pre_check_session
> Validate before session check — page must be loaded.
- **Output to**: ValidationResult, ValidationResult, ValidationResult

### automation.step_validator.StepValidator.validate_pre_extract_key
> Validate before key extraction — must be on correct page.
- **Output to**: self.snapshot_clipboard, ValidationResult, ValidationResult, params.get, ValidationResult

### automation.step_validator.StepValidator.validate_pre_prompt_secret
> Validate before prompting for secret — check if already available.
- **Output to**: params.get, None.strip, os.environ.get, variables.items, ValidationResult

## Public API Surface

Functions exposed as public API (no underscore prefix):

- `pipeline_runner_plans.PlanExecutionMixin.execute_action_plan` - 261 calls
- `pipeline_runner_plans.execute_action_plan` - 261 calls
- `cli.commands.run.handle_run_mode` - 261 calls
- `adapters.canvas.CanvasAdapter.execute_drawing_plan` - 193 calls
- `adapters.canvas.execute_drawing_plan` - 193 calls
- `cli.main.main` - 115 calls
- `execution.runner.ExecutionRunner.run_command` - 109 calls
- `execution.runner.run_command` - 109 calls
- `generation.train_model.train_all_models` - 86 calls
- `web_schema.form_handler.FormHandler.detect_form_fields` - 83 calls
- `web_schema.form_handler.detect_form_fields` - 83 calls
- `web_schema.site_explorer.SiteExplorer.find_form` - 77 calls
- `web_schema.site_explorer.find_form` - 77 calls
- `adapters.browser.BrowserAdapter.generate` - 66 calls
- `adapters.browser.generate` - 66 calls
- `cli.commands.generate.handle_generate_query` - 66 calls
- `web_schema.site_explorer.SiteExplorer.find_content` - 60 calls
- `web_schema.site_explorer.find_content` - 60 calls
- `generation.evolutionary_cache.EvolutionaryCache.lookup` - 57 calls
- `generation.evolutionary_cache.lookup` - 57 calls
- `cli.debug_info.show_schema_info` - 57 calls
- `cli.debug_info.show_decision_tree_info` - 56 calls
- `validators.DockerValidator.validate` - 52 calls
- `feedback.FeedbackAnalyzer.analyze` - 51 calls
- `feedback.analyze` - 51 calls
- `schema_extraction.script_extractors.ShellScriptExtractor.extract_from_source` - 51 calls
- `storage.versioned_store.demonstrate_version_management` - 50 calls
- `execution.runner.ExecutionRunner.run_with_recovery` - 50 calls
- `execution.runner.run_with_recovery` - 50 calls
- `service.cli.add_service_command` - 45 calls
- `generation.pipeline.RuleBasedPipeline.process` - 43 calls
- `generation.pipeline.process` - 43 calls
- `validators.KubernetesValidator.validate` - 43 calls
- `utils.yaml_compat.safe_load` - 42 calls
- `schema_extraction.script_extractors.MakefileExtractor.extract_from_source` - 41 calls
- `schema_extraction.script_extractors.extract_from_source` - 41 calls
- `cli.history.show_stats` - 39 calls
- `generation.thermodynamic.ThermodynamicGenerator.generate` - 38 calls
- `cli.commands.doctor.doctor_command` - 37 calls
- `web_schema.form_handler.FormHandler.fill_form` - 35 calls

## System Interactions

How components interact:

```mermaid
graph TD
    execute_action_plan --> Console
    execute_action_plan --> frozenset
    execute_action_plan --> print
    execute_action_plan --> enumerate
    handle_run_mode --> _verbose_log
    handle_run_mode --> FormDataLoader
    handle_run_mode --> get_nlp_keywords
    execute_drawing_plan --> get
    execute_drawing_plan --> MouseController
    execute_drawing_plan --> enumerate
    execute_drawing_plan --> loads
    run_command --> time
    run_command --> print_markdown_block
    run_command --> ExecutionResult
    run_command --> append
    run_command --> Popen
    detect_form_fields --> query_selector_all
    detect_form_fields --> _print_yaml
    find_form --> perf_counter
    find_form --> set
    find_form --> _find_best_form_cand
    find_form --> ExplorationResult
    find_form --> start
    generate --> str
    generate --> _debug
    generate --> isinstance
    generate --> _has_fill_form_actio
    find_content --> perf_counter
    find_content --> _resolve_platform_ur
    find_content --> _try_github_api
```

## Reverse Engineering Guidelines

When working with this codebase:

1. **Entry Points**: Start analysis from the entry points listed above
2. **Core Logic**: Focus on classes with many methods (top of 'Key Classes' section)
3. **Data Flow**: Follow data transformation functions for understanding data pipeline
4. **Process Flows**: Use the flow diagrams to understand execution paths
5. **API Surface**: Public API functions show intended external interface
6. **Patterns**: Behavioral patterns indicate reusable design approaches

## Context for LLM

You are analyzing a Python codebase with the above architecture.
- Respond with code changes that preserve existing call graph structure
- Maintain the architectural patterns identified
- Respect the public API surface
- Consider the process flows when suggesting modifications