# KGClaw Architecture

> Ontology-driven knowledge graph construction with an AI Agent Harness
>
> Version 0.1.0 | 2024–2026

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [System Architecture](#2-system-architecture)
3. [Core Modules](#3-core-modules)
4. [Adaptive Pipeline](#4-adaptive-pipeline)
5. [Agent System](#5-agent-system)
6. [Tool System](#6-tool-system)
7. [Skill System](#7-skill-system)
8. [Prompt Engineering](#8-prompt-engineering)
9. [Memory & Persistence](#9-memory--persistence)
10. [Sandbox Execution](#10-sandbox-execution)
11. [Progress & Monitoring](#11-progress--monitoring)
12. [Configuration & Extension](#12-configuration--extension)

---

## 1. Design Philosophy

KGClaw combines **Agent Harness architecture** (inspired by Claude Code and OpenCode) with **ontology-driven knowledge engineering** (inspired by Apple ODKE+, edc, and LightRAG). Its core principles:

- **Ontology-first**: Users define entity types and relation types in natural language. The LLM parses them into a structured schema with descriptions, attributes, domain/range constraints, and parent-child hierarchies. Every subsequent extraction phase is guided and constrained by this ontology.
- **Zero-ontology fallback**: When no ontology is provided, the LLM reads the documents and induces one automatically, then builds the KG from the discovered ontology.
- **Adaptive strategy**: The system inspects data characteristics (narrative text vs. tabular vs. mixed) and automatically selects the optimal extraction strategy.
- **Open-to-canonical**: Inspired by edc, relations are first extracted openly, then mapped to the target ontology via multi-choice Schema Canonicalization.
- **Gleaning**: Inspired by LightRAG, a second extraction pass catches entities missed in the first round.
- **Circuit breaker**: Consecutive tool-call failures trigger automatic degradation to prevent infinite loops.

---

## 2. System Architecture

### 2.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interface Layer                         │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  CLI (Click) │  │ Interactive REPL │  │   Python API     │  │
│  │  kgclaw run   │  │  prompt_toolkit   │  │  from kgclaw     │  │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬─────────┘  │
├─────────┼───────────────────┼─────────────────────┼────────────┤
│         │               UI Layer (ui/)            │            │
│         │      progress.py   display.py           │            │
│         ▼                                          ▼            │
├─────────────────────────────────────────────────────────────────┤
│                   Orchestration Layer (harness/)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     Harness Engine                        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │  │
│  │  │  Phases   │  │Strategies│  │ Helpers  │  │  Engine   │ │  │
│  │  │ (Mixin)   │  │ (Mixin)  │  │ (Mixin)  │  │  (Core)   │ │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     Capability Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  Agent   │  │  Skills  │  │  Tools   │  │  Memory  │      │
│  │ agent.py │  │ skills/  │  │ tools/   │  │memory.py │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
├─────────────────────────────────────────────────────────────────┤
│                     Foundation Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  Models  │  │  Config  │  │ Loaders  │  │ Sandbox  │      │
│  │models.py │  │config.py │  │loaders.py│  │sandbox.py│      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                    │
│  │  Logger  │  │ Prompts  │  │   i18n   │                    │
│  │logger.py │  │prompts/  │  │ i18n.py  │                    │
│  └──────────┘  └──────────┘  └──────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Dependency Graph

```
models, config, logger, loaders, sandbox, prompts, i18n
         │
         ▼
   memory, tools, skills, refinement, git_manager
         │
         ▼
       agent
         │
         ▼
  harness (engine + phases + strategies + helpers)
         │
         ▼
    ui (progress + display)
         │
         ▼
   cli, interactive_app
```

### 2.3 Project Structure

```
kgclaw/
├── __init__.py              # Top-level public API
├── models.py                # Pydantic data models
├── config.py                # User configuration (~/.kgclaw/config.yaml)
├── agent.py                 # Agent system (LLM + Tool Use + Stream + SubAgent + Circuit Breaker)
├── memory.py                # Session memory (message compaction + workflow persistence)
├── logger.py                # Structured logging (RotatingFileHandler + debug)
├── loaders.py               # 10+ format file loaders + recursive directory scanning
├── sandbox.py               # Sandbox execution (run_python + AST audit + format analysis)
├── i18n.py                  # Internationalization (gettext-style, zh/en)
├── refinement.py            # KG refinement engine (ontology optimization from user feedback)
├── git_manager.py           # Git version management for build history
├── cli.py                   # CLI entry point (Click + Rich + setup wizard)
├── interactive_app.py       # Interactive REPL (prompt_toolkit + Live/Markdown)
├── harness/                 # Orchestration engine
│   ├── __init__.py
│   ├── engine.py            #   Main engine + document loading + export
│   ├── phases.py            #   8-phase implementation + Gleaning + Canonicalization
│   ├── strategies.py        #   auto/fast/code strategies
│   └── helpers.py           #   Chunking, dedup, fuzzy matching, Agent factory
├── tools/                   # Tool system (13 tools)
│   ├── __init__.py          #   Tool base class + registry
│   ├── file_tools.py        #   read_file, write_file, list_files
│   ├── text_tools.py        #   search_in_text, extract_text_segments, parse_json
│   ├── validation_tools.py  #   validate_against_ontology, deduplicate_entities
│   ├── agent_tools.py       #   propose_action, run_python, analyze_file_format
│   └── extraction_tools.py  #   extract_with_llm_prompt, extract_with_code
├── skills/                  # Skill system
│   ├── __init__.py          #   Skill base class + SkillRegistry
│   └── builtins.py          #   5 built-in skills
├── prompts/                 # Prompt templates
│   └── system_prompts.py    #   11 system/task prompts + few-shot generators
├── ui/                      # Shared UI layer
│   ├── __init__.py
│   ├── progress.py          #   Weighted progress callback factory
│   └── display.py           #   Result display utilities
└── locales/                 # Translation files
    └── en/LC_MESSAGES/
        └── kgclaw.po        # English translations
```

---

## 3. Core Modules

### 3.1 Data Models (models.py)

```python
# Ontology definition
Ontology
├── EntityType   (name, description, parent, attributes)
└── RelationType (name, description, domain, range, inverse)

# Extraction results
Entity     (name, type, description, attributes, mention, confidence)
Relation   (subject, predicate, object, keywords, description, confidence, evidence)
Triple     (subject: Entity, predicate: str, object: Entity, confidence, evidence)

# Workflow state
WorkflowState    (ontology, documents, phases, final_result, output_nt)
PhaseResult      (phase_name, status, output, error_message)
ExtractionResult (entities, relations, triples, metadata)

# Refinement
OntologyChange   (action, target, name, description, reason)
RefinementPlan   (ontology_changes, suggested_strategy, extraction_tips, ...)

# Configuration
LLMConfig        (provider, model, api_key, api_base, temperature, max_tokens)
HarnessConfig    (llm, max_concurrent_agents, chunk_size, enable_gleaning, ...)
AgentConfig      (name, system_prompt, tools, max_tool_calls, structured_output_schema)
```

### 3.2 Agent System (agent.py)

```
Agent
├── run()              # Synchronous execution with full tool-calling loop
├── run_stream()       # Streaming execution, yields (event_type, data) tuples
├── run_structured()   # Structured output with 5-tier JSON parsing + LLM self-repair
├── spawn_subagent()   # Spawn child agent, inherits LLM config and Memory
├── on_event()         # Register event callbacks
└── _emit()            # Broadcast events to registered callbacks

AgentConfig
├── name: str
├── system_prompt: str
├── tools: list[str]           # Names of tools available to this agent
├── max_tool_calls: int = 20   # Hard limit; exceeded → strip tools, force text response
├── model_config: LLMConfig    # Agent-level model override
└── structured_output_schema   # JSON Schema constraint
```

**Tool-calling loop protections:**
- `tools_exhausted` flag: after `max_tool_calls`, tools are stripped from the request to force a text response
- `consecutive_failures` circuit breaker: 3 consecutive failed tool calls trigger automatic degradation
- `max_iterations` cap: prevents infinite loops

### 3.3 Memory System (memory.py)

```
Memory
├── Conversation message management (per-agent message lists, thread-safe)
├── Workflow state persistence (workflow_state.json)
├── Context compaction (compact_messages: keep system + last N)
├── Context store (set_context/get_context — arbitrary key-value)
├── Document management (remove, clear, list with metadata)
├── Document manifest (MD5-based change detection between sessions)
├── Generated code persistence
├── Ontology export (JSON + Markdown)
└── N-Triples export (with URI encoding and literal escaping)
```

### 3.4 File Loaders (loaders.py)

10+ formats, registered via decorator:

| Format | Extensions | Special Handling |
|--------|-----------|-----------------|
| Plain text | .txt, .md, .markdown, .text | UTF-8 read |
| JSONL | .jsonl | Extracts data/text fields |
| DOCX | .docx | ZIP + XML parsing |
| PDF | .pdf | pypdf per-page extraction |
| HTML | .html, .htm | BeautifulSoup tag stripping |
| CSV | .csv, .tsv | DictReader, preserves raw_rows |
| Excel | .xlsx, .xls | openpyxl per-sheet reading |

Directory loading supports recursive scanning, glob exclusion patterns, and automatic encoding detection.

### 3.5 Refinement Engine (refinement.py)

```
RefinementEngine
├── analyze(last_result, ontology, docs, user_feedback, strategy) → RefinementPlan
│   └── LLM analyzes the gap between user expectations and last build output
└── apply(plan, session) → dict of applied changes
    ├── Updates ontology (new entity/relation types)
    ├── Adjusts strategy (fast/standard/code)
    ├── Toggles gleaning and co-occurrence
    └── Records extraction tips for next build
```

### 3.6 Git Version Manager (git_manager.py)

```
GitManager(work_dir)
├── init()                         # Initialize git repo in work directory
├── commit_build(workflow_id, summary) → hash
├── commit_ontology_update(preview) → hash
├── get_history(n)                 # List recent commits
├── get_current_hash()             # Current HEAD
├── rollback(commit_hash)          # Restore to a previous version
└── has_commits()                  # Check if any commits exist
```

---

## 4. Adaptive Pipeline

### 4.1 Full Pipeline (Standard Strategy)

```
Phase 0:   Ontology Auto-Discovery                     [5%]
           When no ontology is provided, LLM induces entity/relation types from documents.

Phase 1:   Ontology Analysis                          [5%]
           LLM parses natural-language ontology into structured EntityType/RelationType objects.

Phase 1.5: Agent Code Extraction                      [optional]
           Agent generates a custom extraction prompt or Python code for the specific dataset.

Phase 2:   Entity Extraction                         [40%]
           ThreadPool parallel chunked LLM extraction + fuzzy dedup + Gleaning second pass.

Phase 2.5: Structured Data Extraction                 [5%]
           CSV/XLSX → LLM column-to-ontology mapping → entity/relation extraction.

Phase 3:   Relation Extraction                       [15%]
           Document grouping + entity pre-filtering + ThreadPool parallel extraction.

Phase 3.5: Co-occurrence Graph                        [5%]
           Paragraph-level entity co-occurrence counting + frequency weighting.
           When domain/range constraints match, pairs are upgraded to ontology relations.

Phase 4:   Quality Check                             [20%]
           LLM review + Schema Canonicalization + type corrections + dedup verification.

Phase 5:   Triple Construction                        [5%]
           Programmatic SPO assembly with 4-level fuzzy matching and domain/range enforcement.
```

### 4.2 Strategy Selection

| Strategy | Trigger Condition | Behavior |
|----------|------------------|----------|
| `fast` | ≤ 5 docs, avg < 2000 chars | Single-pass combined entity + relation extraction |
| `code` | Tabular files > 50% | Agent generates Python code, executes in sandbox |
| `standard` | Default | Full 8-phase pipeline |
| `auto` | Automatic | Inspects data characteristics, selects one of the above |

### 4.3 Gleaning — Second-Pass Catch-Up

Inspired by LightRAG's `entity_continue_extraction`:

1. After the first extraction pass, a summary of already-extracted entities (name + type) is injected into a Gleaning prompt.
2. The LLM re-examines the source text against the existing entity list and finds entities that were missed, truncated, or malformed.
3. New entities are added to the result set. For existing entities, the longer description wins.
4. Controlled by `HarnessConfig.enable_gleaning` (default: on).

### 4.4 Schema Canonicalization

Inspired by edc's Schema Canonicalization, executed during the quality-check phase:

```
For each extracted relation predicate not matching the ontology:
  1. Build candidate list: all ontology relation types with definitions
  2. Multi-choice prompt: "Extracted relation X. Candidates: A. schema_rel_1 (def), B. ..., Z. None of the above"
  3. LLM outputs a letter → parsed into the best-match ontology relation
  4. Mapping applied: original predicate → canonical ontology predicate
```

All unmatched predicates are batched into a single LLM call (up to 20 at a time).

---

## 5. Agent System

### 5.1 Execution Flow

```
run(user_message, max_iterations=10)
│
├─ Build messages: [system] + [memory] + [user]
│
├─ for iteration in range(max_iterations):
│   │
│   ├─ LLM call (OpenAI-compatible API)
│   │   ├─ response.usage → accumulate tokens → emit token_usage
│   │   └─ No usage → character estimate fallback → emit token_usage
│   │
│   ├─ Has tool_calls?
│   │   ├─ Execute each tool → track consecutive_failures
│   │   ├─ consecutive_failures ≥ 3 → circuit breaker → tools_exhausted
│   │   └─ iteration ≥ max_tool_calls - 1 → hard stop → tools_exhausted
│   │
│   ├─ tools_exhausted? → Next request strips tools, forces text response
│   │
│   └─ No tool_calls → return final response
│
└─ max_iterations exhausted → return fallback response
```

### 5.2 Streaming Execution (run_stream)

```
run_stream(user_message) → Generator[(event_type, data)]
│
├─ thinking    → Start of each iteration
├─ token       → Token-by-token streaming output
├─ tool_call   → Tool invocation (name + arguments)
├─ tool_result → Tool result (success/failure + summary)
├─ error       → Exception information
└─ done        → Final response complete
```

### 5.3 Structured Output (run_structured)

Multi-tier JSON parsing:

1. Direct `json.loads()` parse
2. Extract JSON from markdown code blocks
3. Regex match `{...}` or `[...]` boundaries
4. **LLM self-repair**: Send malformed JSON back to the LLM (temperature=0.1, no tools) for correction
5. Apply tiers 1–3 again on the repaired output

### 5.4 Subagent Spawning

```python
parent.spawn_subagent(name, system_prompt, task, tools)
│
├─ Creates AgentConfig (name = "parent.child")
├─ Inherits parent's LLM config and Memory
├─ Has independent message context and tool set
└─ Events forwarded through parent's callback chain
```

### 5.5 Parallel Chunked Extraction

Entity and relation extraction both use `ThreadPoolExecutor`:

```
ThreadPoolExecutor(max_workers=min(config.max_concurrent_agents, chunks))
│
├── Thread-0: Agent("entity_extractor_0") → OpenAI Client → HTTP
├── Thread-1: Agent("entity_extractor_1") → OpenAI Client → HTTP
├── Thread-2: Agent("entity_extractor_2") → OpenAI Client → HTTP
└── Thread-3: Agent("entity_extractor_3") → OpenAI Client → HTTP

Each thread has its own:
  • Agent instance (independent message context)
  • OpenAI Client (independent httpx connection pool)
  • LLMConfig (shared, read-only)

Shared (thread-safe):
  • Memory (per-agent message slots + lock)
  • Result accumulator (threading.Lock)
  • Event emitter (threading.Lock)
  • stop_event (atomic, for Ctrl+C graceful shutdown)
```

---

## 6. Tool System

### 6.1 Tool Registration

```python
@Tool.register(name="tool_name", description="...", parameters={...})
def tool_function(...):
    ...
```

13 built-in tools organized into 5 groups:

| Group | Tools | Module |
|-------|-------|--------|
| **File** | read_file, write_file, list_files | file_tools.py |
| **Text** | search_in_text, extract_text_segments, parse_json | text_tools.py |
| **Validation** | validate_against_ontology, deduplicate_entities | validation_tools.py |
| **Agent** | propose_action, run_python, analyze_file_format | agent_tools.py |
| **Extraction** | extract_with_llm_prompt, extract_with_code | extraction_tools.py |

### 6.2 Tool Execution Wrapper

Each tool invocation includes:

1. **Parameter tolerance**: When LLM-supplied parameter names don't match, falls back to positional matching
2. **Type adaptation**: Accepts both JSON strings and pre-parsed objects
3. **Size limits**: Validation tools cap at 500 entities/relations to prevent LLM from passing oversized payloads
4. **Output truncation**: Sandbox Python output capped at 100K characters

---

## 7. Skill System

### 7.1 Five Built-in Skills

| Skill | System Prompt | Tools | Output Schema |
|-------|---------------|-------|---------------|
| `ontology_analyzer` | Ontology analysis specialist | read_file, parse_json, write_file | entity_types + relation_types |
| `entity_extractor` | KG entity extraction specialist (V2) | None (text in prompt) | entities[] (name + type + description) |
| `relation_extractor` | KG relation extraction specialist | None | relations[] (subject + predicate + object + keywords + description) |
| `quality_checker` | Quality review specialist | validate, deduplicate, read, parse | corrections + approved + rejected |
| `triple_constructor` | Triple construction agent | parse_json, write_file | triples[] |

### 7.2 Skill Registry

```
SkillRegistry
├── register(meta)          # Decorator to register a Skill class
├── get(name, llm_config)   # Get a Skill instance (cached)
├── list_all()              # List all registered Skill metadata
└── discover_from_directory() # Load custom skills from a directory
```

### 7.3 Default Pipeline

```python
get_default_pipeline_skills() → [
    "ontology_analyzer",
    "entity_extractor",
    "relation_extractor",
    "quality_checker",
    "triple_constructor",
]
```

---

## 8. Prompt Engineering

### 8.1 Design Sources

| Feature | Source | Description |
|---------|--------|-------------|
| Entity description field | LightRAG | Rich per-entity descriptions for context |
| Relation keywords + description | LightRAG | `keywords` + `description` fields per relation |
| Naming normalization rules | LightRAG | Title Case, third person, no pronouns, full names preferred |
| Output format safety | LightRAG | Prevent LLM from extracting entities from few-shot examples |
| Gleaning second pass | LightRAG | Feed first-round results back to find missed entities |
| Schema Canonicalization | edc | Multi-choice mapping of open relations to ontology |
| Open-to-canonical | edc | Extract first, map later — avoid premature constraint |
| Agent circuit breaker | OpenCode | Automatic degradation on consecutive failures |

### 8.2 Entity Extraction V2 Prompt Structure

```
---Role---
KG Entity Extraction Specialist

---Core Requirements---
Full coverage + accurate descriptions

---Entity Naming Normalization---
1. Title Case standardization
2. Third person
3. Avoid pronouns ("this article", "the company", "I", "you", "he/she")
4. Prefer full names
5. Strip titles/honorifics

---Output Format---
JSON: {name, type, description, mention, confidence, attributes}

---Key Rules---
1. Exhaustive extraction  2. Type mapping  3. Dedup
4. Low-confidence items included  5. Attribute extraction  6. Format safety
```

### 8.3 Few-shot Example Generation

Few-shot examples are dynamically generated from the ontology and document samples, giving the LLM concrete examples of expected entity and relation output formats tailored to the user's specific domain.

---

## 9. Memory & Persistence

### 9.1 Workflow State

```
.kgclaw/
├── workflow_state.json    # Full workflow state (ontology, docs, phases, results)
├── document_manifest.json # MD5 hashes + mtimes for change detection
├── output.nt              # N-Triples output
├── output.json            # JSON output
├── output.jsonl           # JSONL output
├── ontology.json          # Standalone ontology in JSON
├── ontology.md            # Standalone ontology in Markdown
├── generated_code/        # Agent-generated extraction code
│   └── extraction_prompt_*.txt
└── logs/
    └── kgclaw.log         # Structured logs (10 MB × 3 rotations)
```

### 9.2 Context Compaction

```
compact_messages(agent_id, max_messages=50)
│
├─ Messages ≤ max_messages → no compaction needed
├─ Preserve system messages
├─ Keep last N non-system messages
└─ Middle messages → summarized and inserted as a system message
```

### 9.3 Document Change Detection

```
detect_file_changes(current_paths)
│
├─ Load stored manifest (MD5 hash + mtime + size per file)
├─ Compute current MD5 for each file
├─ Classify each file: unchanged / added / modified / deleted
└─ Return structured change report
```

Used by the interactive REPL to intelligently decide whether a rebuild is needed when resuming a session.

---

## 10. Sandbox Execution

### 10.1 Architecture

```
Agent-generated Python code → AST safety audit → subprocess sandbox execution
                                                       │
                                              ┌────────┴────────┐
                                          pass (30s)        timeout/error
                                              │                 │
                                         stdout/stderr     kill + report
                                      (capped at 100K chars)
```

### 10.2 AST Safety Rules

**Forbidden imports**: `os`, `subprocess`, `socket`, `requests`, `urllib`, `shutil`, `ctypes`, `multiprocessing`, `signal`, `pty`, `fcntl`, `posix`, `grp`, `pwd`, `crypt`, `importlib`, `sys`, `builtins`

**Forbidden calls**: `eval()`, `exec()`, `compile()`, `__import__()`, `breakpoint()`, `open()`

**Blocked bypass patterns**:
- `__builtins__[...]` subscript access
- `__class__.__bases__.__subclasses__()` class-hierarchy navigation
- `getattr(__builtins__, ...)`, `vars(__builtins__)`, etc.
- Attribute chains touching `__globals__`, `__code__`, `__closure__`, `__dict__`

**Allowed modules**: `json`, `csv`, `re`, `collections`, `itertools`, `math`, `pathlib`, `io`, `string`, `textwrap`, `datetime`, `typing`, `dataclasses`, `enum`

---

## 11. Progress & Monitoring

### 11.1 Weighted Progress Bar

8 phases contribute to a weighted progress total of 100%:

| Phase | Weight | Notes |
|-------|--------|-------|
| auto_discover_ontology | 5% | LLM ontology induction |
| ontology_analysis | 5% | LLM ontology structuring |
| entity_extraction | 40% | Most time-consuming; subdivided by chunk count |
| relation_extraction | 15% | LLM relation extraction |
| co_occurrence | 5% | Programmatic co-occurrence computation |
| structured_extraction | 5% | Tabular data mapping |
| quality_check | 20% | LLM review + Schema Canonicalization |
| triple_construction | 5% | Programmatic triple assembly |

### 11.2 Anti-Fake-Completion Lock

- Progress is capped at 98% until the `workflow_complete` event unlocks the final 2%.
- `token_usage`, `agent_call_start`, and `agent_call_end` events force a progress bar refresh.
- When the API doesn't return `usage`, token counts are estimated from character counts (4 chars ≈ 1 token).

### 11.3 Logging

| Mode | File Log | Console Output |
|------|----------|---------------|
| Normal | INFO+ → `.kgclaw/logs/kgclaw.log` (10 MB × 3 rotations) | WARNING+ |
| Debug | DEBUG+ → `.kgclaw/logs/kgclaw.log` (full prompts/responses) | INFO+ |

---

## 12. Configuration & Extension

### 12.1 Config File

```yaml
# ~/.kgclaw/config.yaml
llm:
  provider: openai
  model: gpt-4o
  api_key: sk-xxx
  api_base: https://api.openai.com/v1
  temperature: 0.3
  max_tokens: 16384
preferences:
  output_format: nt
  chunk_size: 2000
  verbose: false
```

### 12.2 Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key |
| `KGCLAW_MODEL` | Model name |
| `KGCLAW_API_BASE` | API endpoint URL |
| `KGCLAW_VERBOSE` | Verbose output (1/true/yes) |
| `KGCLAW_LANG` | UI language (zh/en) |

### 12.3 Custom Skills

```python
# my_skills/my_extractor.py
from kgclaw.skills import Skill, SkillMeta, SkillRegistry

@SkillRegistry.register(SkillMeta(
    name="my_extractor",
    description="Custom entity extractor",
    produces=["entities"],
))
class MyExtractor(Skill):
    def get_system_prompt(self) -> str:
        return "You are a specialized extractor..."

    def get_tool_names(self) -> list[str]:
        return ["read_file"]

    def get_output_schema(self) -> dict:
        return {"type": "object", "properties": {...}}
```

```bash
kgclaw run --skills-dir my_skills/ -d docs.txt
```

### 12.4 Supported LLM Providers

Any OpenAI-compatible API: OpenAI / DeepSeek / Qwen / Ollama (local) / vLLM / custom.

---

## Acknowledgments

This project was inspired by the following excellent projects:

- **Claude Code** (Anthropic, 2024–2026) — Agent Harness architecture, Dynamic Workflows orchestration pattern
- **OpenCode** (2024–2026) — Agent Harness architecture, Tool registry, Permission system
- **LightRAG** (HKU, 2024–2025) — Gleaning, entity description fields, naming normalization rules
- **edc** (2024) — Open extraction → standardization, Schema Canonicalization
- **Apple ODKE+** (2025) — Production-grade ontology-guided KG extraction pipeline
- **Microsoft GraphRAG** (2024) — Unstructured text → entities/relations → community detection
