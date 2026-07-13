# KGClaw Technical Overview

> A technical whitepaper on ontology-driven knowledge graph construction with an AI Agent Harness
>
> Version 0.2.0 | 2024–2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Design Philosophy & Innovations](#2-design-philosophy--innovations)
3. [System Architecture](#3-system-architecture)
4. [Adaptive Knowledge Extraction Pipeline](#4-adaptive-knowledge-extraction-pipeline)
5. [Agent System Design](#5-agent-system-design)
6. [Tool & Skill Systems](#6-tool--skill-systems)
7. [Prompt Engineering & Multi-Source Fusion](#7-prompt-engineering--multi-source-fusion)
8. [Ontology Building System](#8-ontology-building-system)
9. [Interaction Modes](#9-interaction-modes)
10. [Sandbox Execution & Safety](#10-sandbox-execution--safety)
11. [Performance & Monitoring](#11-performance--monitoring)
12. [Engineering Practices](#12-engineering-practices)
13. [Acknowledgments](#13-acknowledgments)
14. [Summary & Future Directions](#14-summary--future-directions)

---

## 1. Project Overview

KGClaw is an **LLM-powered Agent Harness system for ontology-driven knowledge graph construction from unstructured text**. Users provide documents and an ontology definition (entity types + relation types) in natural language, and the system automatically completes the full pipeline—from document parsing, entity extraction, relation extraction, and quality review to triple construction—outputting a structured knowledge graph in N-Triples, JSON, or JSONL formats.

### 1.1 Key Metrics

| Metric | Value |
|--------|-------|
| Source files | 34+ Python modules |
| Built-in tools | 13 (file, text, validation, agent, extraction) |
| Built-in skills | 5 (ontology analysis, entity extraction, relation extraction, quality check, triple construction) |
| Pipeline phases | 8 (including optional agent code extraction and co-occurrence graph) |
| Ontology building modes | 6 (T-O, R-O, HT-R-O, affinity-clustering, dense-ontology, auto) |
| Dataset presets | 6 (WebNLG, NYT, CoNLL04, SRedFM, Rebel, Wiki-NRE) |
| File formats | 10+ (txt, md, jsonl, docx, pdf, html, csv, tsv, xlsx, xls) |
| Built-in ontology templates | 3 (character relations, enterprise, legislation) |
| LLM backends | Any OpenAI-compatible API |
| Dependencies | Pure Python, zero external database dependencies |

### 1.2 Three Usage Modes

- **CLI mode** (`kgclaw run`): One command for end-to-end KG construction. Ideal for batch processing and CI/CD integration.
- **Interactive REPL** (`kgclaw interactive`): Claude Code-style terminal experience with streaming chat, 22+ slash commands, agent proposals, and Ctrl+O real-time verbose toggle.
- **Python API** (`from kgclaw import Harness`): Programmatic integration with custom configuration, progress callbacks, and pipeline extension.

---

## 2. Design Philosophy & Innovations

### 2.1 Core Design Principles

KGClaw fuses **Agent Harness architecture** (inspired by Claude Code / OpenCode) with **ontology-driven knowledge engineering** (inspired by Apple ODKE+ / edc / LightRAG):

- **Ontology-first, natural language entry**: Users define entity types and relation types in plain natural language—no OWL, RDF, or formal ontology language required. The LLM parses this into a structured schema with descriptions, attributes, domain/range constraints, and parent-child hierarchies. Every subsequent extraction phase is driven and constrained by this ontology.
- **Zero-ontology auto-discovery**: When no ontology is provided, the LLM reads the documents and induces one automatically, then builds the KG from the discovered ontology.
- **Adaptive strategy**: The system inspects data characteristics (narrative text vs. tabular vs. unknown formats) and automatically selects the optimal extraction strategy from four options (auto/fast/standard/code).
- **Open-to-canonical**: Inspired by edc, relations are first extracted openly, then mapped to the target ontology via multi-choice Schema Canonicalization—avoiding premature constraints while ensuring Schema compliance.
- **Gleaning second pass**: Inspired by LightRAG, a second extraction pass feeds previously extracted entities back to the LLM to catch overlooked, truncated, or malformed entities.
- **Circuit breaker protection**: Consecutive tool-call failures trigger automatic degradation to a text-only mode, preventing infinite loops.

### 2.2 Key Innovations

1. **Natural language ontology with iterative refinement**: Users define ontologies conversationally and refine them via `/refine`—the LLM analyzes the last build against user feedback and proposes concrete ontology changes, strategy adjustments, and prompt improvements.

2. **Schema Canonicalization** (inspired by edc): Open extraction results are mapped to the target ontology through a batched multi-choice mechanism, solving the "extraction freedom vs. Schema compliance" tension.

3. **Gleaning second-pass catch-up** (inspired by LightRAG): Previously extracted entity summaries are fed back to the LLM for a targeted second extraction pass, significantly improving entity recall.

4. **Agent circuit breaker**: Three consecutive tool-call failures automatically degrade the agent to text-only mode; hitting the `max_tool_calls` limit strips tools entirely—a defense-in-depth approach against runaway tool-calling loops.

5. **4-level fuzzy entity matching + fallback**: Triple construction resolves entity references through exact match → normalized match → substring match → placeholder creation, ensuring that extraction noise doesn't silently drop valid triples.

6. **LLM JSON self-repair**: When structured output parsing fails across multiple regex-based fallback strategies, the malformed response is sent back to the LLM at low temperature for correction—significantly improving structured output success rates.

7. **Session resume with intelligent change detection**: On restart, KGClaw compares file MD5 hashes against a stored manifest. If nothing changed, cached results are reused instantly. If files were added, modified, or deleted, the user is told exactly what changed and why a rebuild is needed.

---

## 3. System Architecture

### 3.1 Layered Architecture

KGClaw is organized into four layers:

```
┌──────────────────────────────────────────────────────────────┐
│                    User Interface Layer                       │
│  CLI (Click + Rich)  │  Interactive REPL (prompt_toolkit)    │
│                       │  Python API (Harness class)           │
├──────────────────────────────────────────────────────────────┤
│                    Orchestration Layer                        │
│  Harness Engine: 4 Mixins (Core + Phases + Strategies + Helpers) │
│  RefinementEngine: analyze + apply                            │
│  GitManager: version control for build history                │
├──────────────────────────────────────────────────────────────┤
│                    Capability Layer                           │
│  Agent (sync/stream/structured/subagent)                      │
│  Skills (5 built-in + custom discovery)                       │
│  Tools (13 built-in, 5 categories)                            │
│  Memory (conversation + workflow + context compaction)        │
│  OntologyBuilder (6 modes) │ Presets (6 datasets)             │
│  Tracer (JSONL trace writer)                                  │
├──────────────────────────────────────────────────────────────┤
│                    Foundation Layer                           │
│  Models (Pydantic) │ Config (YAML + env) │ Loaders (10+ formats) │
│  Sandbox (subprocess + AST audit) │ Logger │ Prompts │ i18n   │
│  Utils (Levenshtein, etc.)                                    │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Module Dependency Graph

Foundation modules (`models`, `config`, `logger`, `loaders`, `sandbox`, `prompts`, `i18n`) have no internal dependencies and feed upward. Capability modules (`memory`, `tools`, `skills`, `refinement`, `git_manager`) depend on the foundation. The `agent` module composes capabilities. The `harness` orchestration layer composes the agent with phase implementations and strategies. Finally, `ui` → `cli` / `interactive_app` form the user-facing layer.

### 3.3 Design Patterns

| Pattern | Application | Description |
|---------|-------------|-------------|
| **Mixin multiple inheritance** | Harness class | Engine + Phases + Strategies + Helpers composed via Mixins |
| **Registry** | Tools, Skills, Loaders | Decorator-based registration with name-based lookup |
| **Observer** | Event callbacks | `on_event()` register → `_emit()` broadcast |
| **Factory** | `create_agent()`, `create_subagent_factory()` | Agent instance creation |
| **Strategy** | auto/fast/standard/code | Runtime strategy selection based on data characteristics |
| **Singleton** | Logger, SkillRegistry | Global unique instances |
| **Template Method** | Skill base class | pre_process → core logic → post_process |

---

## 4. Adaptive Knowledge Extraction Pipeline

### 4.1 Full Pipeline (Standard Strategy)

```
User Input (Ontology + Documents/Directory)
        │
        ▼
┌──────────────────────────────────────────────┐
│ Phase 0:   Ontology Auto-Discovery            │  ← LLM induces ontology from documents
├──────────────────────────────────────────────┤
│ Phase 1:   Ontology Analysis                  │  ← LLM structured schema extraction
├──────────────────────────────────────────────┤
│ Phase 1.5: Agent Code Extraction (optional)   │  ← Agent-generated custom prompt/code
├──────────────────────────────────────────────┤
│ Phase 2:   Entity Extraction (Parallel TPool) │  ← LLM parallel chunks + Gleaning
├──────────────────────────────────────────────┤
│ Phase 2.5: Structured Data Extraction         │  ← CSV/XLSX → column-to-ontology mapping
├──────────────────────────────────────────────┤
│ Phase 3:   Relation Extraction                │  ← Doc grouping + entity pre-filter parallel
├──────────────────────────────────────────────┤
│ Phase 3.5: Co-occurrence Graph                │  ← Paragraph co-occur + frequency weight
├──────────────────────────────────────────────┤
│ Phase 4:   Quality Check                      │  ← LLM review + Schema Canonicalization
├──────────────────────────────────────────────┤
│ Phase 5:   Triple Construction (programmatic) │  ← 4-level fuzzy match + domain/range
└──────────────────────────────────────────────┘
        │
        ▼
   Output: N-Triples / JSON / JSONL
```

### 4.2 Phase-by-Phase Breakdown

| Phase | Weight | Input | Output | Key Technique |
|-------|--------|-------|--------|---------------|
| **Phase 0: Auto-Discovery** | 5% | First 3 documents | Induced ontology | LLM induction + JSON structuring |
| **Phase 1: Ontology Analysis** | 5% | User ontology (natural language) | Structured Schema | LLM parsing + auto-completion |
| **Phase 1.5: Agent Code Extraction** | optional | Doc samples + ontology | Custom extraction prompt | Agent generation + LLM execution |
| **Phase 2: Entity Extraction** | 40% | All documents (chunked) | Entity list | **ThreadPool parallel + Gleaning** |
| **Phase 2.5: Structured Extraction** | 5% | CSV/XLSX tabular data | Entities + relations | LLM column-to-ontology mapping |
| **Phase 3: Relation Extraction** | 15% | Doc groups + entity list | Relation list | Entity pre-filter + ThreadPool parallel |
| **Phase 3.5: Co-occurrence Graph** | 5% | All docs + entity list | Co-occurrence relations | Paragraph-level co-occur + frequency weighting |
| **Phase 4: Quality Check** | 20% | All extraction results | Corrections + review report | **Schema Canonicalization** |
| **Phase 5: Triple Construction** | 5% | Entities + relations | SPO triples | **4-level fuzzy matching** + fallback |

### 4.3 Phase 2 — Entity Extraction in Detail

1. **Strategy analysis**: A lightweight LLM call inspects document samples and decides whether to use LLM extraction or code-based extraction, full text or truncation.
2. **Chunking**: Documents are concatenated and split into chunks (configurable size).
3. **Parallel extraction**: Each chunk is dispatched to an independent Agent instance via `ThreadPoolExecutor`. Agents share a read-only `LLMConfig` but have independent message contexts and HTTP clients.
4. **Deduplication**: Results are merged with exact-match dedup (name + type key), then fuzzy dedup (difflib similarity > threshold).
5. **Gleaning pass**: The deduplicated entity list is fed back to a Gleaning Agent that finds missed or malformed entities. New entities are added; existing entities get description updates if the gleaned version is richer.

### 4.4 Strategy Selection

| Strategy | Trigger | LLM Calls | Characteristics |
|----------|---------|-----------|-----------------|
| `fast` | ≤ 5 docs, avg < 2000 chars | O(n) docs | Single-pass combined entity + relation extraction |
| `standard` | Default | O(n) chunks × phases | Full 8-phase, Gleaning, Canonicalization |
| `code` | Tabular files > 50% | O(1) generation + O(1) execution | Agent generates Python code, sandbox execution |
| `auto` | Automatic | — | Inspects data, selects one of the above |

### 4.5 Schema Canonicalization Mechanism

```
Open extraction results:         Target ontology schema:
  "father"                        A. 生父 (biological_father)
  "dad"                           B. 老师 (teacher)
  "papa"                          C. 朋友 (friend)
  "old man"                       D. None of the above
        │                                    │
        └────────────┬───────────────────────┘
                     ▼
        Batch multi-choice LLM prompt:
        "Map each extracted relation to the
         best ontology match or NONE."
                     │
                     ▼
        Mapping: "father"→生父, "dad"→生父,
                 "papa"→生父, "old man"→NONE
```

All unmatched predicates (up to 20) are batched into a single LLM call, avoiding per-relation latency.

### 4.6 Triple Construction — 4-Level Fuzzy Matching

```
For each relation (subject_name, predicate, object_name):

  Level 1: Exact match in entity index
  Level 2: Normalized match (stripped whitespace/punctuation/brackets)
  Level 3: Substring match
  Level 4: Create placeholder Entity with confidence=0.3

Apply domain/range constraints from ontology:
  If subject.entity_type ≠ predicate.domain → try alternative candidates
  If object.entity_type ≠ predicate.range → try alternative candidates

Fallback for unmatched relations:
  Create low-confidence (≤ 0.35) placeholder entities and triples
```

---

## 5. Agent System Design

### 5.1 Agent Execution Model

Each Agent instance encapsulates an independent conversation with the LLM, including tool-use loops, streaming output, and structured output parsing:

```
Agent.run(user_message)
│
├─ Build messages: [system prompt] + [memory context] + [user message]
│
├─ Loop (up to max_iterations):
│   ├─ Call LLM (OpenAI-compatible API)
│   ├─ Track token usage (API-reported or character-estimated fallback)
│   ├─ If tool_calls present AND not tools_exhausted:
│   │   ├─ Execute each tool
│   │   ├─ Track consecutive_failures (circuit breaker at 3)
│   │   ├─ Check max_tool_calls (hard stop)
│   │   └─ Append results to message history, continue loop
│   └─ If no tool_calls: return final text response
│
└─ Return fallback message if max_iterations exhausted
```

### 5.2 Agent Interfaces

```python
class Agent:
    def run(self, user_message: str, max_iterations: int = 10) -> str:
        """Synchronous execution with full tool-calling loop."""

    def run_stream(self, user_message: str, max_iterations: int = 10):
        """Streaming execution. Yields: thinking | token | tool_call | tool_result | error | done."""

    def run_structured(self, user_message: str, output_schema: dict, max_iterations: int = 10) -> Optional[dict]:
        """Structured output with 5-tier JSON parsing + LLM self-repair (temperature=0.1)."""

    def spawn_subagent(self, name: str, system_prompt: str, task: str, tools: Optional[list[str]] = None) -> str:
        """Spawn independent child agent inheriting LLM config and Memory."""
```

### 5.3 Parallel Chunked Extraction Design

Each parallel worker thread gets:
- An **independent Agent instance** (no shared message context)
- An **independent OpenAI Client** (no shared httpx connection pool)
- A **shared read-only LLMConfig**

Thread-safe shared state:
- **Memory**: per-agent message slots with `threading.Lock`
- **Result accumulator**: `threading.Lock`-protected list
- **Event emitter**: `threading.Lock`-protected callback dispatch
- **Stop event**: `threading.Event` for graceful Ctrl+C shutdown

The GIL is automatically released during HTTP I/O, achieving true concurrency for LLM API calls.

---

## 6. Tool & Skill Systems

### 6.1 Tool System

Tools are atomic capabilities callable by agents via function calling. Each tool is registered with a decorator specifying its name, description, and JSON Schema parameters:

```python
@Tool.register(
    name="read_file",
    description="Read the contents of a file",
    parameters={"type": "object", "properties": {...}, "required": [...]}
)
def read_file(path: str) -> str: ...
```

**13 built-in tools in 5 categories:**

| Category | Tools | Purpose |
|----------|-------|---------|
| **File** | read_file, write_file, list_files | Filesystem access |
| **Text** | search_in_text, extract_text_segments, parse_json | Text manipulation |
| **Validation** | validate_against_ontology, deduplicate_entities | Quality assurance |
| **Agent** | propose_action, run_python, analyze_file_format | Agent autonomy |
| **Extraction** | extract_with_llm_prompt, extract_with_code | Custom extraction |

Each tool execution includes: parameter-name tolerance (LLMs may use different names), type adaptation (accepts both JSON strings and parsed objects), size limits (validation tools cap at 500 items), and output truncation (sandbox output at 100K chars).

### 6.2 Skill System

Skills are higher-level abstractions that encapsulate a complete KG construction capability: a system prompt, a tool set, input/output schemas, and pre/post-processing logic.

**Five built-in skills form the default pipeline:**

```
ontology_analyzer → entity_extractor → relation_extractor → quality_checker → triple_constructor
```

| Skill | Role | Tools |
|-------|------|-------|
| `ontology_analyzer` | Parses natural language into structured EntityType/RelationType objects | read_file, parse_json, write_file |
| `entity_extractor` | Extracts entities from text chunks with LightRAG-inspired naming rules | None (text in prompt) |
| `relation_extractor` | Extracts relations between known entities from document groups | None |
| `quality_checker` | Reviews extraction quality, performs Schema Canonicalization | validate, deduplicate, read, parse |
| `triple_constructor` | Assembles validated entities and relations into SPO triples | parse_json, write_file |

Custom skills can be loaded from a directory via `--skills-dir`, enabling user-defined extraction pipelines.

---

## 7. Prompt Engineering & Multi-Source Fusion

### 7.1 Design Source Map

KGClaw's prompt system synthesizes techniques from multiple established projects:

| Feature | Source | Description |
|---------|--------|-------------|
| Entity description field | LightRAG | Rich per-entity descriptions enhance downstream context |
| Relation keywords + description | LightRAG | Multi-field relation representation |
| Naming normalization rules | LightRAG | Title Case, third person, no pronouns, full names preferred |
| Output format safety | LightRAG | Prevent LLM from hallucinating entities from few-shot examples |
| Gleaning second pass | LightRAG | First-round results fed back to find missed entities |
| Schema Canonicalization | edc | Multi-choice mapping of open relations to target ontology |
| Open-to-canonical strategy | edc | Extract openly first, constrain to Schema later |
| Agent circuit breaker | OpenCode | Automatic degradation on consecutive tool failures |

### 7.2 Entity Extraction V2 Prompt Design

The entity extraction prompt is a carefully structured template with these components:

| Component | Content | Origin |
|-----------|---------|--------|
| **Role** | "Knowledge Graph Entity Extraction Specialist" | — |
| **Core Requirements** | Exhaustive coverage + accurate descriptions | LightRAG |
| **Naming Normalization** | Title Case, third person, no pronouns, full names, strip titles | LightRAG |
| **Description Field** | Mandatory per-entity text description | LightRAG |
| **Output Format** | JSON Schema constrained | — |
| **Format Safety** | Prevent LLM from extracting entities from few-shot template examples | LightRAG |
| **Few-shot Examples** | Dynamically generated from ontology + document samples | — |

### 7.3 Gleaning Prompt Design

```
First-round extracted entities: [{name: "Entity A", type: "Person"}, ...]

┌─────────────────────────────────────────────┐
│ Gleaning Prompt:                             │
│                                              │
│ Based on the first-round results, find       │
│ entities that were MISSED.                   │
│                                              │
│ Key rules:                                    │
│ 1. Only output missed entities               │
│ 2. Fix formatting errors in existing entities │
│ 3. Low-confidence items are acceptable        │
│ 4. If nothing missed, return {"entities": []} │
└─────────────────────────────────────────────┘
                        │
                        ▼
          New entities + enhanced descriptions for existing ones
```

---

## 8. Ontology Building System

### 8.1 Multi-Paradigm Ontology Builder

KGClaw implements a `OntologyBuilder` class (`kgclaw/ontology_builder.py`, 1266 lines) that provides 6 distinct ontology construction paradigms inspired by the LLM4Onto taxonomy (Ouyang, Tang & Huang, *Semantic Web Journal*):

| Paradigm | Mode Key | Stages | Description |
|----------|----------|--------|-------------|
| **T-O** | `text-to-ontology` | 1 | Full text → LLM → structured ontology. Enhanced with type-list detection and hierarchy-focused prompts. |
| **R-O** | `relation-to-ontology` | 2 | Stage 1: relation discovery from text. Stage 2: cluster relations → induce entity types → build ontology with domain/range. Falls back to T-O if < 2 relations found. |
| **HT-R-O** | `ht-relation-to-ontology` | 1+retry | Detects input type (type list vs. raw text). First pass builds hierarchy with parent fields. If < 3 entity types produced, retries with broader prompt emphasizing naming pattern analysis. |
| **D-O** | `dense-ontology` | 3 | Stage 1: exhaustive type extraction (30-50 candidates). Stage 2: 3-4 level hierarchy with parent fields + 8-15 cross-type relations. Targets maximum Graph F1. |
| **Affinity** | `affinity-clustering` | 5 | spaCy noun extraction → TF-IDF char-wb vectorization → Affinity Propagation (sklearn) → LLM cluster naming (15/cluster batch) → multi-round LLM merge (up to 3) → LLM relation discovery. Falls back to T-O if < 10 nouns extracted or < 2 clusters formed. |
| **Auto** | `auto` | — | Heuristic selection based on noun density (> 5 → affinity), avg text length (> 10K + ≤ 3 docs → HT-R-O, > 3K → R-O), single short doc → T-O. |

**Shared infrastructure:**
- `_create_agent(name, system_prompt)`: Lightweight agent factory with no tools, single tool-call cap
- `_result_to_ontology(result, paradigm)`: LLM output → `Ontology` object with:
  - Entity type name normalization (Title Case, whitespace cleanup)
  - Deduplication by name
  - Parent reference validation (clears invalid parents)
  - **Implicit parent inference**:
    - Suffix matching: "Lung Cancer" → parent "Cancer" (if "Cancer" exists in entity types)
    - Last-word matching: "Business Organization" → parent "Organization"

**Affinity Clustering detailed pipeline:**
```
Documents
  → spaCy noun extraction (NOUN + PROPN + noun chunks, dedup)
  → TF-IDF char-wb vectorization (ngram_range=(2,4), max_features=1000)
  → Affinity Propagation (damping=0.9, convergence_iter=30, random_state=42)
  → Filter clusters (min size = max(2, 1% of total nouns))
  → Top 30 largest clusters kept
  → LLM names clusters (batches of 15, with {name, description, parent})
  → Multi-round LLM merge (up to 3 rounds, merges similar-named clusters)
  → LLM relation discovery between named types
  → Structured Ontology with entity_types + relation_types
```

### 8.2 Dataset Presets

The `kgclaw.presets` package provides pre-built `Ontology` objects for 6 common knowledge graph evaluation datasets:

```
presets/
├── __init__.py     # DatasetPreset dataclass, registry, build_ontology()
├── webnlg.py       # WebNLG (RDF-to-text benchmark)
├── nyt_repo.py     # NYT (New York Times relation extraction)
├── kochet.py       # CoNLL04 (named entity + relation)
├── sredfm.py       # SRedFM (sentence-level relation extraction)
├── rebel.py        # Rebel (relation extraction by end-to-end BERT)
└── wiki_nre.py     # Wiki-NRE (Wikipedia relation extraction)
```

**Key design decisions:**
- Each preset is a `DatasetPreset` dataclass with `entity_types`, `relation_types`, language, and entity naming convention
- Auto-registered via `register()` decorator at module import time
- `build_ontology(name)` returns a fully-structured `Ontology` with `is_structured == True`
- `Harness.set_ontology_structured(ontology)` bypasses Phase 1 (LLM ontology analysis), using the dataset's own label system directly
- Sub-modules are auto-discovered and imported on `import kgclaw.presets`

### 8.3 Structured Trace Writer

The `TraceWriter` class (`kgclaw/tracer.py`) provides thread-safe JSONL tracing for debugging and analysis:

```
.kgclaw/traces/
└── build_20260713T143052_abc123def456.jsonl
```

Each line is a JSON event with timestamp and event type. Events tracked:
- `trace_start` / `trace_end`: Workflow lifecycle with elapsed time
- `llm_request` / `llm_response`: Full prompt (prompt_chars + full text) and response (token counts + content)
- `tool_call` / `tool_result`: Tool name, arguments, success/failure, output data
- `phase`: Phase transitions (start/complete/failed) with metadata
- `workflow_*`: Custom workflow events

All string fields are safely serialized and truncated at 50K characters. Files are flushed after every write for crash-recovery inspection. The tracer is thread-safe (threading.Lock on all writes).

---

## 9. Interaction Modes

### 8.1 CLI Mode

```bash
# Built-in template + example data
kgclaw run -T 1 -d examples/character_relations.txt

# Natural language ontology + multi-document
kgclaw run -t "Entity: Person, Location  Relation: lives_in, born_in" -d docs.txt

# File ontology + directory batch + JSON output
kgclaw run -O ontology.yaml -D my_docs/ -f json

# Fast strategy + no co-occurrence
kgclaw run --strategy fast --no-co-occurrence -d simple.txt

# Debug mode (full logs)
kgclaw run --debug -T 1 -d docs.txt
```

### 8.2 Interactive REPL

The REPL offers a Claude Code-style terminal experience with 22+ slash commands, natural language chat, streaming Markdown rendering, and agent proposals:

```
> /load examples/character_relations.txt
  [OK] Loaded character_relations.txt (3,463 lines, 362,621 bytes)

> /ontology Entity Types: Person\nRelation Types: father, son, teacher
  [OK] Ontology set

> /refine Add Author and Editor as entity types, the relation extraction missed cross-sentence ones
  (LLM analyzes last build, proposes ontology changes and strategy adjustments)

> /rebuild
  (Rebuilds with updated ontology, shows ontology diff)

> /history
  (Shows git build history)

> /rollback abc1234
  (Restores to a previous version)
```

### 8.3 Python API

```python
from kgclaw import Harness, HarnessConfig, LLMConfig
from kgclaw.ui import make_progress_callback

config = HarnessConfig(
    llm=LLMConfig(model="gpt-4o", api_key="sk-..."),
    enable_gleaning=True,
    chunk_size=2000,
    max_concurrent_agents=4,
)

harness = Harness(config)

# Register progress callback
cb, stop = make_progress_callback()
harness.on_event(cb)

# Load data and set ontology
harness.load_documents(["docs.txt"])
harness.set_ontology("Entity Types: Person\nRelation Types: father, son")

# Run and export
result = harness.run(strategy="auto")
harness.export_nt("output.nt")
harness.export_json("output.json")
stop()
```

---

## 10. Sandbox Execution & Safety

### 9.1 Architecture

When the code strategy is active or when an agent uses the `run_python` tool, generated Python code passes through a defense-in-depth safety pipeline:

```
Agent-generated code → AST safety audit → subprocess execution (30s timeout)
                              │                    │
                          ┌───┴───┐          ┌────┴────┐
                       Pass     Block       Success   Timeout/Error
                          │       │            │         │
                          ▼       ▼            ▼         ▼
                      Execute   Return      stdout    Kill +
                      in sub-   error       stderr    report
                      process             (100K cap)
```

### 9.2 AST Safety Rules

**Forbidden imports (19 modules)**: `os`, `subprocess`, `socket`, `requests`, `urllib`, `shutil`, `ctypes`, `multiprocessing`, `signal`, `pty`, `fcntl`, `posix`, `grp`, `pwd`, `crypt`, `importlib`, `sys`, `builtins`

**Forbidden calls (5 functions)**: `eval()`, `exec()`, `compile()`, `__import__()`, `breakpoint()`

**Blocked sandbox escape patterns**:
- Subscript access: `__builtins__["eval"]`, `__builtins__.__dict__[...]`
- Class-hierarchy navigation: `().__class__.__bases__[0].__subclasses__()`
- Attribute chains touching `__globals__`, `__code__`, `__closure__`, `__dict__`
- Restricted calls with dangerous builtin arguments: `getattr(__builtins__, ...)`, `vars(__builtins__)`

**Allowed modules**: `json`, `csv`, `re`, `collections`, `itertools`, `math`, `pathlib`, `io`, `string`, `textwrap`, `datetime`, `typing`, `dataclasses`, `enum`

**Important**: AST-based analysis is a best-effort defense designed for agent-generated code, not untrusted user input. It is not a security guarantee against malicious actors with deep Python knowledge.

---

## 11. Performance & Monitoring

### 10.1 Weighted Progress Bar

The 8 phases contribute to a weighted progress total of 100%, with entity extraction dominating at 40%:

| Phase | Weight |
|-------|--------|
| entity_extraction | 40% |
| quality_check | 20% |
| relation_extraction | 15% |
| auto_discover_ontology | 5% |
| ontology_analysis | 5% |
| co_occurrence | 5% |
| structured_extraction | 5% |
| triple_construction | 5% |

### 10.2 Anti-Fake-Completion Lock

- Progress is **capped at 98%** until the `workflow_complete` event fires, ensuring the progress bar never falsely shows 100%.
- `token_usage`, `agent_call_start`, and `agent_call_end` events force a progress bar refresh.
- When the API doesn't return `usage` data, token counts are estimated from character counts (4 chars ≈ 1 token).
- When progress reaches the cap, the display shows "Processing... | ↓ X tokens ↑ Y tokens | current phase name".

### 10.3 Logging System

| Mode | File Output | Console Output |
|------|-------------|---------------|
| Normal | INFO+ → `.kgclaw/logs/kgclaw.log` (10 MB × 3 rotations) | WARNING+ |
| Debug | DEBUG+ → `.kgclaw/logs/kgclaw.log` (includes full prompts/responses) | INFO+ |

The logger uses Python's `RotatingFileHandler` with 3-backup rotation. In debug mode, the full LLM request/response payloads are written to disk, enabling post-hoc analysis and debugging.

---

## 12. Engineering Practices

### 11.1 Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.10+ | Core development |
| **LLM API** | OpenAI SDK ≥ 1.0.0 | Unified LLM interface |
| **Data Models** | Pydantic ≥ 2.0.0 | Type-safe validation and serialization |
| **CLI** | Click ≥ 8.0.0 | Argument parsing |
| **Terminal UI** | Rich ≥ 13.0.0 | Color output, progress bars, Markdown rendering |
| **REPL** | prompt_toolkit ≥ 3.0.0 | Interactive terminal input |
| **Config** | PyYAML ≥ 6.0 | YAML config file I/O |
| **HTTP** | httpx ≥ 0.24.0 | HTTP transport |
| **Document parsing** | pypdf, beautifulsoup4, openpyxl, lxml | Multi-format document loading |
| **Testing** | pytest ≥ 7.0.0, pytest-asyncio | Integration tests (no API key required) |

**Dependency philosophy**: Pure Python, zero external database dependencies. All persistence uses JSON files (`workflow_state.json`) and N-Triples text files.

### 11.2 Concurrency Model

KGClaw uses `ThreadPoolExecutor` for I/O-bound parallelism (LLM API calls). The GIL is released during HTTP I/O, enabling true concurrency. The number of workers is capped at `min(config.max_concurrent_agents, task_count)`.

### 11.3 Error Handling Strategy

- **LLM API errors**: Authentication errors surface immediately with actionable messages. Transient errors benefit from the OpenAI SDK's built-in retry (max 3, exponential backoff).
- **Tool failures**: Individual tool failures don't crash the agent. The circuit breaker activates after 3 consecutive failures.
- **JSON parse failures**: The structured output path uses 5 tiers of fallback before giving up.
- **KeyboardInterrupt**: The `_stop_event` (threading.Event) propagates to all worker threads for graceful shutdown.
- **File I/O errors**: Loaders catch and log individual file failures without aborting directory-level loads.

---

## 13. Acknowledgments

This project was inspired by the following excellent projects:

| Project | Year | Key Ideas Adapted |
|---------|------|-------------------|
| **Claude Code** (Anthropic) | 2024–2026 | Agent Harness architecture, Dynamic Workflows orchestration, interactive REPL design |
| **OpenCode** | 2024–2026 | Agent Harness architecture, Tool registry, circuit breaker, SubAgent spawning |
| **LightRAG** (HKU) | 2024–2025 | Gleaning second pass, entity description fields, naming normalization rules |
| **edc** | 2024 | Open-to-canonical two-stage strategy, Schema Canonicalization, multi-choice mapping |
| **Apple ODKE+** | 2025 | Production-grade ontology-guided KG extraction pipeline, quality review mechanisms |
| **Microsoft GraphRAG** | 2024 | Unstructured text → entities/relations → community detection paradigm |
| **LLM4Onto** (Ouyang, Tang & Huang) | *Semantic Web Journal* | T-O / R-O / HT-R-O paradigm taxonomy, four-dimensional evaluation framework (Literal F1, Fuzzy F1, Continuous F1, Graph F1) — theoretical foundation of KGClaw's multi-paradigm ontology builder |

---

## 14. Summary & Future Directions

### 14.1 Current Capabilities

KGClaw delivers end-to-end automation from unstructured text to structured knowledge graphs:

1. **Natural language ontology**: Users define ontologies conversationally. The system parses, structures, and applies them throughout the pipeline.
2. **Zero-ontology auto-discovery**: When no ontology is provided, the system induces one from the documents using any of 6 building modes.
3. **Multi-paradigm ontology building**: 6 distinct ontology construction paradigms (T-O, R-O, HT-R-O, D-O, Affinity Clustering, Auto) inspired by LLM4Onto, with spaCy noun extraction, Affinity Propagation clustering, LLM-driven cluster naming, and multi-round merge.
4. **Dataset presets**: 6 pre-built ontology definitions for common KG evaluation datasets (WebNLG, NYT, CoNLL04, SRedFM, Rebel, Wiki-NRE) that bypass LLM-based analysis.
5. **Iterative refinement**: Users provide natural language feedback; the system analyzes results, proposes concrete changes, and applies them with one click.
6. **Adaptive strategy**: Automatic selection of the optimal extraction strategy based on data characteristics.
7. **Parallel extraction**: ThreadPool parallel chunked processing with linear speedup for large document collections.
8. **Quality assurance**: Gleaning second pass + Schema Canonicalization + 4-level fuzzy matching + domain/range enforcement.
9. **Smart incremental rebuild**: Session resume with MD5-based file change detection. When nothing changed, cached results are reused instantly. When files or ontology changed, the system reports exactly what changed and why a rebuild is needed.
10. **Engineering completeness**: Circuit breaker, weighted progress bar, structured logging, session persistence, graceful KeyboardInterrupt handling, Git version management with rollback, structured JSONL tracing for debugging.
11. **Flexible usage**: CLI / Interactive REPL / Python API.

### 14.2 Future Directions

- **Graph database integration**: Direct writes to gStore / Neo4j / TuGraph for large-scale graph queries.
- **Multi-modal support**: Entity recognition from images (OCR + vision models), enhanced table semantic understanding.
- **Human-in-the-loop**: Integrate manual annotation feedback loops into the quality review phase.
- **Distributed scaling**: Ray / Celery-based distributed parallel processing for ultra-large corpora.
- **KG quality evaluation**: Built-in automated metrics (entity coverage, relation precision, Schema consistency).
- **Custom embedding models**: Support for local embedding models for entity linking and co-reference resolution.
