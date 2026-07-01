# KGClaw — Knowledge Graph Construction Harness

**[English](README.md) | [中文](README.zh.md)**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21107149.svg)](https://doi.org/10.5281/zenodo.21107149)

<p align="center">
  <em>An AI Agent Harness system for ontology-driven knowledge graph construction from unstructured text</em>
</p>

---

## Overview

KGClaw is a **Claude Code / OpenCode-inspired Agent Harness** designed to **automatically construct knowledge graphs from unstructured text based on user-provided ontologies**.

Three usage modes:

- **CLI mode** (`kgclaw run`): Provide ontology + documents, automatically run an 8-phase pipeline, output N-Triples / JSON / JSONL
- **Interactive REPL** (`kgclaw interactive`) — **Recommended**: Claude Code-style terminal with streaming chat, slash commands, agent proposals, Ctrl+O verbose toggle
- **Python API** (`from kgclaw import Harness`): Programmatic integration

### Key Features

- **Natural language custom ontology**: Define entity types and relation types in plain natural language — no OWL, RDF, or formal ontology language required. LLM automatically parses it into a structured schema with descriptions, attributes, domain/range constraints, and parent-child hierarchies.
- **Zero-ontology auto-discovery**: Don't have an ontology? Just throw in your documents — KGClaw reads the content and lets the LLM induce a reasonable set of entity types and relation types automatically, then builds the KG from the discovered ontology.
- **Ontology-guided full-pipeline construction**: All 8 phases are driven and constrained by your ontology. Entities and relations are extracted only for defined types. Schema Canonicalization maps open relation names to standard ontology relations. Triple construction enforces domain/range constraints to ensure type consistency.
- **Ontology modification + iterative refinement**: Modify your ontology at any time and rebuild. Use `/refine` with natural language feedback (e.g., "add Author and Editor as entity types") — the LLM analyzes the last build result, proposes concrete ontology changes, strategy adjustments, and prompt improvements. Apply them with one click.
- **Structured data ingestion**: CSV and XLSX files aren't just flattened to text. KGClaw preserves row-column structure and uses the LLM to map columns to ontology types (e.g., "Name" column → Person entity, "Department" column → "works_at" relation), extracting entities and relations directly from tabular data.
- **Co-occurrence graph for implicit relations**: Even when the LLM doesn't explicitly extract a relation between two entities, KGClaw builds a weighted co-occurrence network from paragraph-level entity co-appearances. When domain/range constraints match, co-occurrence pairs are automatically upgraded to ontology relations.
- **Gleaning second-pass catch-up** (LightRAG-inspired): After the first extraction pass, previously extracted entities are fed back to the LLM with a targeted prompt: "What was missed?" This catches truncated, malformed, or overlooked entities and significantly improves recall.
- **Quality check + auto-correction**: A dedicated quality-review phase vets every entity and relation against the ontology, marks flawed items for rejection, proposes type corrections, and performs Schema Canonicalization. The final aggregation automatically applies these corrections.
- **Session resume + intelligent incremental rebuild**: Restart KGClaw and it detects your previous session. It compares file hashes (MD5) against a stored manifest — if nothing changed, cached results are reused instantly. If files were added, modified, or deleted, it tells you exactly what changed and why a rebuild is needed.
- **Build history + rollback**: Every `/run` is automatically committed via Git. Use `/history` to browse past builds and `/rollback <hash>` to restore any historical version. Ontology changes are tracked as separate commits.

---

## Quick Start

### 1. Install

```bash
cd KGClaw
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. First-time Setup

```bash
kgclaw setup
```

An interactive wizard guides you through provider selection → API Key input → model selection. Configuration is saved to `~/.kgclaw/config.yaml`.

### 3. Run

```bash
# Use built-in template + example data
kgclaw run -T 1 -d examples/人物图谱/人物关系图谱原始数据.txt

# Or start the interactive REPL
kgclaw interactive
```

---

## CLI Mode

```bash
# Natural language ontology + document
kgclaw run -t "Entity: Person, Location  Relation: lives_in, born_in" -d docs.txt

# Built-in templates (1=Character Relations, 2=Enterprise, 3=Legislation)
kgclaw run -T 1 -d examples/人物图谱/人物关系图谱原始数据.txt

# Load ontology from file, process a directory
kgclaw run -O ontology.yaml -D my_docs/ -f json

# Fast strategy + disable co-occurrence
kgclaw run --strategy fast --no-co-occurrence -d simple.txt

# Debug mode (all details written to .kgclaw/logs/)
kgclaw run --debug -T 1 -d examples/人物图谱/人物关系图谱原始数据.txt
```

| Option | Description |
|--------|-------------|
| `--ontology` / `-t` | Ontology definition text |
| `--ontology-file` / `-O` | Ontology definition file path |
| `--template` / `-T` | Built-in template: 1=Character Relations, 2=Enterprise, 3=Legislation |
| `--docs` / `-d` | Document paths (can specify multiple) |
| `--docs-dir` / `-D` | Document directory |
| `--output` / `-o` | Output path (default `.kgclaw/output.nt`) |
| `--format` / `-f` | `nt` / `json` / `jsonl` |
| `--model` / `-m` | Override default model |
| `--lang` | UI language: `zh` (default) or `en` |
| `--verbose` / `-v` | Verbose output |
| `--debug` | Write debug logs to file |
| `--strategy` / `-s` | `auto` / `fast` / `standard` / `code` |
| `--co-occurrence` / `--no-co-occurrence` | Enable/disable co-occurrence graph (default: on) |
| `--work-dir` | Working directory (default `.kgclaw`) |

---

## Interactive REPL

The interactive mode provides a Claude Code-style terminal experience. Beyond slash commands, you can also use **natural language** to interact with the AI assistant — for example, type `"加载这个目录下的全部doc文件"` and the agent will find and load all `.docx` files in the directory for you.

```
> /load examples/character_relations.txt
  [OK] Loaded character_relations.txt (3463 lines, 362,621 bytes)

> /ontology Entity Types: Person\nRelation Types: father, son, teacher
  [OK] Ontology set

> /strategy standard
  [OK] Workflow strategy: standard

> /run
  (8-phase pipeline execution with real-time weighted progress bar + token stats)

> /chat This text mainly describes parent-child and friend relationships
  ... (streaming Markdown response, agent proactively proposes actions)
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/load <path>` | Load document files or directories |
| `/docs` | List all loaded documents with metadata |
| `/unload <path\|filename>` | Remove a loaded document |
| `/clear-docs` | Clear all loaded documents |
| `/ontology <text>` | Set ontology definition (natural language supported) |
| `/template <1\|2\|3>` | Use built-in ontology template |
| `/run` | Run full KG construction pipeline |
| `/extract-entities` | Extract entities only |
| `/strategy <auto\|fast\|standard\|code>` | Set workflow strategy |
| `/cooccur` | Toggle co-occurrence graph construction |
| `/output <path>` | Set output file path |
| `/format <nt\|json\|jsonl>` | Set output format |
| `/chat <msg>` | Free-form chat with AI assistant (streaming Markdown) |
| `/verbose` | Toggle verbose mode (show LLM interaction details) |
| `/debug` | Toggle debug logging (all details to file) |
| `/status` | Show current session status |
| `/history` | Show conversation history |
| `/examples` | View built-in example datasets |
| `/config` | View current LLM configuration |
| `/clear` | Clear conversation history |
| `/reset` | Completely reset session |
| `/help` | Show all commands |
| `/quit` | Exit |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+O` | Toggle verbose mode in real-time |
| `Ctrl+C` | Interrupt current operation |
| `Ctrl+D` | Exit REPL |

---

## Architecture

### Adaptive Pipeline

```
User Input (Ontology + Documents/Directory)
        │
        ▼
┌──────────────────────────────────────────────┐
│ Phase 0:   Ontology Auto-Discovery            │  ← LLM induction (when no ontology)
├──────────────────────────────────────────────┤
│ Phase 1:   Ontology Analysis                  │  ← LLM structured schema
├──────────────────────────────────────────────┤
│ Phase 1.5: Agent Code Extraction (optional)   │  ← Agent-generated custom prompt/code
├──────────────────────────────────────────────┤
│ Phase 2:   Entity Extraction (Parallel TPool) │  ← LLM parallel chunks + Gleaning
├──────────────────────────────────────────────┤
│ Phase 2.5: Structured Data Extraction         │  ← CSV/XLSX → col→ontology mapping
├──────────────────────────────────────────────┤
│ Phase 3:   Relation Extraction                │  ← Doc grouping + entity prefilter parallel
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

### Strategy Selection

| Strategy | Use Case | Description |
|----------|----------|-------------|
| `fast` | Few short documents | Single-pass combined entity+relation extraction |
| `standard` | Long text / mixed formats | Full 8-phase pipeline |
| `code` | Large tabular data | Agent generates Python code for sandbox execution |
| `auto` | Uncertain | Automatically selects based on data characteristics |

### Project Structure

```
kgclaw/
├── agent.py                # Agent system (LLM + Tool Use + Stream + SubAgent + Circuit Breaker)
├── cli.py                  # CLI entry (Click + Rich)
├── interactive_app.py      # Interactive REPL (prompt_toolkit + Live/Markdown)
├── models.py               # Pydantic data models
├── config.py               # User configuration (~/.kgclaw/config.yaml)
├── memory.py               # Session memory (msg compaction + workflow persistence)
├── logger.py               # Structured logging (RotatingFileHandler + Debug)
├── loaders.py              # 13 format file loaders + recursive directory scanning
├── sandbox.py              # Sandbox execution (run_python + AST audit + format analysis)
├── i18n.py                 # Internationalization (gettext-style, zh/en)
├── refinement.py           # KG refinement engine (ontology optimization)
├── git_manager.py          # Git version management for build history
├── harness/                # Orchestration engine
│   ├── engine.py           #   Main engine + doc loading + export
│   ├── phases.py           #   8-phase impl + Gleaning + Canonicalization
│   ├── strategies.py       #   auto/fast/code strategies
│   └── helpers.py          #   Chunking/dedup/fuzzy matching/Agent factory
├── tools/                  # Tool system (13 tools)
├── skills/                 # Skill system (5 built-in skills)
├── prompts/                # Prompt templates
├── ui/                     # Shared UI layer
│   ├── progress.py         #   Weighted progress callback factory
│   └── display.py          #   Result display utilities
└── locales/                # Translation files
    └── en/LC_MESSAGES/
        └── kgclaw.po       # English translations
```

Detailed docs: [ARCHITECTURE.md](docs/ARCHITECTURE.md) — system architecture and module reference; [TECHNICAL_OVERVIEW.md](docs/TECHNICAL_OVERVIEW.md) — comprehensive technical whitepaper.

---

## Python API

```python
from kgclaw import Harness, HarnessConfig, LLMConfig

config = HarnessConfig(
    llm=LLMConfig(model="gpt-4o", api_key="sk-..."),
    enable_gleaning=True,
    chunk_size=2000,
)
harness = Harness(config)
harness.load_documents(["docs.txt"])
harness.set_ontology("Entity Types: Person\nRelation Types: father, son")

# Register progress callback
from kgclaw.ui import make_progress_callback
cb, stop = make_progress_callback()
harness.on_event(cb)

result = harness.run()
harness.export_nt("output.nt")
harness.export_json("output.json")
stop()
```

---

## Configuration

### Config file `~/.kgclaw/config.yaml`

```yaml
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

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key |
| `KGCLAW_MODEL` | Model name |
| `KGCLAW_API_BASE` | API endpoint URL |
| `KGCLAW_VERBOSE` | Verbose output (1/true/yes) |
| `KGCLAW_LANG` | UI language (zh/en) |

### Supported LLMs

Any OpenAI-compatible API: OpenAI / DeepSeek / Qwen / Ollama (local) / vLLM / Custom

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v                # 198 tests, no API key required
```

### Dependencies

```
openai >= 1.0.0         # LLM API
pydantic >= 2.0.0       # Data models
rich >= 13.0.0          # Terminal UI
click >= 8.0.0          # CLI
prompt_toolkit >= 3.0.0 # REPL
pyyaml >= 6.0           # Config parsing
httpx >= 0.24.0         # HTTP client
pypdf >= 4.0            # PDF support
beautifulsoup4 >= 4.12  # HTML support
openpyxl >= 3.1         # Excel support
lxml >= 4.9             # XML parsing
```

Pure Python, no external database dependencies.

---

## Acknowledgments

This project was inspired by the following excellent projects:

- **Claude Code** (Anthropic, 2024-2026) — Agent Harness architecture, Dynamic Workflows orchestration pattern
- **OpenCode** (2024-2026) — Agent Harness architecture, Tool registry, Permission system
- **LightRAG** (HKU, 2024-2025) — Gleaning, entity description fields, naming normalization
- **edc** (2024) — Open extraction→standardization, Schema Canonicalization
- **Apple ODKE+** (2025) — Production-grade ontology-guided KG extraction pipeline
- **Microsoft GraphRAG** (2024) — Unstructured text→entities/relations→community detection

---

## License

MIT
