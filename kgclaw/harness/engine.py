"""
Harness engine — the main orchestrator for KG construction workflows.

The Harness is the central coordinator that:
1. Receives user input (ontology + documents)
2. Plans and executes a multi-phase KG construction workflow
3. Spawns agents (and subagents) for each phase
4. Manages parallel extraction across document chunks
5. Aggregates results and produces the final KG output
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from ..agent import Agent, AgentConfig
from ..git_manager import GitManager
from ..logger import get_logger
from ..memory import Memory
from ..models import (
    Document,
    EntityType,
    ExtractionResult,
    HarnessConfig,
    LLMConfig,
    Ontology,
    RelationType,
    WorkflowState,
)
from ..prompts.system_prompts import (
    SYSTEM_PROMPT_ONTOLOGY_ANALYZER,
    SYSTEM_PROMPT_ORCHESTRATOR,
    build_ontology_analysis_prompt,
)
from ..skills import get_skill, get_default_pipeline_skills

from .helpers import _HarnessHelpers
from .strategies import _HarnessStrategies
from .phases import _HarnessPhases


# ─── Shared phase metadata (used by CLI and REPL progress displays) ──────────

# Internal raw labels — use phase_label() for translated versions.
_RAW_PHASE_LABELS: dict[str, str] = {
    "ontology_analysis": "本体分析",
    "auto_discover_ontology": "本体自动发现",
    "entity_extraction": "实体抽取",
    "relation_extraction": "关系抽取",
    "co_occurrence": "共现图谱",
    "structured_extraction": "结构化数据抽取",
    "quality_check": "质量审核",
    "triple_construction": "三元组构造",
}

# Backward-compatible: PHASE_LABELS is still a dict (consumers that import it).
# New code should use phase_label() for translated output.
PHASE_LABELS: dict[str, str] = _RAW_PHASE_LABELS


def phase_label(key: str) -> str:
    """Return the translated label for a phase key."""
    from ..i18n import _
    raw = _RAW_PHASE_LABELS.get(key, key)
    return _(raw)


PHASE_WEIGHTS: dict[str, int] = {
    "auto_discover_ontology": 5,
    "ontology_analysis": 5,
    "entity_extraction": 40,
    "relation_extraction": 15,
    "co_occurrence": 5,
    "structured_extraction": 5,
    "quality_check": 20,
    "triple_construction": 5,
}

TOTAL_PHASE_WEIGHT = sum(PHASE_WEIGHTS.values())


class Harness(_HarnessPhases, _HarnessStrategies, _HarnessHelpers):
    """
    The main KG construction harness.

    Usage:
        config = HarnessConfig(llm=LLMConfig(model="gpt-4o", api_key="..."))
        harness = Harness(config)
        harness.load_documents(["path/to/doc1.txt", "path/to/doc2.txt"])
        harness.set_ontology("人物: 人物\n关系: 生父, 儿子, 现妻\n...")
        result = harness.run()
        print(harness.export_nt())
    """

    def __init__(self, config: Optional[HarnessConfig] = None):
        self.config = config or HarnessConfig()
        self.llm_config = self.config.llm
        self.memory = Memory(work_dir=self.config.work_dir)
        self.log = get_logger()
        self._stop_event = threading.Event()

        # Initialize orchestrator agent
        self._orchestrator: Optional[Agent] = None
        self._event_callbacks: list[Callable[[str, dict], None]] = []

        # Structured data from tabular files
        self._structured_rows: list[dict[str, Any]] = []
        # Phase results accumulator
        self._phase_outputs: dict[str, dict[str, Any]] = {}
        # Token tracking (accumulated across all agents)
        self._agent_token_state: dict[str, tuple[int, int]] = {}
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

        # Load custom skills if configured
        if self.config.skills_dir:
            from ..skills import SkillRegistry
            SkillRegistry.discover_from_directory(self.config.skills_dir)

        # Ensure API key is available
        if not self.llm_config.api_key:
            self.llm_config.api_key = os.environ.get("OPENAI_API_KEY", "")

    @classmethod
    def from_env(cls, work_dir: str = ".kgclaw") -> "Harness":
        """Create a Harness from environment variables."""
        config = HarnessConfig(
            llm=LLMConfig(
                model=os.environ.get("KGCLAW_MODEL", "deepseek-v4-flash"),
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                api_base=os.environ.get("KGCLAW_API_BASE", "https://api.deepseek.com/v1"),
            ),
            work_dir=work_dir,
            verbose=os.environ.get("KGCLAW_VERBOSE", "").lower() in ("1", "true", "yes"),
        )
        return cls(config)

    def on_event(self, callback: Callable[[str, dict], None]):
        """Register an event callback, wrapping to accumulate global token stats."""
        def _wrapped(event_type: str, data: dict):
            if event_type == "token_usage":
                aid = data.get("_agent_id", "")
                p = data.get("prompt_tokens", 0)
                c = data.get("completion_tokens", 0)
                prev_p, prev_c = self._agent_token_state.get(aid, (0, 0))
                delta_p = max(0, p - prev_p)
                delta_c = max(0, c - prev_c)
                self._total_prompt_tokens += delta_p
                self._total_completion_tokens += delta_c
                self._agent_token_state[aid] = (p, c)
                callback("token_usage", {
                    "prompt_tokens": self._total_prompt_tokens,
                    "completion_tokens": self._total_completion_tokens,
                })
            else:
                callback(event_type, data)
        self._event_callbacks.append(_wrapped)

    def _emit(self, event_type: str, data: dict[str, Any]):
        for cb in self._event_callbacks:
            try:
                cb(event_type, data)
            except Exception:
                pass

    def _get_orchestrator(self) -> Agent:
        if self._orchestrator is None:
            orch_config = AgentConfig(
                name="orchestrator",
                system_prompt=SYSTEM_PROMPT_ORCHESTRATOR,
                tools=[],
                max_tool_calls=20,
            )
            self._orchestrator = Agent(orch_config, self.memory, self.llm_config)
            self._orchestrator.on_event(lambda et, d: self._emit(et, d))
        return self._orchestrator

    # ─── Document Loading ───────────────────────────────────────────────────

    def load_documents(self, paths: list[str]) -> list[Document]:
        """Load documents from file paths using registered format loaders.

        Supports directories (recursive) and all registered formats
        (txt, md, jsonl, docx, pdf, html, csv, xlsx, etc.).
        Preserves structured data (raw_rows, is_tabular) for tabular files.
        """
        from ..loaders import get_loader, load_directory

        docs = []
        expanded_paths = []

        for path in paths:
            p = Path(path)
            if not p.exists():
                self._emit("warning", {"message": f"File not found: {path}"})
                continue
            if p.is_dir():
                try:
                    dir_docs = load_directory(
                        str(p),
                        recursive=True,
                        exclude_patterns=[
                            ".kgclaw/**",
                            ".git/**", "__pycache__/**", ".venv/**", "venv/**",
                        ],
                    )
                    for ld in dir_docs:
                        expanded_paths.append(ld.source)
                        doc = Document(
                            text=ld.text,
                            source=ld.source,
                            metadata=ld.metadata,
                        )
                        self._stamp_doc_hash(doc, Path(ld.source))
                        docs.append(doc)
                    self._emit("info", {
                        "message": f"Loaded {len(dir_docs)} files from directory: {path}",
                    })
                except Exception as e:
                    self._emit("warning", {"message": f"Directory load failed: {path}: {e}"})
            else:
                expanded_paths.append(str(p))

        for path in expanded_paths:
            p = Path(path)
            ext = p.suffix.lower()
            loader = get_loader(ext)
            if loader:
                try:
                    ld = loader(str(p))
                    doc = Document(
                        text=ld.text,
                        source=ld.source,
                        metadata=ld.metadata,
                    )
                    self._stamp_doc_hash(doc, p)
                    docs.append(doc)
                except Exception as e:
                    self._emit("warning", {
                        "message": f"Loader failed for {path}: {e}, trying raw read"
                    })
                    try:
                        content = p.read_text(encoding='utf-8', errors='replace')
                        doc = Document(
                            text=content,
                            source=str(p),
                            metadata={"filename": p.name, "ext": ext, "path": str(p)},
                        )
                        self._stamp_doc_hash(doc, p)
                        docs.append(doc)
                    except Exception:
                        self._emit("warning", {"message": f"Cannot read: {path}"})
            else:
                try:
                    content = p.read_text(encoding='utf-8', errors='replace')
                except Exception:
                    content = p.read_text(encoding='utf-8', errors='ignore')
                doc = Document(
                    text=content,
                    source=str(p),
                    metadata={"filename": p.name, "ext": ext, "path": str(p)},
                )
                self._stamp_doc_hash(doc, p)
                docs.append(doc)

        self._structured_rows = []
        for d in docs:
            if d.metadata.get("is_tabular") and "raw_rows" in d.metadata:
                for row in d.metadata["raw_rows"]:
                    row["_source"] = d.source
                    self._structured_rows.append(row)

        self.memory.init_workflow()
        self.memory.workflow.documents = docs
        self.memory.save_workflow()

        self._emit("documents_loaded", {
            "count": len(docs),
            "paths": expanded_paths,
            "tabular": len(self._structured_rows),
        })
        return docs

    def load_texts(self, texts: list[str]) -> list[Document]:
        """Load documents from raw text strings."""
        docs = [Document(text=t, source=f"input_{i}") for i, t in enumerate(texts)]
        self.memory.init_workflow()
        self.memory.workflow.documents = docs
        self.memory.save_workflow()
        self._structured_rows = []  # text inputs are never tabular
        return docs

    # ─── Document Unloading ──────────────────────────────────────────────────

    def unload_document(self, source: str) -> bool:
        """Remove a single loaded document by its source path.

        Returns True if the document was found and removed.
        Emits 'documents_unloaded' event.
        """
        removed = self.memory.remove_document(source)
        if removed:
            self._refresh_structured_rows()
            remaining = len(self.memory.workflow.documents) if self.memory.workflow else 0
            self._emit("documents_unloaded", {
                "sources": [source], "count": 1, "remaining": remaining,
            })
        return removed

    def unload_documents(self, sources: list[str]) -> int:
        """Remove multiple documents by their source paths.

        Returns the count of documents actually removed.
        """
        removed = self.memory.remove_documents(sources)
        if removed > 0:
            self._refresh_structured_rows()
            remaining = len(self.memory.workflow.documents) if self.memory.workflow else 0
            self._emit("documents_unloaded", {
                "sources": sources, "count": removed, "remaining": remaining,
            })
        return removed

    def clear_documents(self) -> int:
        """Remove all loaded documents.

        Returns the count of documents removed.
        Emits 'documents_cleared' event.
        """
        count = self.memory.clear_documents()
        if count > 0:
            self._structured_rows = []
            self._emit("documents_cleared", {"count": count})
        return count

    def list_documents(self) -> list[dict]:
        """Return a summary list of all loaded documents.

        Each dict: source, filename, chars, ext, size, is_tabular.
        """
        return self.memory.get_loaded_documents()

    def _refresh_structured_rows(self):
        """Recalculate _structured_rows from the current document set."""
        self._structured_rows = []
        if not self.memory.workflow:
            return
        for doc in self.memory.workflow.documents:
            meta = doc.metadata or {}
            if meta.get("is_tabular") and "raw_rows" in meta:
                for row in meta["raw_rows"]:
                    row["_source"] = doc.source
                    self._structured_rows.append(row)

    # ─── Document Hash Stamping ──────────────────────────────────────────────

    @staticmethod
    def _stamp_doc_hash(doc: Document, file_path: Path):
        """Compute and stamp content hash + file metadata on a Document.

        Adds content_hash (MD5), file_mtime, and file_size to doc.metadata.
        """
        try:
            stat = file_path.stat()
            doc.metadata["file_mtime"] = stat.st_mtime
            doc.metadata["file_size"] = stat.st_size
        except (IOError, OSError):
            doc.metadata["file_mtime"] = 0
            doc.metadata["file_size"] = 0
        try:
            content_bytes = file_path.read_bytes()
            doc.metadata["content_hash"] = hashlib.md5(content_bytes, usedforsecurity=False).hexdigest()
        except (IOError, OSError):
            doc.metadata["content_hash"] = ""

    # ─── Ontology Setting ───────────────────────────────────────────────────

    def set_ontology(self, ontology_raw: str) -> Ontology:
        """Set the ontology from a raw text description, with eager LLM analysis.

        When an LLM is configured, the raw text is immediately analyzed to produce
        structured EntityType and RelationType objects.  If the LLM is unavailable
        or analysis fails, the raw text is stored as-is and analysis is deferred
        to run()'s Phase 1.
        """
        # Always start with raw definition (the fallback)
        ontology = Ontology(raw_definition=ontology_raw)

        if self._can_analyze_ontology():
            try:
                structured = self._analyze_ontology_raw(ontology_raw)
                if structured:
                    ontology = structured
                    self._emit("ontology_analysis_result", {
                        "success": True,
                        "entity_types": [et.model_dump() for et in structured.entity_types],
                        "relation_types": [rt.model_dump() for rt in structured.relation_types],
                    })
                else:
                    self._emit("ontology_analysis_result", {
                        "success": False,
                        "error": "LLM returned no structured output — deferred to run()",
                    })
            except Exception as e:
                self.log.warning(f"Eager ontology analysis failed: {e}, deferring to run()")
                self._emit("ontology_analysis_result", {
                    "success": False,
                    "error": str(e),
                })

        # Ensure a workflow exists (init_workflow is normally called by
        # load_documents, but set_ontology may be called first in some flows).
        if self.memory._workflow is None:
            self.memory.init_workflow()
        self.memory.workflow.ontology = ontology
        self.memory.save_workflow()
        return ontology

    def set_ontology_structured(self, ontology: Ontology):
        """Set a pre-built Ontology object directly."""
        if self.memory._workflow is None:
            self.memory.init_workflow()
        self.memory.workflow.ontology = ontology
        self.memory.save_workflow()
        self._emit("phase_start", {
            "phase": "ontology_set",
            "entity_types": len(ontology.entity_types),
            "relation_types": len(ontology.relation_types),
        })

    # ─── Ontology Analysis (eager, reusable) ──────────────────────────────────

    def _can_analyze_ontology(self) -> bool:
        """Return True if an LLM is configured and available for analysis."""
        return bool(self.llm_config and self.llm_config.api_key)

    def _analyze_ontology_raw(self, raw_text: str) -> Optional[Ontology]:
        """Analyze raw ontology text via LLM and return a structured Ontology, or None.

        Extracted from _run_phase_ontology_analysis so it can be called eagerly
        from set_ontology() or lazily from run()'s Phase 1.
        """
        if not raw_text or not raw_text.strip():
            return None

        try:
            skill = get_skill("ontology_analyzer", self.llm_config)
            if skill is None:
                return None

            agent = self._create_skill_agent("ontology_analyzer_eager", skill)
            prompt = build_ontology_analysis_prompt(raw_text)
            result = agent.run_structured(prompt, skill.get_output_schema())

            # Retry once on empty result
            if not result:
                retry_prompt = (
                    f"{prompt}\n\n"
                    "**重要**: 你必须只返回一个 JSON 对象。entity_types 和 relation_types 不能为空。"
                )
                result = agent.run_structured(retry_prompt, skill.get_output_schema())

            if not result:
                return None

            entity_types = [
                EntityType(
                    name=et.get("name", ""),
                    description=et.get("description", ""),
                    parent=et.get("parent"),
                    attributes=et.get("attributes", {}),
                )
                for et in result.get("entity_types", [])
            ]
            relation_types = [
                RelationType(
                    name=rt.get("name", ""),
                    description=rt.get("description", ""),
                    domain=rt.get("domain"),
                    range=rt.get("range"),
                    inverse=rt.get("inverse"),
                )
                for rt in result.get("relation_types", [])
            ]

            return Ontology(
                name=result.get("ontology_name", "user_defined"),
                description=result.get("description", ""),
                entity_types=entity_types,
                relation_types=relation_types,
                raw_definition=raw_text,
            )
        except Exception as e:
            self.log.warning(f"Ontology analysis failed: {e}")
            return None

    # ─── Main Run ────────────────────────────────────────────────────────────

    def run(
        self,
        ontology_raw: Optional[str] = None,
        documents: Optional[list[Document]] = None,
        skills: Optional[list[str]] = None,
        strategy: str = "auto",
        enable_co_occurrence: bool = True,
    ) -> ExtractionResult:
        """
        Execute the full KG construction workflow.

        Args:
            ontology_raw: Raw ontology definition text (or None if already set)
            documents: Documents to process (or None if already loaded)
            skills: Skill pipeline to use (default: standard 5-phase pipeline)
            strategy: Workflow strategy: "auto", "fast", "standard", or "code"
            enable_co_occurrence: Whether to build co-occurrence graph

        Returns:
            ExtractionResult containing all extracted entities, relations, and triples
        """
        self._stop_event.clear()
        skill_names = skills or get_default_pipeline_skills()

        # Ensure a workflow exists (may be None if run() is called directly
        # without prior load_documents() or set_ontology()).
        if self.memory._workflow is None:
            self.memory.init_workflow()

        # Only set raw ontology from the parameter when the workflow doesn't
        # already have a structured one (e.g. from eager analysis in set_ontology).
        if ontology_raw:
            existing = self.memory.workflow.ontology
            if existing is None or not existing.is_structured:
                self.memory.workflow.ontology = Ontology(raw_definition=ontology_raw)
                self.memory.save_workflow()

        if documents:
            self.memory.workflow.documents = documents
            self.memory.save_workflow()

        ontology = self.memory.workflow.ontology
        docs = self.memory.workflow.documents

        if not docs:
            return ExtractionResult()

        # Auto-detect strategy based on data characteristics
        effective_strategy = strategy
        if strategy == "auto" and docs:
            effective_strategy = self._auto_detect_strategy(docs)

        self.log.info(f"Workflow strategy: {effective_strategy} (requested: {strategy})")

        # ── Fast Path ──
        if effective_strategy == "fast":
            return self._run_fast_path(ontology, docs)

        # ── Code Path ──
        if effective_strategy == "code":
            return self._run_code_path(ontology, docs)

        # ── Standard Path: full multi-phase pipeline ──
        self.log.workflow_start(
            ontology_len=len(ontology.raw_definition) if ontology and ontology.raw_definition else 0,
            doc_count=len(docs),
            skills=skill_names,
            model=self.llm_config.model,
        )
        self._emit("workflow_start", {
            "skills": skill_names,
            "doc_count": len(docs),
            "ontology": ontology.raw_definition[:200] if ontology else "None",
        })

        # ── Phase 0: Auto-discover ontology if none provided ────────────────
        if not ontology or not ontology.raw_definition:
            self._emit("phase_start", {"phase": "auto_discover_ontology"})
            console_text = "\n".join([d.text[:3000] for d in docs[:3]])
            discover_prompt = f"""请分析以下文档内容，自动归纳总结出合理的知识图谱本体定义。

你需要：
1. 识别文档中出现的核心实体类型
2. 识别这些实体之间存在的关系类型
3. 生成一个完整的、可直接用于知识抽取的本体定义

## 文档内容（前 3 篇）
{console_text[:5000]}

请输出 JSON 格式的本体定义。"""
            try:
                skill = get_skill("ontology_analyzer", self.llm_config)
                agent_cfg = AgentConfig(
                    name="auto_discover",
                    system_prompt=SYSTEM_PROMPT_ONTOLOGY_ANALYZER,
                    tools=[],
                    max_tool_calls=1,
                )
                agent = Agent(agent_cfg, self.memory, self.llm_config)
                agent.on_event(lambda et, d: self._emit(et, d))
                result = agent.run_structured(discover_prompt, skill.get_output_schema())
                if result:
                    entity_types = [
                        EntityType(name=et.get("name", ""), description=et.get("description", ""))
                        for et in result.get("entity_types", [])
                    ]
                    relation_types = [
                        RelationType(name=rt.get("name", ""), description=rt.get("description", ""),
                                     domain=rt.get("domain"), range=rt.get("range"))
                        for rt in result.get("relation_types", [])
                    ]
                    ontology = Ontology(
                        name=result.get("ontology_name", "auto_discovered"),
                        description=result.get("description", ""),
                        entity_types=entity_types,
                        relation_types=relation_types,
                        raw_definition=result.get("extraction_guide", ""),
                    )
                    self.memory.workflow.ontology = ontology
                    self.memory.save_workflow()
                    self.memory.export_ontology()
                    self._emit("phase_complete", {
                        "phase": "auto_discover_ontology",
                        "entity_types": len(entity_types),
                        "relation_types": len(relation_types),
                    })
                else:
                    self._emit("phase_failed", {
                        "phase": "auto_discover_ontology",
                        "error": "无法自动推断本体",
                    })
                    return ExtractionResult()
            except Exception as e:
                self._emit("phase_failed", {
                    "phase": "auto_discover_ontology",
                    "error": str(e),
                })
                return ExtractionResult()

        # ── Phase 1: Ontology Analysis ─────────────────────────────────────
        if "ontology_analyzer" in skill_names and ontology and ontology.raw_definition:
            if ontology.is_structured:
                # Already analyzed eagerly by set_ontology() — skip Phase 1
                self._emit("phase_start", {"phase": "ontology_analysis"})
                self._emit("phase_complete", {
                    "phase": "ontology_analysis",
                    "entity_types": len(ontology.entity_types),
                    "relation_types": len(ontology.relation_types),
                    "note": "skipped — already analyzed eagerly",
                })
                self._phase_outputs["ontology_analysis"] = {
                    "ontology_json": {
                        "entity_types": [et.model_dump() for et in ontology.entity_types],
                        "relation_types": [rt.model_dump() for rt in ontology.relation_types],
                    },
                    "extraction_guide": ontology.to_extraction_guide(),
                }
            else:
                phase = self._run_phase_ontology_analysis(ontology.raw_definition)
                self.memory.add_phase_result(phase)
                if phase.output:
                    self._phase_outputs["ontology_analysis"] = {
                        "ontology_json": phase.output.metadata.get("structured_ontology", {}),
                        "extraction_guide": phase.output.metadata.get("extraction_guide", ""),
                    }
            # Export standalone ontology files after analysis
            self.memory.export_ontology()

        # ── Phase 1.5: Agent-guided code extraction (optional) ──────────────
        if ontology and "agent_code_extraction" in skill_names:
            phase = self._run_phase_agent_code_extraction(docs, ontology)
            if phase.output:
                if phase.output.entities:
                    self._phase_outputs.setdefault("entity_extraction", {}) \
                        .setdefault("entities", []).extend([e.model_dump() for e in phase.output.entities])
                if phase.output.relations:
                    self._phase_outputs.setdefault("relation_extraction", {}) \
                        .setdefault("relations", []).extend([r.model_dump() for r in phase.output.relations])
                self.memory.add_phase_result(phase)

        # ── Phase 2: Entity Extraction (fan-out across chunks) ──────────────
        if "entity_extractor" in skill_names:
            phase = self._run_phase_entity_extraction(docs, ontology)
            self.memory.add_phase_result(phase)
            if phase.output:
                existing = self._phase_outputs.get("entity_extraction", {}).get("entities", [])
                existing.extend([e.model_dump() for e in phase.output.entities])
                self._phase_outputs["entity_extraction"] = {"entities": existing}

        # ── Phase 2.5: Structured Data Extraction (tabular files) ───────────
        if self._structured_rows and ontology:
            phase = self._run_phase_structured_extraction(ontology)
            if phase.output:
                if phase.output.entities:
                    self._phase_outputs.setdefault("entity_extraction", {}) \
                        .setdefault("entities", []).extend([e.model_dump() for e in phase.output.entities])
                if phase.output.relations:
                    self._phase_outputs.setdefault("relation_extraction", {}) \
                        .setdefault("relations", []).extend([r.model_dump() for r in phase.output.relations])
                self.memory.add_phase_result(phase)

        # ── Phase 3: Relation Extraction ────────────────────────────────────
        if "relation_extractor" in skill_names:
            existing_entities = self._phase_outputs.get("entity_extraction", {}).get("entities", [])
            phase = self._run_phase_relation_extraction(docs, ontology, existing_entities)
            self.memory.add_phase_result(phase)
            if phase.output:
                existing_rel = self._phase_outputs.get("relation_extraction", {}).get("relations", [])
                existing_rel.extend([r.model_dump() for r in phase.output.relations])
                self._phase_outputs["relation_extraction"] = {"relations": existing_rel}

        # ── Phase 3.5: Co-occurrence Graph (optional parallel branch) ───
        if enable_co_occurrence:
            cooccur_phase = self._run_phase_co_occurrence(docs, ontology)
            self.memory.add_phase_result(cooccur_phase)
            if cooccur_phase.output and cooccur_phase.output.relations:
                self._phase_outputs["co_occurrence"] = {
                    "relations": [r.model_dump() for r in cooccur_phase.output.relations],
                    "triples": [t.model_dump() for t in cooccur_phase.output.triples],
                }
            existing_rels = self._phase_outputs.get("relation_extraction", {}).get("relations", [])
            if len(existing_rels) < 10:
                if cooccur_phase.output and cooccur_phase.output.relations:
                    existing_rel = self._phase_outputs.get("relation_extraction", {}).get("relations", [])
                    existing_rel.extend([r.model_dump() for r in cooccur_phase.output.relations])
                    self._phase_outputs["relation_extraction"] = {"relations": existing_rel}

        # ── Phase 4: Quality Check ──────────────────────────────────────────
        if "quality_checker" in skill_names:
            entities = self._phase_outputs.get("entity_extraction", {}).get("entities", [])
            relations = self._phase_outputs.get("relation_extraction", {}).get("relations", [])
            phase = self._run_phase_quality_check(ontology, entities, relations, docs)
            self.memory.add_phase_result(phase)

        # ── Phase 5: Triple Construction ────────────────────────────────────
        if "triple_constructor" in skill_names:
            entities = self._phase_outputs.get("entity_extraction", {}).get("entities", [])
            relations = self._phase_outputs.get("relation_extraction", {}).get("relations", [])
            phase = self._run_phase_triple_construction(ontology, entities, relations)
            self.memory.add_phase_result(phase)

        # ── Finalize ────────────────────────────────────────────────────────
        final_result = self._build_final_result()
        nt_output = self.memory.export_final_nt()
        self.memory.set_final_result(final_result, nt_output)

        # Export standalone ontology files
        self.memory.export_ontology()

        # Save document manifest for future change detection
        self.memory.save_document_manifest(self.memory.workflow.documents)

        # Git: commit this build run for version history
        git = GitManager(self.memory.work_dir)
        if git.init():
            wf = self.memory.workflow
            summary = {
                "entities": len(final_result.entities),
                "relations": len(final_result.relations),
                "triples": len(final_result.triples),
            }
            commit_hash = git.commit_build(
                wf.workflow_id if wf else "unknown",
                summary,
            )
            if commit_hash:
                self.log.info(f"Git commit: {commit_hash}")

        self.log.workflow_end(
            entities=len(final_result.entities),
            relations=len(final_result.relations),
            triples=len(final_result.triples),
        )
        self._emit("workflow_complete", {
            "entities": len(final_result.entities),
            "relations": len(final_result.relations),
            "triples": len(final_result.triples),
            "output_file": str(self.memory.work_dir / "output.nt"),
        })

        return final_result

    # ─── Export ─────────────────────────────────────────────────────────────

    def export_nt(self, output_path: Optional[str] = None) -> str:
        """Export the knowledge graph as N-Triples."""
        return self.memory.export_final_nt(output_path)

    def export_json(self, output_path: Optional[str] = None) -> str:
        """Export the knowledge graph as JSON."""
        result = self._build_final_result()

        output = {
            "entities": [
                {
                    "name": e.name,
                    "type": e.type,
                    "attributes": e.attributes,
                    "confidence": e.confidence,
                }
                for e in result.entities
            ],
            "relations": [
                {
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                }
                for r in result.relations
            ],
            "triples": [
                {
                    "subject": {"name": t.subject.name, "type": t.subject.type},
                    "predicate": t.predicate,
                    "object": {"name": t.object.name, "type": t.object.type},
                    "confidence": t.confidence,
                    "evidence": t.evidence,
                }
                for t in result.triples
            ],
        }

        json_str = json.dumps(output, ensure_ascii=False, indent=2)
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json_str)
        else:
            out_path = self.memory.work_dir / "output.json"
            out_path.write_text(json_str)
        return json_str

    def export_jsonl(self, output_path: Optional[str] = None) -> str:
        """Export as JSONL (one triple per line with source text)."""
        result = self._build_final_result()
        lines = []
        for triple in result.triples:
            lines.append(json.dumps({
                "data": triple.evidence or "",
                "triple": [[
                    {triple.subject.name: triple.subject.type},
                    triple.predicate,
                    {triple.object.name: triple.object.type},
                ]],
            }, ensure_ascii=False))

        output = '\n'.join(lines)
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(output)
        else:
            out_path = self.memory.work_dir / "output.jsonl"
            out_path.write_text(output)
        return output
