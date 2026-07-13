"""
Core data models for KGClaw.

Defines the fundamental types used throughout the harness:
ontologies, entities, relations, triples, documents, and workflow state.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Generic, Literal, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ─── Ontology Building Mode ─────────────────────────────────────────────────

class OntologyMode(str, Enum):
    """Ontology building paradigm from LLM4Onto taxonomy."""
    TEXT_TO_ONTOLOGY = "text-to-ontology"        # T-O: full text → LLM → ontology (KGClaw default)
    RELATION_TO_ONTOLOGY = "relation-to-ontology"  # R-O: identify relations → constrain ontology triples
    HT_RELATION_TO_ONTOLOGY = "ht-relation-to-ontology"  # HT-R-O: entity-ontology pairs + text → head-tail relations
    AFFINITY_CLUSTERING = "affinity-clustering"    # AP clustering for auto entity type discovery
    DENSE_ONTOLOGY = "dense-ontology"             # D-O: max density, optimized for Graph F1
    AUTO = "auto"  # Auto-select best mode based on data characteristics


# ─── IDs ────────────────────────────────────────────────────────────────────

def new_id() -> str:
    return uuid4().hex[:12]


# ─── Ontology Models ─────────────────────────────────────────────────────────

class EntityType(BaseModel):
    """A single entity type defined in the ontology."""
    name: str
    description: str = ""
    parent: Optional[str] = None  # parent entity type name (for hierarchy)
    attributes: dict[str, Any] = Field(default_factory=dict)  # attr_name → attr_type


class RelationType(BaseModel):
    """A single relation type defined in the ontology."""
    name: str
    description: str = ""
    domain: Optional[str] = None  # source entity type
    range: Optional[str] = None   # target entity type
    inverse: Optional[str] = None  # inverse relation name
    attributes: dict[str, Any] = Field(default_factory=dict)


class Ontology(BaseModel):
    """A complete ontology definition for KG construction."""
    name: str = "unnamed"
    description: str = ""
    entity_types: list[EntityType] = Field(default_factory=list)
    relation_types: list[RelationType] = Field(default_factory=list)
    raw_definition: Optional[str] = None  # original user-provided ontology text

    @property
    def entity_type_names(self) -> list[str]:
        return [et.name for et in self.entity_types]

    @property
    def relation_type_names(self) -> list[str]:
        return [rt.name for rt in self.relation_types]

    @property
    def is_structured(self) -> bool:
        """True when LLM analysis has populated entity_types (not just raw_definition)."""
        return len(self.entity_types) > 0

    def get_entity_type(self, name: str) -> Optional[EntityType]:
        for et in self.entity_types:
            if et.name == name:
                return et
        return None

    def get_relation_type(self, name: str) -> Optional[RelationType]:
        for rt in self.relation_types:
            if rt.name == name:
                return rt
        return None

    def to_extraction_guide(self) -> str:
        """Generate a human-readable extraction guide from the ontology."""
        lines = [f"# Ontology: {self.name}", f"{self.description}", ""]
        lines.append("## Entity Types")
        for et in self.entity_types:
            parent_info = f" (subtype of: {et.parent})" if et.parent else ""
            lines.append(f"  - **{et.name}**{parent_info}: {et.description}")
        lines.append("")
        lines.append("## Relation Types")
        for rt in self.relation_types:
            domain_info = f" from `{rt.domain}`" if rt.domain else ""
            range_info = f" to `{rt.range}`" if rt.range else ""
            lines.append(f"  - **{rt.name}**{domain_info}{range_info}: {rt.description}")
        return "\n".join(lines)


# ─── Extraction Models ───────────────────────────────────────────────────────

class Entity(BaseModel):
    """A single extracted entity."""
    name: str
    type: str  # entity type name (must correspond to ontology)
    description: str = ""  # concise description of the entity based on source text
    attributes: dict[str, Any] = Field(default_factory=dict)
    mention: Optional[str] = None  # exact text mention
    confidence: float = 1.0


class Relation(BaseModel):
    """A relation between two entities."""
    subject: str  # entity name
    predicate: str  # relation type name (must correspond to ontology)
    object: str  # entity name
    keywords: str = ""  # comma-separated keywords summarizing the relationship
    description: str = ""  # concise explanation of the relationship
    confidence: float = 1.0
    evidence: Optional[str] = None  # source text snippet


class Triple(BaseModel):
    """A complete (subject, predicate, object) triple for knowledge graph output."""
    subject: Entity
    predicate: str
    object: Entity
    confidence: float = 1.0
    evidence: Optional[str] = None

    def to_nt_line(self) -> str:
        """Convert to N-Triples format line with proper URI encoding."""
        from urllib.parse import quote
        subj_uri = f"<{quote(self.subject.type, safe='')}/{quote(self.subject.name, safe='')}>"
        pred_uri = f"<{quote(self.predicate, safe='')}>"
        obj_uri = f"<{quote(self.object.type, safe='')}/{quote(self.object.name, safe='')}>"
        return f"{subj_uri} {pred_uri} {obj_uri} ."


class ExtractionResult(BaseModel):
    """Results from a single extraction pass over a set of documents."""
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Refinement Models ────────────────────────────────────────────────────────

class OntologyChange(BaseModel):
    """A single change to an ontology entity or relation type."""
    action: str  # "add", "remove", "modify"
    target: str  # "entity_type" or "relation_type"
    name: str
    description: str = ""
    domain: Optional[str] = None   # for relation_type
    range: Optional[str] = None    # for relation_type
    parent: Optional[str] = None   # for entity_type hierarchy
    reason: str = ""               # why this change is proposed


def apply_ontology_changes(
    ontology: Ontology,
    changes: list[OntologyChange],
) -> Ontology:
    """
    Apply a list of OntologyChange objects to an Ontology, returning a new Ontology.

    Handles add/remove/modify for both entity_types and relation_types,
    and rebuilds the raw_definition markdown text.

    This is shared between the CLI refine command and the RefinementEngine.
    """
    et_list = list(ontology.entity_types) if ontology.entity_types else []
    rt_list = list(ontology.relation_types) if ontology.relation_types else []

    for change in changes:
        if change.target == "entity_type":
            if change.action == "add":
                et_list.append(EntityType(
                    name=change.name,
                    description=change.description,
                    parent=change.parent,
                ))
            elif change.action == "remove":
                et_list = [et for et in et_list if et.name != change.name]
            elif change.action == "modify":
                for et in et_list:
                    if et.name == change.name:
                        if change.description:
                            et.description = change.description
                        if change.parent is not None:
                            et.parent = change.parent
                        break
        elif change.target == "relation_type":
            if change.action == "add":
                rt_list.append(RelationType(
                    name=change.name,
                    description=change.description,
                    domain=change.domain,
                    range=change.range,
                ))
            elif change.action == "remove":
                rt_list = [rt for rt in rt_list if rt.name != change.name]
            elif change.action == "modify":
                for rt in rt_list:
                    if rt.name == change.name:
                        if change.description:
                            rt.description = change.description
                        if change.domain is not None:
                            rt.domain = change.domain
                        if change.range is not None:
                            rt.range = change.range
                        break

    # Rebuild raw definition markdown text
    lines = [f"# Ontology: {ontology.name or 'refined'}", ""]
    lines.append("## Entity Types")
    for et in et_list:
        p = f" (subtype of: {et.parent})" if et.parent else ""
        lines.append(f"  - **{et.name}**{p}: {et.description or ''}")
    lines.append("")
    lines.append("## Relation Types")
    for rt in rt_list:
        d = f" from `{rt.domain}`" if rt.domain else ""
        r = f" to `{rt.range}`" if rt.range else ""
        lines.append(f"  - **{rt.name}**{d}{r}: {rt.description or ''}")

    new_raw = "\n".join(lines)
    return Ontology(
        name=ontology.name or "refined",
        description=ontology.description or "",
        entity_types=et_list,
        relation_types=rt_list,
        raw_definition=new_raw,
    )


class RefinementPlan(BaseModel):
    """LLM-generated plan for improving knowledge graph construction.

    Produced by the refinement agent when a user provides feedback
    on a previous build.  Contains concrete, applicable changes.
    """
    rationale: str = ""  # overall analysis of what went wrong / what to improve

    # Ontology changes
    ontology_changes: list[OntologyChange] = Field(default_factory=list)

    # Updated raw ontology definition (natural language)
    updated_ontology_raw: str = ""

    # Strategy recommendation
    suggested_strategy: str = ""  # "auto", "fast", "standard", "code" or "" (no change)

    # Extraction prompts improvements (natural language guidance)
    extraction_tips: str = ""

    # Specific prompt sections to emphasize / add / remove
    prompt_additions: list[str] = Field(default_factory=list)
    prompt_removals: list[str] = Field(default_factory=list)

    # Gleaning and co-occurrence toggles
    enable_gleaning: Optional[bool] = None  # None = no change
    enable_co_occurrence: Optional[bool] = None

    # Chunk size adjustment
    suggested_chunk_size: int = 0  # 0 = no change

    # Whether any changes were proposed at all
    @property
    def has_changes(self) -> bool:
        return bool(
            self.ontology_changes
            or self.updated_ontology_raw
            or self.suggested_strategy
            or self.extraction_tips
            or self.prompt_additions
            or self.prompt_removals
            or self.enable_gleaning is not None
            or self.enable_co_occurrence is not None
            or self.suggested_chunk_size > 0
        )


class Document(BaseModel):
    """A single document/sentence to process."""
    id: str = Field(default_factory=new_id)
    text: str
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A chunk of text with associated extraction results."""
    id: str = Field(default_factory=new_id)
    text: str
    source_doc_id: str = ""
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Workflow State Models ───────────────────────────────────────────────────

class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhaseResult(BaseModel):
    """Result from a workflow phase."""
    phase_name: str
    status: PhaseStatus = PhaseStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output: Optional[ExtractionResult] = None
    error_message: Optional[str] = None
    agent_log: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """Complete state of a KG construction workflow."""
    workflow_id: str = Field(default_factory=new_id)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    ontology: Optional[Ontology] = None
    documents: list[Document] = Field(default_factory=list)
    phases: list[PhaseResult] = Field(default_factory=list)
    final_result: Optional[ExtractionResult] = None
    output_nt: Optional[str] = None  # Final N-Triples output
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def all_entities(self) -> list[Entity]:
        entities: list[Entity] = []
        for phase in self.phases:
            if phase.output:
                entities.extend(phase.output.entities)
        return entities

    @property
    def all_triples(self) -> list[Triple]:
        triples: list[Triple] = []
        for phase in self.phases:
            if phase.output:
                triples.extend(phase.output.triples)
        if self.final_result:
            triples.extend(self.final_result.triples)
        return triples


# ─── Agent/Message Models ────────────────────────────────────────────────────

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in an agent conversation."""
    role: Role
    content: str
    name: Optional[str] = None  # tool name for tool messages
    tool_call_id: Optional[str] = None


class ToolDefinition(BaseModel):
    """Definition of a tool that an agent can use."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


T = TypeVar("T")


class ToolResult(BaseModel, Generic[T]):
    """Result from a tool execution."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None


# ─── Configuration Models ────────────────────────────────────────────────────

class LLMConfig(BaseModel):
    """LLM provider configuration.

    All providers (openai, deepseek, ollama, custom) are treated as
    OpenAI-compatible APIs. The `provider` field is informational/display only.
    """
    provider: str = Field(default="openai", description="Provider name (display only; all use OpenAI-compatible API)")
    model: str = "deepseek-v4-flash"
    api_key: str = ""
    api_base: str = "https://api.deepseek.com/v1"
    temperature: float = 0.3
    max_tokens: int = 16384
    extra_params: dict[str, Any] = Field(default_factory=dict)


class HarnessConfig(BaseModel):
    """Top-level harness configuration."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    max_concurrent_agents: int = Field(default=8, ge=1, le=64, description="Max parallel agents (1-64)")
    max_retries_per_phase: int = Field(default=3, description="Max retries per phase (used as LLM API max_retries)")
    chunk_size: int = Field(default=4000, ge=500, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=300, ge=0, description="Overlap between chunks in characters")
    output_format: str = "nt"  # nt, jsonl, json
    verbose: bool = False
    work_dir: str = ".kgclaw"
    skills_dir: Optional[str] = None  # custom skills directory
    enable_gleaning: bool = True  # second-pass extraction to catch missed entities
    # ── Advanced tuning knobs ────────────────────────────────────────────
    docs_per_relation_group: int = Field(default=8, ge=1, description="Documents per relation extraction group")
    chars_per_doc_relation: int = Field(default=4000, ge=500, description="Chars per doc for relation extraction")
    max_entities_in_qc: int = Field(default=500, ge=50, description="Max entities sent to quality checker")
    max_relations_in_qc: int = Field(default=500, ge=50, description="Max relations sent to quality checker")
    max_chunks: int = Field(default=200, ge=10, description="Max chunks before forced merge")
    ontology_mode: str = Field(default="auto", description="Ontology building mode: auto, text-to-ontology, relation-to-ontology, ht-relation-to-ontology, affinity-clustering")

    @field_validator("output_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v not in ("nt", "jsonl", "json"):
            raise ValueError(f"output_format must be one of: nt, jsonl, json")
        return v
