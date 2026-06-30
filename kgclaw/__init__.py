"""
KGClaw — A Harness-based CLI for ontology-driven knowledge graph construction.

KGClaw provides a Claude Code-inspired harness workflow for building
knowledge graphs from unstructured text guided by user-provided ontologies.

Core components:
- Harness: Main orchestrator for KG construction workflows
- Agent: LLM-powered agents with tool use and subagent spawning
- Skill: Self-contained KGC capabilities (NER, RE, QC, etc.)
- Memory: Session-level context and workflow state management
- Tools: Built-in utilities for file I/O, text processing, validation
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from .config import UserConfig
from .harness import Harness
from .models import (
    Document,
    Entity,
    EntityType,
    ExtractionResult,
    HarnessConfig,
    LLMConfig,
    Ontology,
    OntologyChange,
    RefinementPlan,
    Relation,
    RelationType,
    Triple,
    apply_ontology_changes,
)

__version__ = "0.1.0"
__all__ = [
    "Harness",
    "HarnessConfig",
    "LLMConfig",
    "Ontology",
    "EntityType",
    "RelationType",
    "Entity",
    "Relation",
    "Triple",
    "Document",
    "ExtractionResult",
    "OntologyChange",
    "RefinementPlan",
    "UserConfig",
    "apply_ontology_changes",
]
