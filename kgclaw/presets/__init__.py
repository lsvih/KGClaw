"""
Preset ontologies for common knowledge graph evaluation datasets.

Provides pre-built :class:`Ontology` objects for each dataset so that
KGClaw can skip the LLM-based ontology analysis phase and directly use
the dataset's own label system as structured input.

Usage::

    from kgclaw.presets import get_preset, build_ontology

    preset = get_preset("webnlg")
    ontology = build_ontology("webnlg")
    harness.set_ontology_structured(ontology)
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..models import EntityType, Ontology, RelationType


# ─── Preset Dataclass ──────────────────────────────────────────────────────────

@dataclass
class DatasetPreset:
    """Metadata and schema for a single evaluation dataset."""

    name: str                              # internal key
    display_name: str                      # human-readable name
    language: str = "en"                   # ISO 639-1 language code
    description: str = ""                  # dataset description
    entity_types: list[EntityType] = field(default_factory=list)
    relation_types: list[RelationType] = field(default_factory=list)
    entity_naming: str = "as_in_text"      # "as_in_text" | "underscore"


# ─── Preset Registry ───────────────────────────────────────────────────────────

_registry: dict[str, DatasetPreset] = {}


def register(preset: DatasetPreset) -> DatasetPreset:
    """Register a dataset preset."""
    _registry[preset.name] = preset
    return preset


def get_preset(name: str) -> Optional[DatasetPreset]:
    """Get a registered preset by name."""
    return _registry.get(name)


def list_presets() -> list[str]:
    """Return all registered preset names."""
    return sorted(_registry.keys())


def build_ontology(name: str) -> Optional[Ontology]:
    """Build a structured :class:`Ontology` from a registered preset.

    The returned Ontology has ``is_structured == True`` so that
    ``Harness.set_ontology_structured()`` can bypass Phase 1 analysis.
    """
    preset = _registry.get(name)
    if preset is None:
        return None

    return Ontology(
        name=preset.name,
        description=preset.description,
        entity_types=list(preset.entity_types),
        relation_types=list(preset.relation_types),
    )


# ─── Auto-load sub-modules on import ───────────────────────────────────────────

# Each sub-module registers itself via register() at import time.
_loader_dir = Path(__file__).parent
for _f in sorted(_loader_dir.glob("*.py")):
    _stem = _f.stem
    if _stem.startswith("_") or _stem == "__init__":
        continue
    try:
        __import__(f"kgclaw.presets.{_stem}", fromlist=["_"])
    except Exception:
        pass  # allow partial import – missing data files are not fatal
