"""
SREDFM dataset preset.

Multilingual relation extraction dataset (15 languages).
Uses Wikidata property surfaceforms as relation type names.
13 entity types with clear domain/range patterns.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.

import json
from collections import Counter
from pathlib import Path

from ..models import EntityType, RelationType
from . import DatasetPreset, register


_BASE = Path.home() / "papers" / "KGClaw" / "low_resource_datasets" / "SREDFM" / "data"

SREDFM_ENTITY_TYPES = [
    EntityType(name="Concept",  description="Abstract concept (taxon, species, etc.)"),
    EntityType(name="LOC",      description="Geographic location"),
    EntityType(name="DATE",     description="Date or time value"),
    EntityType(name="NUMBER",   description="Numeric value"),
    EntityType(name="PER",      description="Person"),
    EntityType(name="ORG",      description="Organization or institution"),
    EntityType(name="MEDIA",    description="Media work (film, book, album, etc.)"),
    EntityType(name="EVE",      description="Event"),
    EntityType(name="MISC",     description="Miscellaneous entity"),
    EntityType(name="CEL",      description="Celestial body"),
    EntityType(name="TIME",     description="Time duration or interval"),
    EntityType(name="DIS",      description="Disease or medical condition"),
    EntityType(name="UNK",      description="Unknown entity type"),
]


def _load_sredfm_relations(lang: str = "en", max_relations: int = 200) -> list[RelationType]:
    """Extract unique relation surfaceforms from SREDFM JSONL.

    Only the top *max_relations* by frequency are included to keep
    the ontology manageable for LLM extraction.
    """
    file_path = _BASE / f"test.{lang}.jsonl"
    if not file_path.exists():
        return []

    freq: Counter = Counter()

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            for rel in entry.get("relations", []):
                pred = rel.get("predicate", {}).get("surfaceform", "")
                if pred:
                    freq[pred] += 1

    top_relations = [name for name, _ in freq.most_common(max_relations)]
    freq_total = freq.total()

    relations: list[RelationType] = []
    for name in sorted(top_relations):
        count = freq[name]
        relations.append(RelationType(
            name=name,
            description=f"SREDFM: {name} (appears {count} times, {count/max(1,freq_total)*100:.1f}%)",
            domain="Entity",
            range="Entity",
        ))
    return relations


register(DatasetPreset(
    name="sredfm",
    display_name="SREDFM (Multilingual RE, Wikidata properties)",
    language="en",
    description=(
        "SREDFM — Multilingual relation extraction dataset with "
        "Wikidata property surfaceforms across 15 languages. "
        "13 entity types (Concept, LOC, PER, ORG, DATE, NUMBER, MEDIA, "
        "EVE, MISC, CEL, TIME, DIS, UNK)."
    ),
    entity_types=SREDFM_ENTITY_TYPES,
    relation_types=_load_sredfm_relations("en", max_relations=200),
    entity_naming="as_in_text",
))
