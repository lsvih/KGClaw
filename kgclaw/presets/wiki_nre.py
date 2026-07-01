"""
Wiki-NRE dataset preset.

45 relation types (Wikidata properties), Wikipedia-based English sentences.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.

import csv
from pathlib import Path

from ..models import EntityType, RelationType
from . import DatasetPreset, register


_SCHEMA_PATH = Path.home() / "papers" / "KGClaw" / "wiki_nre_dataset" / "wiki-nre_schema.csv"


def _load_wiki_nre_schema() -> list[RelationType]:
    """Load Wiki-NRE relation types from schema CSV."""
    relations: list[RelationType] = []
    if not _SCHEMA_PATH.exists():
        return relations
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                name = row[0].strip()
                desc = row[1].strip()
                if name and desc:
                    relations.append(RelationType(
                        name=name,
                        description=desc,
                        domain="Entity",
                        range="Entity",
                    ))
    return relations


register(DatasetPreset(
    name="wiki_nre",
    display_name="Wiki-NRE (Wikidata, 45 relations)",
    language="en",
    description=(
        "Wiki-NRE — Neural Relation Extraction for Knowledge Base Enrichment, "
        "45 Wikidata relation types from Wikipedia text."
    ),
    entity_types=[
        EntityType(name="Entity", description="Any named entity, concept, location, person, organization, number, or date"),
    ],
    relation_types=_load_wiki_nre_schema(),
    entity_naming="as_in_text",
))
