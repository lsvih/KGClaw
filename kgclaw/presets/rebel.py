"""
REBEL dataset preset.

196 relation types (Wikidata properties), Wikipedia-based English sentences.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.

import csv
from pathlib import Path

from ..models import EntityType, RelationType
from . import DatasetPreset, register


_SCHEMA_PATH = Path.home() / "papers" / "KGClaw" / "rebel_dataset" / "rebel_schema.csv"


def _load_rebel_schema() -> list[RelationType]:
    """Load REBEL relation types from schema CSV."""
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
    name="rebel",
    display_name="REBEL (Wikidata, 196 relations)",
    language="en",
    description=(
        "REBEL — Wikipedia abstract relation extraction, "
        "196 Wikidata property-based relation types."
    ),
    entity_types=[
        EntityType(name="Entity", description="Any named entity, concept, location, person, organization, number, or date"),
    ],
    relation_types=_load_rebel_schema(),
    entity_naming="as_in_text",
))
