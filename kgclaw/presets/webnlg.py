"""
WebNLG dataset preset.

159 relation types (DBpedia properties), general-domain English sentences.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.

import csv
from pathlib import Path

from ..models import EntityType, RelationType
from . import DatasetPreset, register


_SCHEMA_PATH = Path.home() / "papers" / "KGClaw" / "webnlg_dataset" / "webnlg_schema.csv"


def _load_webnlg_schema() -> list[RelationType]:
    """Load WebNLG relation types from schema CSV."""
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
    name="webnlg",
    display_name="WebNLG (DBpedia, 159 relations)",
    language="en",
    description=(
        "WebNLG+ 2020 v3.0 — 16 DBpedia categories, general-domain "
        "knowledge graph construction from natural language sentences."
    ),
    entity_types=[
        EntityType(name="Entity", description="Any named entity, concept, location, person, organization, number, or date"),
    ],
    relation_types=_load_webnlg_schema(),
    entity_naming="underscore",
))
