"""
NYT-repo dataset presets (NYT, NYT-star, WebNLG, WebNLG-star).

From Dataset-for-NYT-and-WebNLG: JSON format with text + triple_list.
NYT/NYT-star: 24 Freebase relations, 5000 test samples each.
WebNLG/WebNLG-star: 171-216 DBpedia relations, 703 test samples each.
"""
import json
from pathlib import Path

from ..models import EntityType, RelationType
from . import DatasetPreset, register

_BASE = Path("/home/liyanzeng/papers/KGClaw/nyt_dataset/data")

def _load_nyt_relations(ds_name: str) -> list[RelationType]:
    p = _BASE / ds_name / "rel2id.json"
    if not p.exists():
        return []
    with open(p) as f:
        rel2id = json.load(f)
    rels = rel2id[-1]
    result = []
    for full_path in sorted(rels.keys()):
        short = full_path.rsplit("/", 1)[-1]
        result.append(RelationType(
            name=full_path,
            description=f"NYT: {short} ({full_path})",
            domain="Entity", range="Entity",
        ))
    return result

def _load_webnlg_repo_relations(ds_name: str) -> list[RelationType]:
    p = _BASE / ds_name / "rel2id.json"
    if not p.exists():
        return []
    with open(p) as f:
        rel2id = json.load(f)
    rels = rel2id[-1]
    result = []
    for name in sorted(rels.keys()):
        result.append(RelationType(
            name=name,
            description=f"Relation: {name}",
            domain="Entity", range="Entity",
        ))
    return result

# Register NYT
register(DatasetPreset(
    name="nyt",
    display_name="NYT (24 Freebase relations)",
    language="en",
    description="NYT dataset: 24 Freebase relation types, 5000 test sentences.",
    entity_types=[EntityType(name="Entity", description="Any entity")],
    relation_types=_load_nyt_relations("NYT"),
    entity_naming="as_in_text",
))

# Register NYT-star
register(DatasetPreset(
    name="nyt_star",
    display_name="NYT-star (24 Freebase relations)",
    language="en",
    description="NYT-star: 24 Freebase relations, 5000 test sentences (star version).",
    entity_types=[EntityType(name="Entity", description="Any entity")],
    relation_types=_load_nyt_relations("NYT-star"),
    entity_naming="as_in_text",
))

# Register WebNLG (from NYT repo)
register(DatasetPreset(
    name="webnlg_repo",
    display_name="WebNLG-repo (216 DBpedia relations)",
    language="en",
    description="WebNLG from NYT repo: 216 relation types, 703 test sentences.",
    entity_types=[EntityType(name="Entity", description="Any entity")],
    relation_types=_load_webnlg_repo_relations("WebNLG"),
    entity_naming="underscore",
))

# Register WebNLG-star
register(DatasetPreset(
    name="webnlg_star",
    display_name="WebNLG-star (171 DBpedia relations)",
    language="en",
    description="WebNLG-star: 171 relation types, 703 test sentences (star version).",
    entity_types=[EntityType(name="Entity", description="Any entity")],
    relation_types=_load_webnlg_repo_relations("WebNLG-star"),
    entity_naming="underscore",
))
