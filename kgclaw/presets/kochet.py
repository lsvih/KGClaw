"""
KoCHET dataset preset.

Korean Cultural Heritage dataset with 14 relation types.
The original template format "A OriginatedIn B" is simplified to
bare relation names (e.g., "OriginatedIn") with the original template
preserved in the description field for context.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.

from pathlib import Path

from ..models import EntityType, RelationType
from . import DatasetPreset, register


# ─── Entity types (12 coarse NER types) ────────────────────────────────────────

KOCHET_ENTITY_TYPES = [
    EntityType(name="ARTIFACTS",   description="Artifact: building, craft, documents, historical sites, monument, musical instrument, pagoda, painting, relic, weapon, etc."),
    EntityType(name="PERSON",      description="Person: mythical figure, name, noun, position"),
    EntityType(name="TERM",        description="Term: color, mark, shape, technique"),
    EntityType(name="DATE",        description="Date: day, duration, dynasty, geo-age, month, season, year"),
    EntityType(name="POLITICAL_LOCATION", description="Political location: capital city, city, country, county, province"),
    EntityType(name="CIVILIZATION", description="Civilization: building type, clothing, currency, drink, food, language, law, policy, sports, tribe"),
    EntityType(name="MATERIAL",    description="Material: bone, fiber, grass, jewelry, metal, paper, rock, rubber, soil, wood, etc."),
    EntityType(name="LOCATION",    description="Location: space, others"),
    EntityType(name="ANIMAL",      description="Animal: amphibian, bird, fish, insect, mammal, reptile, type"),
    EntityType(name="PLANT",       description="Plant: flower, fruit, grass, tree, type"),
    EntityType(name="GEOGRAPHICAL_LOCATION", description="Geographical: bay, continent, island, mountain, ocean, river"),
    EntityType(name="EVENT",       description="Event: activity, festival, sports, war/revolution"),
]


# ─── Relation types (14 relations) ─────────────────────────────────────────────
#
# The original KoCHET relation names use template format "A OriginatedIn B".
# We strip the "A " prefix and " B" suffix to get clean relation names that
# the LLM can understand and use directly. The original template is preserved
# in the description.

_KOCHET_RELATION_TEMPLATES = [
    ("OriginatedIn", "A OriginatedIn B — The subject originated in / from the object (origin/location provenance)"),
    ("consistsOf",   "A consistsOf B — The subject consists of the object (composition/part-whole)"),
    ("depicts",      "A depicts B — The subject depicts the object (representation)"),
    ("documents",    "A documents B — The subject documents the object (documentation)"),
    ("fallsWithin",  "A fallsWithin B — The subject falls within the object (spatial/temporal containment)"),
    ("hasCarriedOut","A hasCarriedOut B — The subject has carried out the object (performer/agent of action)"),
    ("hasCreated",   "A hasCreated B — The subject has created the object (creation authorship)"),
    ("hasDestroyed", "A hasDestroyed B — The subject has destroyed the object (destruction)"),
    ("hasSection",   "A hasSection B — The subject has the object as a section (partitive)"),
    ("hasTime",      "A hasTime B — The subject has the object as its time (temporal association)"),
    ("isConnectedWith","A isConnectedWith B — The subject is connected with the object (association/connection)"),
    ("isUsedIn",     "A isUsedIn B — The subject is used in the object (instrumental usage)"),
    ("servedAs",     "A servedAs B — The subject served as the object (function/role)"),
    ("wears",        "A wears B — The subject wears the object (attire/adornment)"),
]


def _build_kochet_relations() -> list[RelationType]:
    """Build KoCHET relation types with clean names and preserved descriptions."""
    relations: list[RelationType] = []
    for name, desc in _KOCHET_RELATION_TEMPLATES:
        relations.append(RelationType(
            name=name,
            description=desc,
            domain="Entity",
            range="Entity",
        ))
    return relations


# ─── Registration ──────────────────────────────────────────────────────────────

register(DatasetPreset(
    name="kochet",
    display_name="KoCHET (Korean Cultural Heritage, 14 relations)",
    language="ko",
    description=(
        "KoCHET — Korean Cultural Heritage corpus for relation extraction. "
        "14 relation types extracted from historical documents. "
        "12 coarse entity types with 92 fine-grained subtypes. "
        "Language: Korean (ko)."
    ),
    entity_types=KOCHET_ENTITY_TYPES,
    relation_types=_build_kochet_relations(),
    entity_naming="as_in_text",
))
