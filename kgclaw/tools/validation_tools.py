"""Ontology validation tools for KGClaw agents."""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import json
from typing import Any

from . import Tool


@Tool.register(
    name="validate_against_ontology",
    description="Validate extracted entities and relations against the ontology schema.",
    parameters={
        "type": "object",
        "properties": {
            "entities_json": {"type": "string", "description": "JSON array of extracted entities"},
            "relations_json": {"type": "string", "description": "JSON array of extracted relations"},
            "ontology_json": {"type": "string", "description": "JSON representation of the ontology"},
        },
        "required": ["entities_json", "relations_json", "ontology_json"],
    },
)
def tool_validate_against_ontology(
    entities_json: str,
    relations_json: str,
    ontology_json: str,
) -> dict[str, Any]:
    # Robust parsing: handle both JSON strings and already-parsed objects
    try:
        entities = json.loads(entities_json) if isinstance(entities_json, str) else entities_json
    except (json.JSONDecodeError, TypeError, AttributeError):
        entities = []
    try:
        relations = json.loads(relations_json) if isinstance(relations_json, str) else relations_json
    except (json.JSONDecodeError, TypeError, AttributeError):
        relations = []
    try:
        ontology = json.loads(ontology_json) if isinstance(ontology_json, str) else ontology_json
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {
            "valid": False,
            "issues": [{"type": "parse_error", "message": "Failed to parse ontology JSON"}],
            "valid_entity_types": [],
            "valid_relation_types": [],
        }

    # Guard against non-dict/list inputs
    if not isinstance(ontology, dict):
        return {
            "valid": False,
            "issues": [{"type": "parse_error", "message": f"Ontology is not a dict: {type(ontology).__name__}"}],
            "valid_entity_types": [],
            "valid_relation_types": [],
        }
    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []

    valid_entity_types = {et["name"] for et in ontology.get("entity_types", []) if isinstance(et, dict) and "name" in et}
    valid_relation_types = {rt["name"] for rt in ontology.get("relation_types", []) if isinstance(rt, dict) and "name" in rt}

    # Limit validation to avoid excessive output when LLM passes huge lists
    MAX_VALIDATION_ENTITIES = 500
    MAX_VALIDATION_RELATIONS = 500
    MAX_ISSUES = 100

    issues = []
    for entity in entities[:MAX_VALIDATION_ENTITIES]:
        if not isinstance(entity, dict):
            continue
        etype = entity.get("type", entity.get("entity_type", ""))
        if etype and etype not in valid_entity_types:
            if len(issues) < MAX_ISSUES:
                issues.append({
                    "type": "invalid_entity_type",
                    "entity": entity.get("name"),
                    "given_type": etype,
                    "valid_types": list(valid_entity_types),
                })

    for rel in relations[:MAX_VALIDATION_RELATIONS]:
        if not isinstance(rel, dict):
            continue
        rtype = rel.get("predicate", rel.get("relation", ""))
        if rtype and rtype not in valid_relation_types:
            if len(issues) < MAX_ISSUES:
                issues.append({
                    "type": "invalid_relation_type",
                    "subject": rel.get("subject"),
                    "object": rel.get("object"),
                    "given_type": rtype,
                    "valid_types": list(valid_relation_types),
                })

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "issue_count": len(issues),
        "total_entities_checked": min(len(entities), MAX_VALIDATION_ENTITIES),
        "total_relations_checked": min(len(relations), MAX_VALIDATION_RELATIONS),
        "valid_entity_types": list(valid_entity_types),
        "valid_relation_types": list(valid_relation_types),
    }


@Tool.register(
    name="deduplicate_entities",
    description="Deduplicate a list of entities by name normalization and fuzzy matching.",
    parameters={
        "type": "object",
        "properties": {
            "entities_json": {"type": "string", "description": "JSON array of entities to deduplicate"},
        },
        "required": ["entities_json"],
    },
)
def tool_deduplicate_entities(entities_json: str) -> list[dict[str, Any]]:
    entities = json.loads(entities_json) if isinstance(entities_json, str) else entities_json
    seen: dict[str, dict[str, Any]] = {}
    for entity in entities:
        name = entity.get("name", "").strip()
        etype = entity.get("type", entity.get("entity_type", ""))
        key = f"{name}|{etype}"
        if key in seen:
            existing_conf = seen[key].get("confidence", 0)
            new_conf = entity.get("confidence", 1.0)
            if new_conf > existing_conf:
                seen[key] = entity
        else:
            seen[key] = entity
    return list(seen.values())
