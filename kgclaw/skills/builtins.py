"""
Built-in skills for KGClaw.

Five skills covering the complete KG construction pipeline:
- ontology_analyzer: Parse user ontology into structured schema
- entity_extractor: Extract entities from text
- relation_extractor: Extract relations between entities
- quality_checker: Validate extraction quality
- triple_constructor: Assemble SPO triples
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

from typing import Any, Optional

from ..models import LLMConfig
from . import Skill, SkillMeta, SkillRegistry


@SkillRegistry.register(SkillMeta(
    name="ontology_analyzer",
    description="Analyze user-provided ontology and generate structured extraction guide",
    tags=["ontology", "analysis"],
    requires_ontology=False,
    produces=["ontology_guide"],
))
class OntologyAnalyzerSkill(Skill):
    """Analyzes raw ontology input and generates structured extraction guide."""

    def get_system_prompt(self) -> str:
        from ..prompts.system_prompts import SYSTEM_PROMPT_ONTOLOGY_ANALYZER
        return SYSTEM_PROMPT_ONTOLOGY_ANALYZER

    def get_tool_names(self) -> list[str]:
        return ["read_file", "parse_json", "write_file"]

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ontology_name": {"type": "string"},
                "entity_types": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "parent": {"type": "string"},
                            "attributes": {"type": "object"},
                            "examples": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "relation_types": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "domain": {"type": "string"},
                            "range": {"type": "string"},
                            "inverse": {"type": "string"},
                            "examples": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "extraction_guide": {"type": "string"},
            },
        }

    def post_process(self, raw_output: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        result = super().post_process(raw_output, context)
        if not result.get("extraction_guide"):
            et_names = [et["name"] for et in result.get("entity_types", [])]
            rt_names = [rt["name"] for rt in result.get("relation_types", [])]
            guide = f"Entity Types: {', '.join(et_names)}\nRelation Types: {', '.join(rt_names)}"
            result["extraction_guide"] = guide
        return result


@SkillRegistry.register(SkillMeta(
    name="entity_extractor",
    description="Extract entities from text according to ontology schema",
    tags=["extraction", "ner", "entities"],
    requires_ontology=True,
    produces=["entities"],
))
class EntityExtractorSkill(Skill):
    """Extracts named entities from text following the ontology constraints."""

    def get_system_prompt(self) -> str:
        from ..prompts.system_prompts import SYSTEM_PROMPT_ENTITY_EXTRACTOR_DEFAULT
        return SYSTEM_PROMPT_ENTITY_EXTRACTOR_DEFAULT

    def get_tool_names(self) -> list[str]:
        return []

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                            "mention": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "attributes": {"type": "object"},
                        },
                        "required": ["name", "type", "confidence"],
                    },
                },
            },
        }


@SkillRegistry.register(SkillMeta(
    name="relation_extractor",
    description="Extract relations between entities according to ontology schema",
    tags=["extraction", "re", "relations"],
    requires_ontology=True,
    produces=["relations"],
))
class RelationExtractorSkill(Skill):
    """Extracts relations between given entities following ontology constraints."""

    def get_system_prompt(self) -> str:
        from ..prompts.system_prompts import SYSTEM_PROMPT_RELATION_EXTRACTOR
        return SYSTEM_PROMPT_RELATION_EXTRACTOR

    def get_tool_names(self) -> list[str]:
        return []

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "keywords": {"type": "string"},
                            "description": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "evidence": {"type": "string"},
                        },
                        "required": ["subject", "predicate", "object", "confidence"],
                    },
                },
            },
        }


@SkillRegistry.register(SkillMeta(
    name="quality_checker",
    description="Validate and correct extracted entities and relations against ontology",
    tags=["quality", "validation"],
    requires_ontology=True,
    produces=["corrections", "approved", "rejected"],
))
class QualityCheckerSkill(Skill):
    """Checks extraction quality: entity types, relation directions, duplicates, schema compliance."""

    def get_system_prompt(self) -> str:
        from ..prompts.system_prompts import SYSTEM_PROMPT_QUALITY_CHECKER
        return SYSTEM_PROMPT_QUALITY_CHECKER

    def get_tool_names(self) -> list[str]:
        return ["read_file", "parse_json", "validate_against_ontology", "deduplicate_entities"]

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "corrections": {"type": "array"},
                "approved_entities": {"type": "array"},
                "approved_relations": {"type": "array"},
                "rejected_items": {"type": "array"},
                "overall_quality_score": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }


@SkillRegistry.register(SkillMeta(
    name="triple_constructor",
    description="Construct SPO triples from entities and relations",
    tags=["construction", "triples"],
    requires_ontology=True,
    produces=["triples"],
))
class TripleConstructorSkill(Skill):
    """Assembles (subject, predicate, object) triples from entities and relations."""

    def get_system_prompt(self) -> str:
        return """你是一个 **三元组构造 Agent**。你的任务是将实体和关系组装成标准的 SPO（Subject-Predicate-Object）三元组。

## 规则
1. 每个关系生成标准的三元组
2. 实体必须已经在实体列表中出现过
3. 三元组的 subject 和 object 必须是完整的 Entity 对象（包含 name 和 type）
4. 保留原始文本作为 evidence
5. 去掉没有文本证据的三元组

## 输出格式
```json
{
  "triples": [
    {
      "subject": {"name": "实体名", "type": "实体类型"},
      "predicate": "关系类型",
      "object": {"name": "实体名", "type": "实体类型"},
      "evidence": "文本证据",
      "confidence": 0.0-1.0
    }
  ]
}
```
"""

    def get_tool_names(self) -> list[str]:
        return ["parse_json", "write_file"]

    def get_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "object"},
                            "predicate": {"type": "string"},
                            "object": {"type": "object"},
                            "evidence": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                    },
                },
            },
        }

    def post_process(self, raw_output: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        result = super().post_process(raw_output, context)
        triples_data = result.get("triples", [])
        result["triples"] = triples_data
        result["count"] = len(triples_data)
        return result
