"""
Workflow strategy methods for the Harness class.

Provides auto-detection of the best strategy based on data characteristics,
and the fast/code path implementations.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import json
from typing import Optional

from ..models import (
    Document,
    Entity,
    ExtractionResult,
    LLMConfig,
    Ontology,
    Relation,
    Triple,
)


class _HarnessStrategies:
    """Mixin providing workflow strategy selection and fast/code path implementations."""

    @staticmethod
    def _auto_detect_strategy(docs: list[Document]) -> str:
        """Auto-detect the best workflow strategy based on data characteristics.

        Heuristics:
        - Few short docs (< 5 docs, avg < 1000 chars) → fast (simple data)
        - Many tabular docs → code (structured data)
        - Long narrative docs → standard (full pipeline)
        - Default → standard
        """
        if not docs:
            return "standard"

        non_empty = [d for d in docs if d.text.strip()]
        if not non_empty:
            return "standard"

        total_chars = sum(len(d.text) for d in non_empty)
        avg_chars = total_chars / len(non_empty)
        tabular_count = sum(1 for d in non_empty if d.metadata.get("is_tabular"))

        if tabular_count > len(docs) * 0.5:
            return "code"
        if len(docs) <= 5 and avg_chars < 2000:
            return "fast"
        if avg_chars < 500 and len(docs) <= 20:
            return "fast"
        return "standard"

    def _run_fast_path(
        self,
        ontology: Optional[Ontology],
        docs: list[Document],
    ) -> ExtractionResult:
        """Fast path: single-pass combined entity+relation extraction.

        Suitable for simple data where one LLM call per document
        can extract both entities and relations simultaneously.
        """
        from ..agent import Agent, AgentConfig

        self._emit("workflow_start", {
            "skills": ["fast_combined_extraction"],
            "doc_count": len(docs),
            "strategy": "fast",
        })

        ontology_guide = ontology.to_extraction_guide() if ontology else ""
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        for i, doc in enumerate(docs):
            text = doc.text[:8000] if len(doc.text) > 8000 else doc.text
            prompt = f"""请从以下文本中同时抽取所有符合本体定义的实体和实体之间的关系。

## 本体定义
{ontology_guide[:2000]}

## 文本
{text}

## 输出格式
请以 JSON 格式返回：
```json
{{
  "entities": [
    {{"name": "实体名", "type": "实体类型", "mention": "原文提及", "confidence": 0.95}}
  ],
  "relations": [
    {{"subject": "主体实体名", "predicate": "关系类型", "object": "客体实体名", "confidence": 0.85, "evidence": "文本证据"}}
  ]
}}
```

## 要求
1. 抽取所有出现的实体（全面覆盖）
2. 抽取所有存在的关系（可跨句）
3. 只输出 JSON，不要任何解释"""

            agent_cfg = AgentConfig(
                name=f"fast_extract_{i}",
                system_prompt="你是一个知识图谱抽取专家。同时抽取实体和关系。只输出JSON。",
                tools=[],
                max_tool_calls=1,
                model_config=LLMConfig(
                    model=self.llm_config.model,
                    api_key=self.llm_config.api_key,
                    api_base=self.llm_config.api_base,
                    max_tokens=16384,
                ),
            )
            agent = Agent(agent_cfg, self.memory, self.llm_config)
            agent.on_event(lambda et, d: self._emit(et, d))

            result = agent.run_structured(prompt, {
                "type": "object",
                "properties": {
                    "entities": {"type": "array"},
                    "relations": {"type": "array"},
                },
            })

            if result:
                for e_data in result.get("entities", []):
                    all_entities.append(Entity(
                        name=e_data.get("name", ""),
                        type=e_data.get("type", e_data.get("entity_type", "")),
                        description=e_data.get("description", ""),
                        mention=e_data.get("mention"),
                        confidence=e_data.get("confidence", 0.9),
                    ))
                for r_data in result.get("relations", []):
                    all_relations.append(Relation(
                        subject=r_data.get("subject", ""),
                        predicate=r_data.get("predicate", r_data.get("relation", "")),
                        object=r_data.get("object", ""),
                        confidence=r_data.get("confidence", 0.85),
                        evidence=r_data.get("evidence"),
                    ))

        # Deduplicate
        seen_e: dict[tuple[str, str], Entity] = {}
        for e in all_entities:
            key = (e.name, e.type)
            if key not in seen_e or e.confidence > seen_e[key].confidence:
                seen_e[key] = e
        seen_r: dict[tuple[str, str, str], Relation] = {}
        for r in all_relations:
            key = (r.subject, r.predicate, r.object)
            if key not in seen_r or r.confidence > seen_r[key].confidence:
                seen_r[key] = r

        # Build triples
        entity_index = {e.name: e for e in seen_e.values()}
        triples = []
        for r in seen_r.values():
            subj_e = entity_index.get(r.subject, Entity(name=r.subject, type="Entity"))
            obj_e = entity_index.get(r.object, Entity(name=r.object, type="Entity"))
            triples.append(Triple(
                subject=subj_e, predicate=r.predicate,
                object=obj_e, confidence=r.confidence,
                evidence=r.evidence,
            ))

        result = ExtractionResult(
            entities=list(seen_e.values()),
            relations=list(seen_r.values()),
            triples=triples,
        )

        self._emit("workflow_complete", {
            "entities": len(result.entities),
            "relations": len(result.relations),
            "triples": len(result.triples),
            "strategy": "fast",
        })
        # Save document manifest for change detection
        self.memory.save_document_manifest(self.memory.workflow.documents)
        return result

    def _run_code_path(
        self,
        ontology: Optional[Ontology],
        docs: list[Document],
    ) -> ExtractionResult:
        """Code path: agent generates Python extraction code executed in sandbox.

        Suitable for structured/semi-structured data where code-based
        extraction is more reliable than LLM-based extraction.
        """
        from ..agent import Agent, AgentConfig

        self._emit("workflow_start", {
            "skills": ["code_extraction"],
            "doc_count": len(docs),
            "strategy": "code",
        })

        ontology_json = json.dumps({
            "entity_types": [{"name": et.name, "description": et.description}
                           for et in (ontology.entity_types if ontology else [])],
            "relation_types": [{"name": rt.name, "description": rt.description,
                              "domain": rt.domain, "range": rt.range}
                             for rt in (ontology.relation_types if ontology else [])],
        }, ensure_ascii=False)

        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        for i, doc in enumerate(docs[:20]):  # Limit to 20 docs for code path
            data_text = doc.text[:50000]
            file_meta = json.dumps({
                "filename": doc.metadata.get("filename", ""),
                "ext": doc.metadata.get("ext", ""),
                "size": len(doc.text),
                "is_tabular": doc.metadata.get("is_tabular", False),
            })

            code_prompt = f"""为以下数据编写 Python 提取代码，输出 JSON 到 stdout。

## 文件信息
{file_meta}

## 本体定义
{ontology_json[:2000]}

## 数据样本
{data_text[:3000]}

## 要求
编写 Python 代码，从 DATA_TEXT 中提取实体和关系。
使用 ONTOLOGY_JSON 获取本体定义。
输出格式: print(json.dumps({{"entities": [...], "relations": [...]}}))

只输出 Python 代码，不要其他文字。"""
            code_agent_cfg = AgentConfig(
                name=f"code_gen_{i}",
                system_prompt="Generate extraction Python code. Output only code, no explanation.",
                tools=[],
                max_tool_calls=1,
            )
            code_agent = Agent(code_agent_cfg, self.memory, self.llm_config)
            code_agent.on_event(lambda et, d: self._emit(et, d))
            generated_code = code_agent.run(code_prompt, max_iterations=1)

            if generated_code and len(generated_code) > 20:
                # Save generated code to work directory for audit/inspection
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%dT%H%M%S")
                self.memory.save_generated_code(
                    f"extract_doc_{i}_{ts}.py",
                    generated_code,
                )

                from ..tools import execute_tool
                exec_result = execute_tool("extract_with_code", {
                    "code": generated_code,
                    "data_text": data_text,
                    "ontology_json": ontology_json,
                })
                if exec_result.success and exec_result.data:
                    result_data = exec_result.data
                    for e_data in result_data.get("entities", []):
                        all_entities.append(Entity(
                            name=e_data.get("name", ""),
                            type=e_data.get("type", e_data.get("entity_type", "Entity")),
                            confidence=e_data.get("confidence", 0.9),
                        ))
                    for r_data in result_data.get("relations", []):
                        all_relations.append(Relation(
                            subject=r_data.get("subject", ""),
                            predicate=r_data.get("predicate", r_data.get("relation", "")),
                            object=r_data.get("object", ""),
                            confidence=r_data.get("confidence", 0.85),
                        ))
                else:
                    error_msg = exec_result.error if not exec_result.success else "No data returned"
                    self.log.warning(f"Code extraction failed for doc {i}: {error_msg}")

        # Dedup
        seen_e: dict[tuple[str, str], Entity] = {}
        for e in all_entities:
            key = (e.name, e.type)
            if key not in seen_e or e.confidence > seen_e[key].confidence:
                seen_e[key] = e
        seen_r: dict[tuple[str, str, str], Relation] = {}
        for r in all_relations:
            key = (r.subject, r.predicate, r.object)
            if key not in seen_r or r.confidence > seen_r[key].confidence:
                seen_r[key] = r

        result = ExtractionResult(
            entities=list(seen_e.values()),
            relations=list(seen_r.values()),
            triples=[],
        )

        self._emit("workflow_complete", {
            "entities": len(result.entities),
            "relations": len(result.relations),
            "triples": len(result.triples),
            "strategy": "code",
        })
        # Save document manifest for change detection
        self.memory.save_document_manifest(self.memory.workflow.documents)
        return result
