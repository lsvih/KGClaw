"""
Iterative refinement engine for KGClaw.

When a user is unsatisfied with a build result, RefinementEngine analyzes
the last build's output together with user feedback and proposes concrete
changes to the ontology, extraction strategy, and prompts.

Usage:
    engine = RefinementEngine(llm_config, memory)
    plan = engine.analyze(last_result, ontology, docs, user_feedback)
    # Review plan with user, then:
    engine.apply(plan, harness)
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import json
from typing import Any, Optional

from .agent import Agent, AgentConfig
from .logger import get_logger
from .memory import Memory
from .models import (
    ExtractionResult,
    LLMConfig,
    Ontology,
    OntologyChange,
    RefinementPlan,
)

# ─── System prompt for the refinement agent ────────────────────────────────────

SYSTEM_PROMPT_REFINEMENT = """你是一个 **知识图谱构建优化专家 (KG Refinement Specialist)**。

你的任务是分析上一次知识图谱构建的结果，结合用户的反馈意见，提出具体、可落地的优化方案。

## 你的能力
1. **本体优化**: 识别缺失的实体类型/关系类型，建议添加、删除或修改
2. **策略调整**: 根据数据特征和用户反馈，推荐更合适的构建策略
3. **Prompt 优化**: 针对抽取中的具体问题，优化 LLM 抽取 prompt
4. **参数调优**: 建议调整 chunk_size、gleaning、co-occurrence 等参数

## 分析方法
- 对比用户反馈与上次构建结果，找出差距
- 分析实体/关系数量是否合理（太少 → 本体太窄，太多无关 → 本体太宽）
- 检查是否有常见的抽取问题（实体碎片化、关系遗漏、类型错误等）
- 结合文档特征给出针对性建议

## 输出原则
- 只提出有数据支撑的具体建议，不要泛泛而谈
- 本体变更需要给出明确的 name 和 description
- 如果需要更新本体定义，给出完整的自然语言版本
- 如果没有好的建议，诚实地说无需修改"""

TASK_REFINE = """## 当前任务：分析构建结果并生成优化方案

### 用户反馈
{user_feedback}

### 上次构建概况
- 策略: {strategy}
- 实体数: {entity_count}
- 关系数: {relation_count}
- 三元组数: {triple_count}
- 质量评分: {quality_score}

### 当前本体定义
{ontology_guide}

### 上次抽取的实体类型分布
{entity_type_distribution}

### 上次抽取的关系类型分布
{relation_type_distribution}

### 问题实体示例（部分）
{entity_samples}

### 问题关系示例（部分）
{relation_samples}

### 文档概况
- 文档数: {doc_count}
- 总字符数: {total_chars}
- 格式类型: {doc_formats}

请分析上述数据，结合用户反馈，生成一个优化方案。输出严格 JSON 格式。"""


class RefinementEngine:
    """Analyzes build results + user feedback and produces a RefinementPlan."""

    def __init__(self, llm_config: LLMConfig, memory: Memory):
        self.llm_config = llm_config
        self.memory = memory
        self.log = get_logger()

    def analyze(
        self,
        last_result: ExtractionResult,
        ontology: Ontology,
        docs: list,
        user_feedback: str,
        strategy: str = "standard",
        quality_score: float = 0.0,
    ) -> RefinementPlan:
        """Analyze the last build and produce a refinement plan.

        Args:
            last_result: The ExtractionResult from the previous build.
            ontology: The current Ontology (structured or raw).
            docs: The document list.
            user_feedback: User's natural language feedback.
            strategy: The strategy used in the last build.
            quality_score: Quality score from the last build (0-1).

        Returns:
            A RefinementPlan with concrete, applicable changes.
        """
        if not user_feedback.strip():
            return RefinementPlan()

        # Build entity type distribution summary
        type_counts: dict[str, int] = {}
        for e in last_result.entities:
            type_counts[e.type] = type_counts.get(e.type, 0) + 1
        sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
        entity_dist = "\n".join(
            f"  - {t}: {c} 个" for t, c in sorted_types[:20]
        ) if sorted_types else "（无）"

        # Build relation type distribution
        rel_counts: dict[str, int] = {}
        for r in last_result.relations:
            rel_counts[r.predicate] = rel_counts.get(r.predicate, 0) + 1
        sorted_rels = sorted(rel_counts.items(), key=lambda x: -x[1])
        rel_dist = "\n".join(
            f"  - {p}: {c} 个" for p, c in sorted_rels[:20]
        ) if sorted_rels else "（无）"

        # Sample entities (take low-confidence ones for diagnosis)
        low_conf_entities = sorted(
            [e for e in last_result.entities if e.confidence < 0.8],
            key=lambda e: e.confidence,
        )[:10]
        entity_samples = json.dumps(
            [{"name": e.name, "type": e.type, "confidence": e.confidence}
             for e in low_conf_entities],
            ensure_ascii=False, indent=2,
        ) if low_conf_entities else "（所有实体置信度均较高）"

        # Sample relations (low confidence for diagnosis)
        low_conf_rels = sorted(
            [r for r in last_result.relations if r.confidence < 0.8],
            key=lambda r: r.confidence,
        )[:10]
        relation_samples = json.dumps(
            [{"subject": r.subject, "predicate": r.predicate, "object": r.object,
              "confidence": r.confidence}
             for r in low_conf_rels],
            ensure_ascii=False, indent=2,
        ) if low_conf_rels else "（所有关系置信度均较高）"

        # Document summary
        doc_count = len(docs) if docs else 0
        total_chars = sum(len(d.text) for d in docs) if docs else 0
        formats = set()
        for d in (docs or []):
            ext = d.metadata.get("ext", "txt") if hasattr(d, 'metadata') else "txt"
            formats.add(ext)
        doc_formats = ", ".join(sorted(formats)) if formats else "txt"

        # Build the prompt
        prompt = TASK_REFINE.format(
            user_feedback=user_feedback,
            strategy=strategy,
            entity_count=len(last_result.entities),
            relation_count=len(last_result.relations),
            triple_count=len(last_result.triples),
            quality_score=f"{quality_score:.0%}" if quality_score else "N/A",
            ontology_guide=ontology.to_extraction_guide() if ontology else "（无本体定义）",
            entity_type_distribution=entity_dist,
            relation_type_distribution=rel_dist,
            entity_samples=entity_samples,
            relation_samples=relation_samples,
            doc_count=doc_count,
            total_chars=f"{total_chars:,}",
            doc_formats=doc_formats,
        )

        # Call the LLM
        agent_cfg = AgentConfig(
            name="refinement_agent",
            system_prompt=SYSTEM_PROMPT_REFINEMENT,
            tools=[],
            max_tool_calls=1,
        )
        agent = Agent(agent_cfg, self.memory, self.llm_config)

        output_schema = {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "ontology_changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["add", "remove", "modify"]},
                            "target": {"type": "string", "enum": ["entity_type", "relation_type"]},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "domain": {"type": "string"},
                            "range": {"type": "string"},
                            "parent": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["action", "target", "name"],
                    },
                },
                "updated_ontology_raw": {"type": "string"},
                "suggested_strategy": {"type": "string"},
                "extraction_tips": {"type": "string"},
                "prompt_additions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "prompt_removals": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "enable_gleaning": {"type": "boolean"},
                "enable_co_occurrence": {"type": "boolean"},
                "suggested_chunk_size": {"type": "integer"},
            },
        }

        try:
            result = agent.run_structured(prompt, output_schema)
        except Exception as e:
            self.log.warning(f"Refinement analysis failed: {e}")
            return RefinementPlan()

        if not result:
            return RefinementPlan()

        # Parse into RefinementPlan
        return RefinementPlan(
            rationale=result.get("rationale", ""),
            ontology_changes=[
                OntologyChange(**oc)
                for oc in result.get("ontology_changes", [])
                if isinstance(oc, dict)
            ],
            updated_ontology_raw=result.get("updated_ontology_raw", ""),
            suggested_strategy=result.get("suggested_strategy", ""),
            extraction_tips=result.get("extraction_tips", ""),
            prompt_additions=result.get("prompt_additions", []),
            prompt_removals=result.get("prompt_removals", []),
            enable_gleaning=result.get("enable_gleaning"),
            enable_co_occurrence=result.get("enable_co_occurrence"),
            suggested_chunk_size=result.get("suggested_chunk_size", 0),
        )

    def apply(self, plan: RefinementPlan, session: Any) -> dict[str, Any]:
        """Apply a refinement plan to the session, returning a summary of changes.

        Args:
            plan: The RefinementPlan to apply.
            session: The REPL Session object (has .set_ontology(), .strategy, etc.).

        Returns:
            Dict summarizing what was changed.
        """
        changes: dict[str, Any] = {
            "ontology_updated": False,
            "strategy_changed": False,
            "gleaning_toggled": False,
            "co_occurrence_toggled": False,
            "chunk_size_changed": False,
            "tips_added": False,
        }

        # Apply ontology changes
        if plan.ontology_changes:
            self._apply_ontology_changes(plan.ontology_changes, session)
            changes["ontology_updated"] = True
        elif plan.updated_ontology_raw:
            session.set_ontology(plan.updated_ontology_raw)
            changes["ontology_updated"] = True

        # Apply strategy change
        if plan.suggested_strategy and plan.suggested_strategy != session.strategy:
            valid_strategies = {"auto", "fast", "standard", "code"}
            if plan.suggested_strategy in valid_strategies:
                session.strategy = plan.suggested_strategy
                changes["strategy_changed"] = True
                changes["new_strategy"] = plan.suggested_strategy

        # Apply gleaning toggle
        if plan.enable_gleaning is not None:
            if hasattr(session, 'harness'):
                session.harness.config.enable_gleaning = plan.enable_gleaning
            changes["gleaning_toggled"] = True
            changes["gleaning"] = plan.enable_gleaning

        # Apply co-occurrence toggle
        if plan.enable_co_occurrence is not None:
            session.enable_co_occurrence = plan.enable_co_occurrence
            changes["co_occurrence_toggled"] = True
            changes["co_occurrence"] = plan.enable_co_occurrence

        # Apply chunk size
        if plan.suggested_chunk_size > 0:
            if hasattr(session, 'harness'):
                session.harness.config.chunk_size = plan.suggested_chunk_size
            changes["chunk_size_changed"] = True
            changes["chunk_size"] = plan.suggested_chunk_size

        # Store extraction tips for the next build
        if plan.extraction_tips or plan.prompt_additions or plan.prompt_removals:
            session.refinement_tips = {
                "tips": plan.extraction_tips,
                "additions": plan.prompt_additions,
                "removals": plan.prompt_removals,
            }
            changes["tips_added"] = True

        return changes

    def _apply_ontology_changes(
        self,
        changes: list[OntologyChange],
        session: Any,
    ):
        """Apply ontology changes and rebuild the raw definition text."""
        wf = session.harness.memory.workflow
        if not wf or not wf.ontology:
            return

        from .models import apply_ontology_changes
        new_onto = apply_ontology_changes(wf.ontology, changes)
        session.set_ontology(new_onto.raw_definition)


def create_refinement_engine(llm_config: LLMConfig, memory: Memory) -> RefinementEngine:
    """Factory function for RefinementEngine."""
    return RefinementEngine(llm_config, memory)
