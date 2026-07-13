"""
Phase implementations for the Harness class.

Contains all _run_phase_* methods and the _build_final_result aggregator.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import json
import re as _re
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..agent import Agent, AgentConfig
from ..models import (
    Chunk,
    Document,
    Entity,
    EntityType,
    ExtractionResult,
    LLMConfig,
    Ontology,
    PhaseResult,
    PhaseStatus,
    Relation,
    RelationType,
    Triple,
)
from ..prompts.system_prompts import (
    SYSTEM_PROMPT_ONTOLOGY_ANALYZER,
    build_entity_extraction_prompt,
    build_ontology_analysis_prompt,
    build_quality_check_prompt,
    build_relation_extraction_prompt,
)
from ..skills import get_skill


class _HarnessPhases:
    """Mixin providing all phase implementations and final result building."""

    # ── Phase: Ontology Analysis ───────────────────────────────────────────

    def _run_phase_ontology_analysis(self, ontology_raw: str) -> PhaseResult:
        phase = PhaseResult(phase_name="ontology_analysis", status=PhaseStatus.RUNNING)
        phase.started_at = datetime.now()

        self._emit("phase_start", {"phase": "ontology_analysis"})

        try:
            structured = self._analyze_ontology_raw(ontology_raw)

            if structured:
                self.memory.workflow.ontology = structured
                self.memory.save_workflow()

                extraction_guide = structured.to_extraction_guide()
                phase.output = ExtractionResult(
                    metadata={
                        "structured_ontology": {
                            "entity_types": [et.model_dump() for et in structured.entity_types],
                            "relation_types": [rt.model_dump() for rt in structured.relation_types],
                        },
                        "extraction_guide": extraction_guide,
                    },
                )
                phase.status = PhaseStatus.COMPLETED
                self._emit("phase_complete", {
                    "phase": "ontology_analysis",
                    "entity_types": len(structured.entity_types),
                    "relation_types": len(structured.relation_types),
                })
            else:
                phase.status = PhaseStatus.FAILED
                phase.error_message = "LLM returned no structured output"
                self._emit("phase_failed", {"phase": "ontology_analysis", "error": phase.error_message})

        except Exception as e:
            phase.status = PhaseStatus.FAILED
            phase.error_message = str(e)
            self._emit("phase_failed", {"phase": "ontology_analysis", "error": str(e)})

        phase.completed_at = datetime.now()
        return phase

    # ── Phase: Entity Extraction ───────────────────────────────────────────

    def _run_phase_entity_extraction(
        self,
        documents: list[Document],
        ontology: Optional[Ontology],
    ) -> PhaseResult:
        phase = PhaseResult(phase_name="entity_extraction", status=PhaseStatus.RUNNING)
        phase.started_at = datetime.now()

        self.log.phase_start("entity_extraction")
        self._emit("phase_start", {"phase": "entity_extraction", "documents": len(documents)})

        try:
            skill = get_skill("entity_extractor", self.llm_config)
            ontology_guide = ontology.to_extraction_guide() if ontology else ""

            # Analyze document characteristics, let Agent decide extraction strategy
            doc_samples = []
            for i, d in enumerate(documents[:5]):
                sample = d.text[:800] if len(d.text) > 800 else d.text
                meta = d.metadata
                doc_samples.append({
                    "index": i, "ext": meta.get("ext", ""), "size": len(d.text),
                    "filename": meta.get("filename", ""), "sample": sample,
                })
            total_docs = len(documents)
            total_size = sum(len(d.text) for d in documents)

            strategy_prompt = f"""分析以下文档集合，决定实体抽取策略。

## 文档概况
总数: {total_docs}, 总大小: {total_size:,} 字符
样本: {json.dumps(doc_samples, ensure_ascii=False, indent=2)[:3000]}

## 本体
{ontology_guide[:1000]}

## 你要决定
1. 这些文档适合 LLM 直接抽取，还是适合编写 Python 代码提取？
2. 如果 LLM 抽取: 应该发送全文、截断前 N 字符、还是跳过某些文档？
3. 如果代码提取: 用 extract_with_code 工具生成代码

返回 JSON:
```json
{{"method": "llm"|"code", "reason": "...", "text_strategy": "full"|"truncate", "truncate_chars": 8000, "skip_docs": []}}
```"""
            strategy_agent_cfg = AgentConfig(
                name="extraction_strategist", system_prompt="分析文档并决定抽取策略。只返回 JSON。",
                tools=[], max_tool_calls=1,
            )
            strategy_agent = Agent(strategy_agent_cfg, self.memory, self.llm_config)
            strategy_agent.on_event(lambda et, d: self._emit(et, d))
            strategy_result = strategy_agent.run_structured(strategy_prompt, {
                "type": "object", "properties": {
                    "method": {"type": "string"},
                    "reason": {"type": "string"},
                    "text_strategy": {"type": "string"},
                    "truncate_chars": {"type": "integer"},
                    "skip_docs": {"type": "array"},
                },
            })
            method = (strategy_result or {}).get("method", "llm")
            text_strategy = (strategy_result or {}).get("text_strategy", "full")
            truncate_chars = (strategy_result or {}).get("truncate_chars", 8000)
            self.log.info(f"Extraction strategy: method={method}, text={text_strategy}, truncate={truncate_chars}")

            # If Phase 1.5 (agent_code_extraction) already produced enough results, skip LLM extraction
            existing_entities = self._phase_outputs.get("entity_extraction", {}).get("entities", [])
            if method == "code" and len(existing_entities) > 10:
                self.log.info(f"Skipping LLM entity extraction: {len(existing_entities)} entities from code extraction")
                phase.output = ExtractionResult()
                phase.status = PhaseStatus.COMPLETED
                return phase

            doc_texts = []
            for i, d in enumerate(documents):
                text = d.text
                if text_strategy == "truncate" and len(text) > truncate_chars:
                    text = text[:truncate_chars] + "\n...[truncated]..."
                doc_texts.append(f"[Doc {i}] {text}")
            all_text = "\n\n".join(doc_texts)

            chunks = self._chunk_text(all_text)

            self._emit("chunk_progress", {
                "phase": "entity_extraction",
                "total_chunks": len(chunks),
                "current": 0,
                "status": "starting",
            })

            all_entities: list[Entity] = []
            entity_lock = threading.Lock()
            chunk_start_time = time.time()
            completed_count = [0]

            emit_lock = threading.Lock()

            def _emit_safe(event_type: str, data: dict):
                with emit_lock:
                    self._emit(event_type, data)

            def _process_chunk(i: int, chunk_text: str):
                import threading as _threading
                _tid = _threading.current_thread().name
                agent_config = AgentConfig(
                    name=f"entity_extractor_{i}",
                    system_prompt=skill.get_system_prompt(),
                    tools=[],
                    max_tool_calls=1,
                    model_config=LLMConfig(
                        model=self.llm_config.model,
                        api_key=self.llm_config.api_key,
                        api_base=self.llm_config.api_base,
                        max_tokens=self.llm_config.max_tokens,
                    ),
                )
                agent = Agent(agent_config, self.memory, self.llm_config)
                agent.on_event(lambda et, d: _emit_safe(et, d))

                prompt = build_entity_extraction_prompt(
                    ontology_guide=ontology_guide,
                    texts=chunk_text,
                    existing_entities="（并行处理中，暂无已确认实体）",
                )

                call_start = time.time()
                self.log.agent_call(f"entity_extractor_{i}", prompt_size=len(prompt), chunk_index=i, thread=_tid)
                _emit_safe("agent_call_start", {
                    "agent": f"entity_extractor_{i}",
                    "chunk_index": i,
                    "prompt_size": len(prompt),
                    "chunk_size": len(chunk_text),
                    "thread": _tid,
                })

                try:
                    result = agent.run_structured(prompt, skill.get_output_schema())
                    if result is None and i < 3:
                        self.log.warning(f"Chunk {i}: structured parse failed, retrying with raw prompt")
                        simple_prompt = prompt + "\n\nPlease output ONLY valid JSON. No markdown, no explanation."
                        raw_response = agent.run(simple_prompt, max_iterations=1)
                        self.log.debug(f"Chunk {i} raw response ({len(raw_response)} chars): {raw_response[:500]}")
                        json_match = _re.search(r'\{[\s\S]*?"entities"[\s\S]*?\}', raw_response)
                        if json_match:
                            try:
                                result = json.loads(json_match.group(0))
                            except Exception:
                                pass
                    if i < 3:
                        ent_count = len(result.get("entities", [])) if result else 0
                        self.log.debug(f"Chunk {i}: prompt={len(prompt)} chars, result={ent_count} entities")
                except Exception as e:
                    _emit_safe("agent_call_end", {
                        "agent": f"entity_extractor_{i}",
                        "chunk_index": i,
                        "duration": time.time() - call_start,
                        "has_result": False,
                        "entities_found": 0,
                    })
                    return []

                duration = time.time() - call_start
                entities_found = len(result.get("entities", [])) if result else 0
                if not self._stop_event.is_set():
                    self.log.agent_result(f"entity_extractor_{i}", duration=duration, entities=entities_found)
                _emit_safe("agent_call_end", {
                    "agent": f"entity_extractor_{i}",
                    "chunk_index": i,
                    "duration": duration,
                    "has_result": result is not None,
                    "entities_found": entities_found,
                })

                entities = []
                if result:
                    for e_data in result.get("entities", []):
                        entities.append(Entity(
                            name=e_data.get("name", ""),
                            type=e_data.get("type", e_data.get("entity_type", "")),
                            description=e_data.get("description", ""),
                            mention=e_data.get("mention"),
                            confidence=e_data.get("confidence", 1.0),
                        ))

                if self._stop_event.is_set():
                    return entities

                with entity_lock:
                    if not self._stop_event.is_set():
                        all_entities.extend(entities)
                        completed_count[0] += 1
                        _emit_safe("chunk_progress", {
                            "phase": "entity_extraction",
                            "total_chunks": len(chunks),
                            "current": completed_count[0],
                            "status": "done",
                            "new_entities": len(entities),
                            "total_entities": len(all_entities),
                            "elapsed": time.time() - chunk_start_time,
                        })

                return entities

            # Parallel processing of all chunks — use full configured concurrency
            max_workers = max(1, min(self.config.max_concurrent_agents, len(chunks)))
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                # ── Phase 1: process all chunks ──
                try:
                    futures = {
                        executor.submit(_process_chunk, i, chunk_text): i
                        for i, chunk_text in enumerate(chunks)
                    }
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            i = futures[future]
                            _emit_safe("phase_failed", {
                                "phase": f"entity_extraction_chunk_{i}",
                                "error": str(e),
                            })
                except KeyboardInterrupt:
                    self._stop_event.set()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

                # ── Deduplicate entities ──
                seen: dict[tuple[str, str], Entity] = {}
                for e in all_entities:
                    key = (e.name, e.type)
                    if key not in seen or e.confidence > seen[key].confidence:
                        seen[key] = e
                unique_entities = list(seen.values())

                # Fuzzy dedup
                before_fuzzy = len(unique_entities)
                unique_entities = self._fuzzy_dedup_entities(unique_entities)
                fuzzy_merged = before_fuzzy - len(unique_entities)
                if fuzzy_merged > 0:
                    self.log.info(f"Fuzzy dedup merged {fuzzy_merged} entities ({before_fuzzy} -> {len(unique_entities)})")

                # ── Gleaning pass: second extraction to catch missed entities ──
                gleaned_count = 0
                if getattr(self.config, 'enable_gleaning', True) and unique_entities:
                    gleaning_future = executor.submit(
                        self._glean_entities,
                        ontology_guide, all_text, unique_entities, skill,
                    )
                    try:
                        gleaned = gleaning_future.result(timeout=120)
                    except Exception:
                        gleaned = None

                    if gleaned:
                        # Merge gleaned entities, preferring longer descriptions
                        existing_names = {(e.name, e.type) for e in unique_entities}
                        for ge in gleaned:
                            key = (ge.name, ge.type)
                            if key in existing_names:
                                # Update description if gleaned version is longer/more detailed
                                for existing in unique_entities:
                                    if (existing.name, existing.type) == key:
                                        if len(ge.description) > len(existing.description):
                                            existing.description = ge.description
                                        if ge.confidence > existing.confidence:
                                            existing.confidence = ge.confidence
                                        break
                            else:
                                unique_entities.append(ge)
                                gleaned_count += 1
                        self.log.info(f"Gleaning pass: added {gleaned_count} missed entities")
            finally:
                # Shutdown executor AFTER all work completes (including gleaning).
                try:
                    executor.shutdown(wait=True)
                except Exception:
                    pass

            phase.output = ExtractionResult(
                entities=unique_entities,
                metadata={"chunks_processed": len(chunks), "gleaned": gleaned_count},
            )
            phase.status = PhaseStatus.COMPLETED
            self._emit("phase_complete", {
                "phase": "entity_extraction",
                "entities": len(unique_entities),
                "chunks": len(chunks),
                "gleaned": gleaned_count,
            })

        except Exception as e:
            phase.status = PhaseStatus.FAILED
            phase.error_message = str(e)
            self._emit("phase_failed", {"phase": "entity_extraction", "error": str(e)})

        phase.completed_at = datetime.now()
        return phase

    # ── Gleaning: Second-pass entity extraction ─────────────────────────────

    def _glean_entities(
        self,
        ontology_guide: str,
        all_text: str,
        existing_entities: list[Entity],
        skill,
    ) -> list[Entity]:
        """Second-pass extraction to catch missed or malformed entities.

        Feeds the first-pass results back to the LLM and asks it to find
        entities that were missed, truncated, or incorrectly formatted.
        """
        from ..prompts.system_prompts import TASK_GLEAN_ENTITIES

        # Build a summary of existing entities (compact to save context)
        existing_summary = json.dumps(
            [{"name": e.name, "type": e.type} for e in existing_entities[:300]],
            ensure_ascii=False,
        )

        prompt = TASK_GLEAN_ENTITIES.format(
            ontology_guide=ontology_guide[:2000],
            texts=all_text[:12000],  # Limit text to avoid context overflow
            extracted_entities_summary=existing_summary,
        )

        agent_cfg = AgentConfig(
            name="entity_gleaner",
            system_prompt=skill.get_system_prompt(),
            tools=[],
            max_tool_calls=1,
            model_config=LLMConfig(
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                api_base=self.llm_config.api_base,
                max_tokens=max(4096, self.llm_config.max_tokens // 2),  # gleaning uses half the main budget
            ),
        )
        agent = Agent(agent_cfg, self.memory, self.llm_config)
        agent.on_event(lambda et, d: self._emit(et, d))

        self._emit("agent_call_start", {
            "agent": "entity_gleaner",
            "chunk_index": 0,
            "prompt_size": len(prompt),
            "entity_count": len(existing_entities),
        })

        try:
            result = agent.run_structured(prompt, skill.get_output_schema())
        except Exception:
            result = None

        entities = []
        if result:
            for e_data in result.get("entities", []):
                entities.append(Entity(
                    name=e_data.get("name", ""),
                    type=e_data.get("type", e_data.get("entity_type", "")),
                    description=e_data.get("description", ""),
                    mention=e_data.get("mention"),
                    confidence=e_data.get("confidence", 0.7),  # Gleaned entities start lower
                ))
        return entities

    # ── Phase: Relation Extraction ─────────────────────────────────────────

    def _run_phase_relation_extraction(
        self,
        documents: list[Document],
        ontology: Optional[Ontology],
        existing_entities: list[dict[str, Any]],
    ) -> PhaseResult:
        phase = PhaseResult(phase_name="relation_extraction", status=PhaseStatus.RUNNING)
        phase.started_at = datetime.now()

        self._emit("phase_start", {"phase": "relation_extraction"})

        try:
            skill = get_skill("relation_extractor", self.llm_config)
            ontology_guide = ontology.to_extraction_guide() if ontology else ""

            entity_names_set: set[str] = set()
            for e in existing_entities:
                name = (e.get("name") or "").strip()
                if name:
                    entity_names_set.add(name)

            DOCS_PER_GROUP = self.config.docs_per_relation_group
            CHARS_PER_DOC = self.config.chars_per_doc_relation
            doc_groups: list[list[tuple[int, str]]] = []
            current_group: list[tuple[int, str]] = []
            for i, d in enumerate(documents):
                text = d.text[:CHARS_PER_DOC] if len(d.text) > CHARS_PER_DOC else d.text
                current_group.append((i, text))
                if len(current_group) >= DOCS_PER_GROUP:
                    doc_groups.append(current_group)
                    current_group = []
            if current_group:
                doc_groups.append(current_group)

            group_texts: list[str] = []
            group_entities: list[list[dict]] = []
            for group in doc_groups:
                group_text = "\n\n".join(f"[Doc {i}] {text}" for i, text in group)
                group_texts.append(group_text)
                group_text_lower = group_text.lower()
                group_ents = []
                for e in existing_entities:
                    name = (e.get("name") or "").strip()
                    if name and len(name) >= 2 and name.lower() in group_text_lower:
                        group_ents.append(e)
                if len(group_ents) > 100:
                    group_ents = group_ents[:100]
                group_entities.append(group_ents)

            # Dynamic relation filtering: for large schemas, only include
            # relations whose keywords appear in the group text or whose
            # domain/range matches entity types in the group.
            import re as _re
            MAX_RELS_PER_GROUP = 80
            all_relation_types = list(ontology.relation_types) if ontology and ontology.relation_types else []
            use_filtered_relations = False
            filtered_ontology_guides = []
            if len(all_relation_types) > MAX_RELS_PER_GROUP:
                use_filtered_relations = True
                # Build filtered ontology per group
                group_entity_type_names = set()
                for gents in group_entities:
                    for e in gents:
                        t = (e.get("type") or "").strip()
                        if t:
                            group_entity_type_names.add(t)
                for gidx, (gtext, gents) in enumerate(zip(group_texts, group_entities)):
                    gtext_lower = gtext.lower()
                    matched_rels = []
                    unmatched_rels = []
                    for rt in all_relation_types:
                        rel_keywords = rt.name.lower().replace("_", " ").split()
                        if any(len(kw) > 3 and kw in gtext_lower for kw in rel_keywords):
                            matched_rels.append(rt)
                        elif rt.domain and rt.domain in group_entity_type_names:
                            matched_rels.append(rt)
                        elif rt.range and rt.range in group_entity_type_names:
                            matched_rels.append(rt)
                        else:
                            unmatched_rels.append(rt)
                    rels_for_group = matched_rels[:MAX_RELS_PER_GROUP]
                    remaining = MAX_RELS_PER_GROUP - len(rels_for_group)
                    if remaining > 0 and unmatched_rels:
                        rels_for_group.extend(unmatched_rels[:remaining])
                    if rels_for_group:
                        from ..models import Ontology as _Ontology
                        filtered_onto = _Ontology(
                            name=ontology.name if ontology else "",
                            entity_types=ontology.entity_types if ontology else [],
                            relation_types=rels_for_group,
                        )
                        filtered_ontology_guides.append(filtered_onto.to_extraction_guide())
                    else:
                        filtered_ontology_guides.append(ontology_guide)

            self._emit("chunk_progress", {
                "phase": "relation_extraction",
                "total_chunks": len(doc_groups),
                "current": 0,
                "status": "starting",
            })

            all_relations: list[Relation] = []
            relation_lock = threading.Lock()
            chunk_start_time = time.time()
            completed_count = [0]
            emit_lock = threading.Lock()

            def _emit_safe(event_type: str, data: dict):
                with emit_lock:
                    self._emit(event_type, data)

            def _process_relation_group(g_idx: int, group_text: str, grp_entities: list[dict]):
                import threading as _threading
                _tid = _threading.current_thread().name
                agent_config = AgentConfig(
                    name=f"relation_extractor_{g_idx}",
                    system_prompt=skill.get_system_prompt(),
                    tools=[],
                    max_tool_calls=1,
                    model_config=LLMConfig(
                        model=self.llm_config.model,
                        api_key=self.llm_config.api_key,
                        api_base=self.llm_config.api_base,
                        max_tokens=self.llm_config.max_tokens,
                    ),
                )
                agent = Agent(agent_config, self.memory, self.llm_config)
                agent.on_event(lambda et, d: _emit_safe(et, d))

                entities_summary = json.dumps(grp_entities, ensure_ascii=False, indent=2)
                # Use filtered ontology guide for large schemas
                active_guide = filtered_ontology_guides[g_idx] if (use_filtered_relations and g_idx < len(filtered_ontology_guides)) else ontology_guide
                prompt = build_relation_extraction_prompt(
                    ontology_guide=active_guide,
                    entities_summary=entities_summary,
                    texts=group_text,
                )

                call_start = time.time()
                self.log.agent_call(f"relation_extractor_{g_idx}", prompt_size=len(prompt), group_index=g_idx, thread=_tid)
                _emit_safe("agent_call_start", {
                    "agent": f"relation_extractor_{g_idx}",
                    "chunk_index": g_idx,
                    "prompt_size": len(prompt),
                    "entity_count": len(grp_entities),
                    "thread": _tid,
                })

                relations = []
                try:
                    result = agent.run_structured(prompt, skill.get_output_schema())
                except Exception:
                    result = None

                duration = time.time() - call_start
                rels_found = len(result.get("relations", [])) if result else 0
                self.log.agent_result(f"relation_extractor_{g_idx}", duration=duration, relations=rels_found)
                _emit_safe("agent_call_end", {
                    "agent": f"relation_extractor_{g_idx}",
                    "chunk_index": g_idx,
                    "duration": duration,
                    "has_result": result is not None,
                    "relations_found": rels_found,
                })

                if result:
                    for r_data in result.get("relations", []):
                        relations.append(Relation(
                            subject=r_data.get("subject", ""),
                            predicate=r_data.get("predicate", r_data.get("relation", "")),
                            object=r_data.get("object", ""),
                            keywords=r_data.get("keywords", ""),
                            description=r_data.get("description", ""),
                            confidence=r_data.get("confidence", 1.0),
                            evidence=r_data.get("evidence"),
                        ))

                with relation_lock:
                    completed_count[0] += 1
                    _emit_safe("chunk_progress", {
                        "phase": "relation_extraction",
                        "total_chunks": len(doc_groups),
                        "current": completed_count[0],
                        "status": "done",
                        "new_relations": len(relations),
                        "total_relations": len(all_relations) + len(relations),
                        "elapsed": time.time() - chunk_start_time,
                    })

                return relations

            max_workers = min(self.config.max_concurrent_agents, max(len(doc_groups), 1))
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                futures = {
                    executor.submit(_process_relation_group, i, group_texts[i], group_entities[i]): i
                    for i in range(len(doc_groups))
                }
                for future in as_completed(futures):
                    try:
                        group_relations = future.result()
                        with relation_lock:
                            all_relations.extend(group_relations)
                    except Exception as e:
                        i = futures[future]
                        _emit_safe("phase_failed", {
                            "phase": f"relation_extraction_group_{i}",
                            "error": str(e),
                        })
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                try:
                    executor.shutdown(wait=False)
                except Exception:
                    pass

            seen_rel: set[tuple[str, str, str]] = set()
            unique_relations = []
            for r in all_relations:
                key = (r.subject.strip(), r.predicate.strip(), r.object.strip())
                if key not in seen_rel:
                    seen_rel.add(key)
                    unique_relations.append(r)

            phase.output = ExtractionResult(relations=unique_relations)
            phase.status = PhaseStatus.COMPLETED
            self._emit("phase_complete", {
                "phase": "relation_extraction",
                "relations": len(unique_relations),
                "groups": len(doc_groups),
            })

        except Exception as e:
            phase.status = PhaseStatus.FAILED
            phase.error_message = str(e)
            self._emit("phase_failed", {"phase": "relation_extraction", "error": str(e)})

        phase.completed_at = datetime.now()
        return phase

    # ── Schema Canonicalization: map open relations to ontology ─────────────

    def _canonicalize_relations(
        self,
        ontology: Optional[Ontology],
        relations: list[dict[str, Any]],
        entities: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Map extracted relation predicates to the closest ontology relation.

        Uses the edc-style approach: for each relation not matching the ontology,
        present candidate schema relations with definitions and ask the LLM
        to pick the best match via multiple choice.
        Returns a mapping of {original_predicate: canonical_predicate}.
        """
        if not ontology or not ontology.relation_types:
            return {}

        # Build schema relation definitions
        rt_defs = {}
        for rt in ontology.relation_types:
            if rt.description:
                rt_defs[rt.name] = rt.description
            else:
                rt_defs[rt.name] = f"关系: {rt.name}"

        # Find relations that don't match the ontology
        valid_names = set(ontology.relation_type_names)
        unique_predicates = list({r.get("predicate", r.get("relation", "")) for r in relations})
        unmatched = [p for p in unique_predicates if p and p not in valid_names]

        if not unmatched:
            return {}

        # Batch all unmatched predicates into a single LLM call
        # (was: up to 20 sequential calls — now 1 batch call)
        canonical_map = {}
        unmatched_batch = unmatched[:20]
        if not unmatched_batch:
            return {}

        # Build the choice definitions once
        choices = list(rt_defs.items())[:12]
        if not choices:
            return {}

        choice_lines = []
        for i, (name, desc) in enumerate(choices):
            choice_lines.append(f"{chr(65+i)}. {name}: {desc}")
        choice_lines.append(f"{chr(65+len(choices))}. 以上都不是（保留原关系名）")
        choices_text = "\n".join(choice_lines)

        # Build a batch prompt listing all unmatched predicates
        pred_list = "\n".join(f"{j+1}. \"{p}\"" for j, p in enumerate(unmatched_batch))
        batch_prompt = f"""给定以下提取的关系和候选 Schema 关系，为每个提取的关系选择最匹配的一项。

候选 Schema 关系（含定义）:
{choices_text}

提取的关系:
{pred_list}

请以 JSON 格式返回映射结果。每个提取的关系对应一个选项字母（A、B、C...），
如果完全无法匹配则返回 "NONE"。

输出格式（严格 JSON，不要其他文字）:
{{"mappings": {{"提取关系名1": "A", "提取关系名2": "NONE", ...}}}}"""

        agent_cfg = AgentConfig(
            name="schema_canonicalizer_batch",
            system_prompt="你是 Schema 映射专家。只输出 JSON，不要任何解释。",
            tools=[],
            max_tool_calls=1,
        )
        try:
            agent = Agent(agent_cfg, self.memory, self.llm_config)
            response = agent.run(batch_prompt, max_iterations=1).strip()

            # Parse JSON from response
            json_match = _re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group(0))
                mappings = parsed.get("mappings", {})
                for pred, letter in mappings.items():
                    letter = str(letter).strip().upper()
                    if letter == "NONE" or not letter:
                        continue
                    match = _re.match(r'([A-Z])', letter)
                    if match:
                        idx = ord(match.group(1)) - ord('A')
                        if 0 <= idx < len(choices):
                            canonical_map[pred] = choices[idx][0]
                            self.log.info(
                                f"Schema canonicalization: '{pred}' -> '{choices[idx][0]}'"
                            )
        except Exception:
            pass

        return canonical_map

    # ── Phase: Quality Check ───────────────────────────────────────────────

    def _run_phase_quality_check(
        self,
        ontology: Optional[Ontology],
        entities: list[dict[str, Any]],
        relations: list[dict[str, Any]],
        documents: list[Document],
    ) -> PhaseResult:
        phase = PhaseResult(phase_name="quality_check", status=PhaseStatus.RUNNING)
        phase.started_at = datetime.now()

        self._emit("phase_start", {"phase": "quality_check"})

        try:
            skill = get_skill("quality_checker", self.llm_config)
            ontology_guide = ontology.to_extraction_guide() if ontology else ""
            all_text = "\n\n".join([f"[Doc {i}] {d.text[:500]}" for i, d in enumerate(documents)])

            # Truncate extraction summary to avoid overwhelming the LLM context.
            # Large entity/relation lists cause tool-call loops as the LLM tries
            # validate_against_ontology on malformed JSON.
            MAX_ENTITIES_IN_QC = self.config.max_entities_in_qc
            MAX_RELATIONS_IN_QC = self.config.max_relations_in_qc
            MAX_QC_SUMMARY_CHARS = 60000

            entities_for_qc = entities[:MAX_ENTITIES_IN_QC]
            relations_for_qc = relations[:MAX_RELATIONS_IN_QC]
            truncation_note = ""
            if len(entities) > MAX_ENTITIES_IN_QC or len(relations) > MAX_RELATIONS_IN_QC:
                truncation_note = (
                    f"\n(注意: 原始结果共有 {len(entities)} 实体, {len(relations)} 关系。"
                    f"以下仅展示前 {MAX_ENTITIES_IN_QC} 实体和 {MAX_RELATIONS_IN_QC} 关系用于审核。)"
                )

            extraction_summary = json.dumps({
                "entities": entities_for_qc,
                "relations": relations_for_qc,
            }, ensure_ascii=False, indent=2)

            if len(extraction_summary) > MAX_QC_SUMMARY_CHARS:
                extraction_summary = extraction_summary[:MAX_QC_SUMMARY_CHARS] + truncation_note

            self._emit("agent_call_start", {
                "agent": "quality_checker",
                "chunk_index": 0,
                "prompt_size": len(extraction_summary),
                "entity_count": len(entities_for_qc),
                "relation_count": len(relations_for_qc),
            })

            agent = self._create_skill_agent("quality_checker", skill)
            prompt = build_quality_check_prompt(
                ontology_guide=ontology_guide,
                extraction_summary=extraction_summary,
                original_texts=all_text[:30000],  # Also limit original texts
            )
            result = agent.run_structured(prompt, skill.get_output_schema())

            if result is None:
                result = {}
                self.log.warning("Quality checker returned None, using empty result")
            elif not isinstance(result, dict):
                self.log.warning(
                    f"Quality checker returned {type(result).__name__} instead of dict, "
                    f"wrapping as corrections"
                )
                result = {"corrections": result if isinstance(result, list) else [result]}

            # ── Schema Canonicalization: map unmatched relations ──
            canonical_map = self._canonicalize_relations(ontology, relations_for_qc, entities_for_qc)
            if canonical_map:
                result["canonical_mappings"] = canonical_map
                # Apply canonicalization to the original relations list
                remapped = 0
                for r in relations:
                    pred = r.get("predicate", r.get("relation", ""))
                    if pred in canonical_map:
                        r["predicate"] = canonical_map[pred]
                        r["_original_predicate"] = pred
                        remapped += 1
                if remapped > 0:
                    self.log.info(f"Schema canonicalization: remapped {remapped} relations")

            phase.output = ExtractionResult(metadata=result)
            phase.status = PhaseStatus.COMPLETED
            qs = result.get("overall_quality_score", 0)
            self._emit("phase_complete", {
                "phase": "quality_check",
                "quality_score": qs,
                "canonicalized": len(canonical_map),
            })

        except Exception as e:
            phase.status = PhaseStatus.FAILED
            phase.error_message = str(e)
            self._emit("phase_failed", {"phase": "quality_check", "error": str(e)})

        phase.completed_at = datetime.now()
        return phase

    # ── Phase: Triple Construction ─────────────────────────────────────────

    def _run_phase_triple_construction(
        self,
        ontology: Optional[Ontology],
        entities: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> PhaseResult:
        phase = PhaseResult(phase_name="triple_construction", status=PhaseStatus.RUNNING)
        phase.started_at = datetime.now()

        self._emit("phase_start", {"phase": "triple_construction"})

        try:
            entity_index: dict[str, list[Entity]] = {}
            norm_index: dict[str, list[tuple[str, Entity]]] = {}
            for e_data in entities:
                name = (e_data.get("name") or "").strip()
                etype = e_data.get("type") or e_data.get("entity_type") or ""
                if not name:
                    continue
                if name not in entity_index:
                    entity_index[name] = []
                entity_index[name].append(Entity(
                    name=name,
                    type=etype,
                    mention=e_data.get("mention"),
                    confidence=e_data.get("confidence", 1.0),
                ))
                nkey = self._normalize_name(name)
                if nkey not in norm_index:
                    norm_index[nkey] = []
                norm_index[nkey].append((name, entity_index[name][-1]))

            triples: list[Triple] = []
            seen = set()
            fuzzy_matches = 0
            dropped = 0

            for r_data in relations:
                subj_name = (r_data.get("subject") or "").strip()
                obj_name = (r_data.get("object") or "").strip()
                predicate = r_data.get("predicate") or r_data.get("relation") or ""
                confidence = r_data.get("confidence", 1.0)
                evidence = r_data.get("evidence", "")

                if not subj_name or not obj_name or not predicate:
                    dropped += 1
                    continue

                subj_entity, subj_penalty, subj_method = self._fuzzy_match_entity(
                    subj_name, entity_index, norm_index
                )
                obj_entity, obj_penalty, obj_method = self._fuzzy_match_entity(
                    obj_name, entity_index, norm_index
                )

                if subj_entity is None or obj_entity is None:
                    dropped += 1
                    if dropped <= 5:
                        self.log.debug(
                            f"Triple dropped: '{subj_name}'->'{obj_name}' "
                            f"(subj_match={subj_method}, obj_match={obj_method})"
                        )
                    continue

                if subj_method != "exact" or obj_method != "exact":
                    fuzzy_matches += 1

                total_penalty = subj_penalty + obj_penalty
                adjusted_confidence = max(0.0, confidence - total_penalty)

                if ontology:
                    rt = ontology.get_relation_type(predicate)
                    if rt:
                        subj_candidates = [subj_entity]
                        if subj_name in entity_index:
                            subj_candidates = entity_index[subj_name]
                        if rt.domain:
                            matched = [e for e in subj_candidates if e.type == rt.domain]
                            if matched:
                                subj_entity = matched[0]

                        obj_candidates = [obj_entity]
                        if obj_name in entity_index:
                            obj_candidates = entity_index[obj_name]
                        if rt.range:
                            matched = [e for e in obj_candidates if e.type == rt.range]
                            if matched:
                                obj_entity = matched[0]

                key = (subj_entity.name, subj_entity.type, predicate, obj_entity.name, obj_entity.type)
                if key in seen:
                    continue
                seen.add(key)

                triples.append(Triple(
                    subject=subj_entity,
                    predicate=predicate,
                    object=obj_entity,
                    confidence=adjusted_confidence,
                    evidence=evidence,
                ))

            # Fallback: for dropped relations, create placeholder entities
            if dropped > 0:
                self.log.info(
                    f"Triple construction: attempting fallback for {dropped} unmatched relations"
                )
                for r_data in relations:
                    subj_name = (r_data.get("subject") or "").strip()
                    obj_name = (r_data.get("object") or "").strip()
                    predicate = r_data.get("predicate") or r_data.get("relation") or ""
                    confidence = r_data.get("confidence", 1.0)
                    evidence = r_data.get("evidence", "")

                    if not subj_name or not obj_name or not predicate:
                        continue

                    subj_entity, _, subj_method = self._fuzzy_match_entity(
                        subj_name, entity_index, norm_index
                    )
                    obj_entity, _, obj_method = self._fuzzy_match_entity(
                        obj_name, entity_index, norm_index
                    )

                    if subj_entity is None:
                        subj_entity = Entity(
                            name=subj_name, type="Entity",
                            confidence=0.3, mention=subj_name,
                            attributes={"_fallback": "Created from unmatched relation"},
                        )
                        entity_index[subj_name] = [subj_entity]
                    if obj_entity is None:
                        obj_entity = Entity(
                            name=obj_name, type="Entity",
                            confidence=0.3, mention=obj_name,
                            attributes={"_fallback": "Created from unmatched relation"},
                        )
                        entity_index[obj_name] = [obj_entity]

                    key = (subj_entity.name, subj_entity.type, predicate, obj_entity.name, obj_entity.type)
                    if key in seen:
                        continue
                    seen.add(key)

                    triples.append(Triple(
                        subject=subj_entity,
                        predicate=predicate,
                        object=obj_entity,
                        confidence=min(confidence, 0.35),
                        evidence=evidence,
                    ))

            self.log.info(
                f"Triple construction: {len(relations)} relations -> {len(triples)} triples "
                f"(fuzzy_matches={fuzzy_matches}, dropped={dropped})"
            )
            phase.output = ExtractionResult(triples=triples)
            phase.status = PhaseStatus.COMPLETED
            self._emit("phase_complete", {"phase": "triple_construction", "triples": len(triples)})

        except Exception as e:
            phase.status = PhaseStatus.FAILED
            phase.error_message = str(e)
            self._emit("phase_failed", {"phase": "triple_construction", "error": str(e)})

        phase.completed_at = datetime.now()
        return phase

    # ── Phase: Agent Code Extraction ───────────────────────────────────────

    def _run_phase_agent_code_extraction(self, docs: list[Document], ontology: Ontology) -> PhaseResult:
        """Phase 1.5: Let Agent write Python extraction code for the dataset and execute in sandbox."""
        phase = PhaseResult(phase_name="agent_code_extraction", status=PhaseStatus.RUNNING)
        phase.started_at = datetime.now()
        self._emit("phase_start", {"phase": "agent_code_extraction"})

        try:
            samples = []
            for d in docs[:5]:
                samples.append({
                    "file": d.metadata.get("filename", "unknown"),
                    "ext": d.metadata.get("ext", ""),
                    "size": len(d.text),
                    "preview": d.text[:1500],
                })

            ontology_json = json.dumps({
                "entity_types": [{"name": et.name, "description": et.description} for et in ontology.entity_types],
                "relation_types": [{"name": rt.name, "description": rt.description,
                                   "domain": rt.domain, "range": rt.range} for rt in ontology.relation_types],
            }, ensure_ascii=False)

            prompt_gen_input = f"""请为以下数据编写一个定制化的实体和关系抽取 prompt。

## 本体定义
{ontology_json}

## 数据样本
{json.dumps(samples, ensure_ascii=False, indent=2)[:4000]}

## 要求
编写一个 LLM 抽取 prompt（不是代码！）。这个 prompt 将被发送给 LLM 来执行实际的实体和关系抽取。
prompt 应该:
1. 明确列出要抽取的实体类型和关系类型（来自本体定义）
2. 给出 2-3 个从样本中提取的具体示例
3. 指定输出 JSON 格式
4. 针对数据特点给出抽取提示

只输出 prompt 文本，不要任何解释或标记。"""
            prompt_agent_cfg = AgentConfig(name="prompt_gen", system_prompt="You write LLM extraction prompts. Output only the prompt text, no markdown.", tools=[], max_tool_calls=1)
            prompt_agent = Agent(prompt_agent_cfg, self.memory, self.llm_config)
            prompt_agent.on_event(lambda et, d: self._emit(et, d))
            generated_prompt = prompt_agent.run(prompt_gen_input, max_iterations=1)
            self.log.debug(f"Agent generated extraction prompt ({len(generated_prompt)} chars)")

            if not generated_prompt or len(generated_prompt) < 50:
                self.log.warning(f"Agent prompt generation: too short ({len(generated_prompt)} chars)")
                phase.status = PhaseStatus.COMPLETED
                phase.output = ExtractionResult()
                return phase

            # Save generated extraction prompt for audit/inspection
            ts_prompt = datetime.now().strftime("%Y%m%dT%H%M%S")
            self.memory.save_generated_code(
                f"extraction_prompt_{ts_prompt}.txt",
                generated_prompt,
            )

            docs_json = json.dumps([{"text": d.text[:8000], "file": d.metadata.get("filename",""), "ext": d.metadata.get("ext","")} for d in docs[:20]], ensure_ascii=False)
            from ..tools import execute_tool
            result_raw = execute_tool("extract_with_llm_prompt", {
                "extraction_prompt": generated_prompt,
                "data_text": docs_json[:80000],
                "ontology_json": ontology_json,
            })
            result = result_raw.data if result_raw.success else {"success": False, "error": result_raw.error}
            if result.get("prompt_tokens") or result.get("completion_tokens"):
                self._emit("token_usage", {
                    "_agent_id": "phase_1_5_llm_extractor",
                    "prompt_tokens": result.get("prompt_tokens", 0),
                    "completion_tokens": result.get("completion_tokens", 0),
                })
            self.log.info(
                f"Agent LLM extraction result: success={result_raw.success}, "
                f"entities={result.get('entity_count', 0)}, relations={result.get('relation_count', 0)}"
            )
            if not result.get("success"):
                self.log.warning(f"Agent LLM extraction FAILED: {result.get('error', 'unknown')}")

            if result.get("success") and result.get("entity_count", 0) > 0:
                entities = [Entity(name=e.get("name",""), type=e.get("type", e.get("entity_type","Entity")),
                                   description=e.get("description", ""),
                                   confidence=e.get("confidence", 0.9))
                            for e in result.get("entities", [])]
                relations = [Relation(subject=r.get("subject",""), predicate=r.get("predicate", r.get("relation","")),
                                      object=r.get("object",""),
                                      keywords=r.get("keywords", ""), description=r.get("description", ""),
                                      confidence=r.get("confidence", 0.85),
                                      evidence="agent code extraction")
                             for r in result.get("relations", [])]
                seen_e = set()
                deduped_e = [e for e in entities if (e.name.strip().lower(), e.type) not in seen_e and not seen_e.add((e.name.strip().lower(), e.type))]
                seen_r = set()
                deduped_r = [r for r in relations if r.subject and r.predicate and r.object and (r.subject.strip().lower(), r.predicate, r.object.strip().lower()) not in seen_r and not seen_r.add((r.subject.strip().lower(), r.predicate, r.object.strip().lower()))]
                self.log.info(f"Agent code extraction: {len(deduped_e)} entities, {len(deduped_r)} relations")
                phase.output = ExtractionResult(entities=deduped_e, relations=deduped_r)
                phase.status = PhaseStatus.COMPLETED
                self._emit("phase_complete", {"phase": "agent_code_extraction", "entities": len(deduped_e), "relations": len(deduped_r)})
            else:
                phase.status = PhaseStatus.COMPLETED
                phase.output = ExtractionResult()
        except Exception as e:
            self.log.error(f"Phase 1.5 agent_code_extraction failed: {e}")
            self.log.debug(traceback.format_exc())
            phase.status = PhaseStatus.FAILED
            phase.error_message = str(e)
        phase.completed_at = datetime.now()
        return phase

    # ── Phase: Co-occurrence Graph ─────────────────────────────────────────

    def _run_phase_co_occurrence(self, docs: list[Document], ontology: Ontology) -> PhaseResult:
        """Phase 3.5: Build entity co-occurrence graph.

        A parallel branch to relation extraction — counts co-occurrence frequency
        of entities across all documents, generating a weighted co-occurrence network.
        """
        phase = PhaseResult(phase_name="co_occurrence", status=PhaseStatus.RUNNING)
        phase.started_at = datetime.now()
        self._emit("phase_start", {"phase": "co_occurrence"})

        try:
            entities = self._phase_outputs.get("entity_extraction", {}).get("entities", [])
            if not entities:
                phase.status = PhaseStatus.COMPLETED
                phase.output = ExtractionResult()
                return phase

            entity_names: dict[str, tuple[str, str]] = {}
            for e in entities:
                name = (e.get("name") or "").strip()
                if name and len(name) >= 2:
                    etype = e.get("type") or e.get("entity_type") or "Entity"
                    entity_names[name] = (name, etype)

            rt_domain_range: dict[tuple[str, str], list[str]] = {}
            if ontology:
                for rt in ontology.relation_types:
                    if rt.domain and rt.range:
                        key = (rt.domain, rt.range)
                        if key not in rt_domain_range:
                            rt_domain_range[key] = []
                        rt_domain_range[key].append(rt.name)

            cooccur_counts: dict[tuple[str, str], tuple[int, set[str]]] = defaultdict(
                lambda: (0, set())
            )

            for doc in docs:
                doc_name = doc.metadata.get("filename", doc.source)
                paragraphs = _re.split(r'\n\s*\n', doc.text)
                for para in paragraphs:
                    if len(para) < 50:
                        continue
                    para_lower = para.lower()
                    found_in_para = []
                    para_words = set(para_lower.split())
                    for ename, (ename_orig, etype) in entity_names.items():
                        ename_lower = ename.lower()
                        if ' ' not in ename_lower and len(ename_lower) >= 2:
                            if ename_lower in para_words:
                                found_in_para.append((ename_orig, etype))
                        elif ename_lower in para_lower:
                            found_in_para.append((ename_orig, etype))
                    for i in range(len(found_in_para)):
                        for j in range(i + 1, min(len(found_in_para), i + 10)):
                            e1_name, e1_type = found_in_para[i]
                            e2_name, e2_type = found_in_para[j]
                            if e1_name == e2_name:
                                continue
                            pair_key = tuple(sorted([(e1_name, e1_type), (e2_name, e2_type)],
                                                   key=lambda x: x[0]))
                            cnt, srcs = cooccur_counts[pair_key]
                            cooccur_counts[pair_key] = (cnt + 1, srcs | {doc_name})

            relations: list[Relation] = []
            triples: list[Triple] = []
            max_count = max((c for c, _ in cooccur_counts.values()), default=1)

            for (e1_name, e1_type), (e2_name, e2_type) in cooccur_counts:
                cnt, srcs = cooccur_counts[((e1_name, e1_type), (e2_name, e2_type))]
                freq_weight = cnt / max_count if max_count > 0 else 0
                confidence = 0.3 + 0.6 * freq_weight

                predicate = "co_occur"
                if (e1_type, e2_type) in rt_domain_range:
                    predicate = rt_domain_range[(e1_type, e2_type)][0]
                elif (e2_type, e1_type) in rt_domain_range:
                    e1_name, e2_name = e2_name, e1_name
                    e1_type, e2_type = e2_type, e1_type
                    predicate = rt_domain_range[(e1_type, e2_type)][0]

                evidence = f"Co-occurred {cnt}x in: {', '.join(sorted(srcs)[:3])}"
                if len(srcs) > 3:
                    evidence += f" and {len(srcs) - 3} more docs"

                relations.append(Relation(
                    subject=e1_name, predicate=predicate,
                    object=e2_name, confidence=round(confidence, 2),
                    evidence=evidence,
                ))
                triples.append(Triple(
                    subject=Entity(name=e1_name, type=e1_type, confidence=0.9),
                    predicate=predicate,
                    object=Entity(name=e2_name, type=e2_type, confidence=0.9),
                    confidence=round(confidence, 2),
                    evidence=evidence,
                ))

            relations.sort(key=lambda r: r.confidence, reverse=True)
            if len(relations) > 5000:
                relations = relations[:5000]
            triples.sort(key=lambda t: t.confidence, reverse=True)
            if len(triples) > 5000:
                triples = triples[:5000]

            self.log.info(
                f"Co-occurrence: {len(entity_names)} entities in {len(docs)} docs "
                f"→ {len(relations)} co-occurrence relations"
            )
            phase.output = ExtractionResult(relations=relations, triples=triples)
            phase.status = PhaseStatus.COMPLETED
            self._emit("phase_complete", {
                "phase": "co_occurrence",
                "relations": len(relations),
                "triples": len(triples),
            })

        except Exception as e:
            phase.status = PhaseStatus.FAILED
            phase.error_message = str(e)
            self._emit("phase_failed", {"phase": "co_occurrence", "error": str(e)})

        phase.completed_at = datetime.now()
        return phase

    # ── Phase: Structured Data Extraction ──────────────────────────────────

    def _run_phase_structured_extraction(self, ontology: Ontology) -> PhaseResult:
        """Phase 2.5: LLM-guided column mapping for tabular data (CSV/XLSX)."""
        phase = PhaseResult(phase_name="structured_extraction", status=PhaseStatus.RUNNING)
        phase.started_at = datetime.now()
        self._emit("phase_start", {"phase": "structured_extraction"})

        try:
            rows = self._structured_rows
            if not rows or not ontology:
                phase.status = PhaseStatus.COMPLETED
                phase.output = ExtractionResult()
                return phase

            # Build column mapping prompt
            cols = [c for c in rows[0].keys() if c != "_source"]
            sample = json.dumps(
                [{k: v for k, v in list(r.items())[:10] if k != "_source"} for r in rows[:5]],
                ensure_ascii=False,
            )
            mapping_prompt = (
                f"Map these columns to the ontology. "
                f"Columns: {json.dumps(cols)}. "
                f"Sample: {sample[:3000]}. "
                f"Ontology: {ontology.to_extraction_guide()[:1000]}. "
                f'Return JSON: {{"column_mapping": {{"col": {{"role": "id"|"reference"|"attribute", '
                f'"entity_type": "...", "relation": "..."}}}}}}'
            )

            agent_cfg = AgentConfig(
                name="col_mapper", system_prompt="JSON only.", tools=[], max_tool_calls=1,
            )
            agent = Agent(agent_cfg, self.memory, self.llm_config)
            agent.on_event(lambda et, d: self._emit(et, d))
            mapping = agent.run_structured(mapping_prompt, {
                "type": "object",
                "properties": {"column_mapping": {"type": "object"}},
            })

            if not mapping or not mapping.get("column_mapping"):
                phase.status = PhaseStatus.COMPLETED
                phase.output = ExtractionResult()
                return phase

            col_map = mapping["column_mapping"]
            entities: list[Entity] = []
            relations: list[Relation] = []
            id_cols = [c for c, i in col_map.items() if i.get("role") == "id"]
            ref_cols = [c for c, i in col_map.items() if i.get("role") == "reference"]

            for row in rows:
                row_ents: list[str] = []
                # Extract ID columns as entities
                for col in id_cols:
                    v = str(row.get(col, "")).strip()
                    if v:
                        entities.append(Entity(
                            name=v,
                            type=col_map[col].get("entity_type", "Entity"),
                            confidence=0.95,
                        ))
                        row_ents.append(v)

                # Extract reference columns as entities + relations
                for col in ref_cols:
                    v = str(row.get(col, "")).strip()
                    if not v:
                        continue
                    info = col_map[col]
                    etype = info.get("entity_type", "Entity")
                    rel = info.get("relation", "")
                    # Handle multi-value fields (comma-separated)
                    if info.get("is_multi"):
                        names = [n.strip() for n in v.replace("，", ",").split(",")]
                    else:
                        names = [v]
                    for name in names:
                        if not name:
                            continue
                        entities.append(Entity(name=name, type=etype, confidence=0.9))
                        for id_name in row_ents:
                            if id_name != name and rel:
                                relations.append(Relation(
                                    subject=id_name, predicate=rel, object=name,
                                    confidence=0.9,
                                    evidence=f"Row from {row.get('_source', '')}",
                                ))

            # Deduplicate
            seen_e: set[tuple[str, str]] = set()
            deduped_e = [
                e for e in entities
                if (e.name, e.type) not in seen_e and not seen_e.add((e.name, e.type))
            ]
            seen_r: set[tuple[str, str, str]] = set()
            deduped_r = [
                r for r in relations
                if r.subject and r.predicate and r.object
                and (r.subject, r.predicate, r.object) not in seen_r
                and not seen_r.add((r.subject, r.predicate, r.object))
            ]

            self.log.info(
                f"Structured extraction: {len(rows)} rows -> "
                f"{len(deduped_e)} entities, {len(deduped_r)} relations"
            )
            phase.output = ExtractionResult(entities=deduped_e, relations=deduped_r)
            phase.status = PhaseStatus.COMPLETED
            self._emit("phase_complete", {
                "phase": "structured_extraction",
                "entities": len(deduped_e),
                "relations": len(deduped_r),
            })

        except Exception as e:
            phase.status = PhaseStatus.FAILED
            phase.error_message = str(e)

        phase.completed_at = datetime.now()
        return phase

    # ── Final Result Aggregation ───────────────────────────────────────────

    def _build_final_result(self) -> ExtractionResult:
        """Build the final ExtractionResult from all phases, applying quality checks."""
        all_entities = []
        all_relations = []
        all_triples = []

        qc_metadata = {}
        for phase in self.memory.workflow.phases if self.memory._workflow else []:
            if phase.phase_name == "quality_check" and phase.output and phase.output.metadata:
                qc_metadata = phase.output.metadata

        rejected_entities: set[tuple[str, str]] = set()
        rejected_relations: set[tuple[str, str, str]] = set()
        corrections: dict[str, dict] = {}

        if qc_metadata:
            for item in qc_metadata.get("rejected_items", []):
                if not isinstance(item, dict):
                    continue
                if item.get("type") in ("entity", "entity_type"):
                    name = (item.get("name") or "").strip()
                    etype = (item.get("entity_type") or item.get("type_val") or "").strip()
                    if name:
                        rejected_entities.add((name, etype))
                elif item.get("type") in ("relation", "relation_direction"):
                    subj = (item.get("subject") or "").strip()
                    pred = (item.get("predicate") or "").strip()
                    obj = (item.get("object") or "").strip()
                    if subj and pred and obj:
                        rejected_relations.add((subj, pred, obj))

            for corr in qc_metadata.get("corrections", []):
                if not isinstance(corr, dict):
                    continue
                corr_type = corr.get("type", "")
                corrected = corr.get("corrected", {})
                original = corr.get("original", {})
                if corr_type == "entity_type" and isinstance(corrected, dict):
                    old_name = (original.get("name") or "").strip() if isinstance(original, dict) else ""
                    old_type = (original.get("type") or "").strip() if isinstance(original, dict) else ""
                    new_type = (corrected.get("type") or "").strip()
                    if old_name and new_type:
                        corrections[(old_name, old_type)] = {"new_type": new_type}

        for phase in self.memory.workflow.phases if self.memory._workflow else []:
            if not phase.output:
                continue
            if phase.phase_name == "quality_check":
                continue

            for e in phase.output.entities:
                key = (e.name, e.type)
                if key in rejected_entities:
                    continue
                if key in corrections and corrections[key].get("new_type"):
                    e = Entity(
                        name=e.name,
                        type=corrections[key]["new_type"],
                        attributes=e.attributes,
                        mention=e.mention,
                        confidence=e.confidence,
                    )
                all_entities.append(e)

            for r in phase.output.relations:
                key = (r.subject.strip(), r.predicate.strip(), r.object.strip())
                if key in rejected_relations:
                    continue
                all_relations.append(r)

            all_triples.extend(phase.output.triples)

        seen_entities: dict[tuple[str, str], Entity] = {}
        for e in all_entities:
            key = (e.name.strip(), e.type)
            if key not in seen_entities or e.confidence > seen_entities[key].confidence:
                seen_entities[key] = e

        seen_relations: dict[tuple[str, str, str], Relation] = {}
        for r in all_relations:
            key = (r.subject.strip(), r.predicate.strip(), r.object.strip())
            if key not in seen_relations or r.confidence > seen_relations[key].confidence:
                seen_relations[key] = r

        rejected_count = len(rejected_entities) + len(rejected_relations)
        if rejected_count > 0:
            self.log.info(f"Quality check: filtered {len(rejected_entities)} entities, "
                          f"{len(rejected_relations)} relations")

        return ExtractionResult(
            entities=list(seen_entities.values()),
            relations=list(seen_relations.values()),
            triples=all_triples,
        )
