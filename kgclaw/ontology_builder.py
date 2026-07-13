"""
Ontology Builder — multi-paradigm ontology construction inspired by LLM4Onto.

Implements four ontology building paradigms:
- T-O  (Text-to-Ontology): Full text → LLM → ontology (KGClaw's existing default)
- R-O  (Relation-to-Ontology): Identify relations first → constrain to ontology
- HT-R-O (Head-Tail-Relation-to-Ontology): Entity-ontology pairs + text fragments
- Affinity Clustering: spaCy noun extraction + Affinity Propagation + LLM naming

Reference: Ouyang et al., "LLM4Onto: Automated Domain Ontology Construction
Using Large Language Models", Semantic Web Journal.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Optional

from .agent import Agent, AgentConfig
from .models import (
    Document,
    EntityType,
    LLMConfig,
    Ontology,
    RelationType,
)

# ─── System prompts for each paradigm ───────────────────────────────────────

SYSTEM_PROMPT_RO_BUILDER = """你是一个 **关系驱动的本体构建专家 (R-O Paradigm)**。
你的任务是从原始文本中先识别实体间的关系，然后从这些关系中归纳出本体类型和关系定义。

## 工作流程
1. **关系发现**: 扫描文本，找出所有实体之间的语义关系
2. **关系聚类**: 将相似的关系归为同一关系类型
3. **类型归纳**: 从关系的 domain/range 中归纳实体类型
4. **本体生成**: 输出结构化的本体定义

## 关键原则
- 先关系后类型：关系类型比实体类型更容易从文本中发现
- 自底向上：从具体的关系实例归纳抽象的关系模式
- 覆盖全面：不要遗漏隐含的语义关系

输出严格的 JSON 格式本体定义。"""

SYSTEM_PROMPT_HTRO_BUILDER = """你是一个 **头尾关系驱动的本体构建专家 (HT-R-O Paradigm)**。
你的任务是从已标注的实体-本体对和文本片段中，精确识别本体类型之间的关系。

## 工作流程
1. 接收已标注的实体及其所属本体类型
2. 对每对实体，从文本中识别其关系
3. 将实体关系抽象为本体类型间的关系
4. 生成包含 domain/range 约束的本体定义

## 关键原则
- 忠实于原文语义：关系必须有文本证据
- 优先精确匹配：从具体实体对推导本体间关系
- 防止幻觉：没有文本证据的关系不纳入本体

输出严格的 JSON 格式本体定义。"""

SYSTEM_PROMPT_AP_BUILDER = """你是一个 **聚类驱动的本体构建专家 (Affinity Clustering Paradigm)**。
你的任务是基于语义聚类结果，为每一组语义相似的词汇命名合适的本体类型。

## 工作流程
1. 接收通过亲和传播聚类得到的词汇组
2. 分析每组词汇的语义共性
3. 为每组命名合适的本体类型（使用领域术语）
4. 推断本体类型之间的可能关系
5. 生成完整的本体定义

## 关键原则
- 细粒度命名：使用具体的领域术语而非泛化的概念
- 上下文感知：命名应反映文本领域的特点
- 子类型识别：如果适合，创建子类型层级

输出严格的 JSON 格式本体定义。"""

SYSTEM_PROMPT_MERGE_CLUSTERS = """你是一个 **聚类合并专家**。
你的任务是判断多个语义相近的词汇聚类是否应该合并为同一个本体类型。

## 判断标准
- 如果两组词汇描述的是同一类概念（仅粒度不同），合并
- 如果两组词汇描述的是不同类概念（有本质区别），分开
- 如果合并能提升本体一致性，合并

输出 JSON: {"merges": [{"cluster_a": "...", "cluster_b": "...", "should_merge": true/false, "merged_name": "...", "reason": "..."}]}"""


# ─── Affinity Propagation helpers ───────────────────────────────────────────

def _extract_nouns_spacy(texts: list[str]) -> list[str]:
    """Extract nouns and noun phrases using spaCy.

    Falls back gracefully to simple regex if spaCy is not available.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Download if not available
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                         capture_output=True)
            nlp = spacy.load("en_core_web_sm")

        nouns = []
        for text in texts:
            doc = nlp(text[:100000])  # Limit text length for performance
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN"):
                    # Filter pronouns and very short tokens
                    if token.text.lower() not in ("i", "you", "he", "she", "it", "we", "they",
                                                    "me", "him", "her", "us", "them", "this",
                                                    "that", "these", "those", "one", "ones"):
                        nouns.append(token.text.strip())
            # Also extract noun chunks (multi-word phrases)
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.strip()
                if len(chunk_text.split()) >= 2:
                    nouns.append(chunk_text)

        # Deduplicate while preserving order
        seen = set()
        unique_nouns = []
        for n in nouns:
            n_lower = n.lower()
            if n_lower not in seen and len(n) >= 2:
                seen.add(n_lower)
                unique_nouns.append(n)
        return unique_nouns

    except ImportError:
        # Fallback: simple noun extraction via regex and heuristics
        words = []
        for text in texts:
            # Extract capitalized words/phrases as potential named entities
            capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
            words.extend(capitalized)
            # Extract common noun patterns
            common_nouns = re.findall(r'\b([a-z]{3,}(?:ing|tion|ment|ence|ity|ism|ology|graphy|ology)?)\b', text)
            words.extend(common_nouns)
        seen = set()
        unique_words = []
        for w in words:
            w_lower = w.lower()
            if w_lower not in seen and len(w) >= 2:
                seen.add(w_lower)
                unique_words.append(w)
        return unique_words


def _affinity_propagation_clustering(
    nouns: list[str],
    embeddings: Optional[dict[str, list[float]]] = None,
    preference: Optional[float] = None,
    damping: float = 0.9,
    max_iter: int = 200,
    min_cluster_size: int = 3,
) -> list[list[str]]:
    """Cluster nouns using Affinity Propagation with optional embeddings.

    If embeddings are not provided, falls back to TF-IDF + cosine similarity.

    Args:
        nouns: List of noun phrases to cluster
        embeddings: Dict mapping noun → embedding vector (optional)
        preference: AP preference value (median of similarities if None)
        damping: Damping factor for AP (0.5-1.0)
        max_iter: Maximum iterations
        min_cluster_size: Minimum nouns per cluster (smaller clusters discarded)

    Returns:
        List of clusters, each cluster is a list of noun phrases
    """
    from sklearn.cluster import AffinityPropagation
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    if len(nouns) < 5:
        return [nouns] if nouns else []

    # Build similarity matrix
    if embeddings:
        # Use provided embeddings
        noun_list = [n for n in nouns if n in embeddings]
        if len(noun_list) < min_cluster_size:
            return [noun_list]
        vectors = np.array([embeddings[n] for n in noun_list])
        similarity_matrix = cosine_similarity(vectors)
    else:
        # Use TF-IDF
        noun_list = nouns
        vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4),
            max_features=1000, stop_words='english',
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(noun_list)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except ValueError:
            # If all nouns are too short, use simple string similarity
            similarity_matrix = np.eye(len(noun_list))
            for i in range(len(noun_list)):
                for j in range(i + 1, len(noun_list)):
                    # Simple Jaccard on character trigrams
                    def trigrams(s):
                        s = '  ' + s.lower() + '  '
                        return {s[k:k+3] for k in range(len(s)-2)}
                    ti = trigrams(noun_list[i])
                    tj = trigrams(noun_list[j])
                    if ti or tj:
                        sim = len(ti & tj) / len(ti | tj)
                    else:
                        sim = 0
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim

    # Run Affinity Propagation
    af = AffinityPropagation(
        damping=damping,
        max_iter=max_iter,
        convergence_iter=30,
        preference=preference,  # None = median of similarities
        random_state=42,
    )
    af.fit(similarity_matrix)

    # Group nouns by cluster
    labels = af.labels_
    clusters: dict[int, list[str]] = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:  # -1 means noise point
            clusters[int(label)].append(noun_list[i])

    # Filter small clusters
    result = [sorted(cluster) for cluster in clusters.values()
              if len(cluster) >= min_cluster_size]

    # Sort clusters by size (largest first)
    result.sort(key=len, reverse=True)

    return result


def _name_clusters_with_llm(
    clusters: list[list[str]],
    text_samples: str,
    agent: Agent,
    max_clusters_per_call: int = 15,
) -> list[dict[str, Any]]:
    """Use LLM to name clusters and generate entity type definitions.

    Returns list of {"name": str, "description": str, "members": list[str]}.
    """
    if not clusters:
        return []

    results = []
    for batch_idx in range(0, len(clusters), max_clusters_per_call):
        batch = clusters[batch_idx:batch_idx + max_clusters_per_call]

        cluster_text = ""
        for i, cluster in enumerate(batch):
            members = ", ".join(cluster[:20])  # Show up to 20 members
            if len(cluster) > 20:
                members += f", ... (+{len(cluster) - 20} more)"
            cluster_text += f"## Cluster {batch_idx + i + 1}\nMembers: {members}\n\n"

        prompt = f"""请为以下语义聚类命名合适的本体类型。

## 文本上下文（用于理解领域）
{text_samples[:2000]}

## 语义聚类
{cluster_text}

请为每个聚类输出:
1. name: 本体类型名称（使用领域术语，英文）
2. description: 简要描述该类型
3. parent: 父类型名称（如果适用）

输出严格的 JSON:
{{"entity_types": [
  {{"cluster_id": 1, "name": "...", "description": "...", "parent": null}},
  ...
]}}"""

        try:
            result = agent.run_structured(prompt, {
                "type": "object",
                "properties": {
                    "entity_types": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cluster_id": {"type": "integer"},
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "parent": {"type": "string"},
                            },
                        },
                    },
                },
            })

            if result:
                for et_data in result.get("entity_types", []):
                    cid = et_data.get("cluster_id", 0) - 1  # Convert to 0-based
                    if 0 <= cid < len(batch):
                        et_data["members"] = batch[cid]
                        results.append(et_data)
        except Exception:
            continue

    return results


def _discover_relations_from_types(
    entity_types: list[dict[str, Any]],
    text_samples: str,
    agent: Agent,
) -> list[dict[str, Any]]:
    """Use LLM to discover relation types between ontology entity types.

    Analyzes entity type pairs and text context to propose relation types.
    """
    et_names = [et["name"] for et in entity_types]
    et_desc = "\n".join(f"- {et['name']}: {et.get('description', '')}"
                        for et in entity_types)

    prompt = f"""请基于以下实体类型和文本上下文，推断这些实体类型之间可能存在的关系类型。

## 实体类型
{et_desc}

## 文本上下文
{text_samples[:3000]}

## 要求
1. 为每对有语义关联的实体类型定义关系
2. 关系必须有文本证据支持（在上下文中有体现）
3. 指定关系的 domain（源实体类型）和 range（目标实体类型）
4. 包含关系的描述和逆向关系（如果存在）

输出严格的 JSON:
{{"relation_types": [
  {{"name": "关系名称", "description": "描述", "domain": "源类型", "range": "目标类型", "inverse": "逆向关系名"}}
]}}"""

    try:
        result = agent.run_structured(prompt, {
            "type": "object",
            "properties": {
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
                        },
                    },
                },
            },
        })
        return result.get("relation_types", []) if result else []
    except Exception:
        return []


# ─── Builder class ─────────────────────────────────────────────────────────

class OntologyBuilder:
    """Multi-paradigm ontology builder.

    Usage:
        builder = OntologyBuilder(llm_config, memory)
        ontology = builder.build(documents, mode="affinity-clustering")
    """

    def __init__(self, llm_config: LLMConfig, memory=None):
        self.llm_config = llm_config
        self.memory = memory
        if self.memory is None:
            from .memory import Memory
            self.memory = Memory(work_dir=".kgclaw_onto_builder")
        self.log = None
        from .logger import get_logger
        self.log = get_logger()

    def _create_agent(self, name: str, system_prompt: str, max_tokens: int = 16384) -> Agent:
        """Create a lightweight agent for ontology building tasks."""
        cfg = AgentConfig(
            name=name,
            system_prompt=system_prompt,
            tools=[],
            max_tool_calls=1,
            model_config=LLMConfig(
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                api_base=self.llm_config.api_base,
                max_tokens=max_tokens,
            ),
        )
        return Agent(cfg, self.memory, self.llm_config)

    def build(
        self,
        documents: list[Document],
        mode: str = "auto",
        existing_ontology: Optional[Ontology] = None,
    ) -> Optional[Ontology]:
        """Build ontology from documents using the specified mode.

        Args:
            documents: List of Document objects
            mode: One of "auto", "text-to-ontology", "relation-to-ontology",
                  "ht-relation-to-ontology", "affinity-clustering"
            existing_ontology: Optional existing ontology to refine

        Returns:
            Ontology object or None if building fails
        """
        if not documents:
            return None

        # Extract text content
        texts = [d.text for d in documents]
        all_text = "\n\n".join(texts)
        samples = "\n\n".join([t[:3000] for t in texts[:5]])

        # Auto-select mode based on data characteristics
        if mode == "auto":
            mode = self._auto_select_mode(documents)
            if self.log:
                self.log.info(f"Auto-selected ontology mode: {mode}")

        result = None
        if mode == "dense-ontology":
            result = self._build_dense(documents, samples, all_text, existing_ontology)
        elif mode == "text-to-ontology":
            result = self._build_to(documents, samples, all_text, existing_ontology)
        elif mode == "relation-to-ontology":
            result = self._build_ro(documents, samples, all_text, existing_ontology)
        elif mode == "ht-relation-to-ontology":
            result = self._build_htro(documents, samples, all_text, existing_ontology)
        elif mode == "affinity-clustering":
            result = self._build_affinity(documents, samples, all_text, existing_ontology)
        else:
            if self.log:
                self.log.warning(f"Unknown ontology mode '{mode}', falling back to text-to-ontology")
            result = self._build_to(documents, samples, all_text, existing_ontology)

        return result

    def _auto_select_mode(self, documents: list[Document]) -> str:
        """Auto-select the best ontology building mode.

        Heuristics:
        - Short single doc → text-to-ontology (simple, fast)
        - Many short docs with entities → affinity-clustering (noun-rich)
        - Long narrative texts → ht-relation-to-ontology (precise)
        - Mixed/long texts → relation-to-ontology (balanced)
        - Default → text-to-ontology
        """
        if not documents:
            return "text-to-ontology"

        non_empty = [d for d in documents if d.text.strip()]
        if not non_empty:
            return "text-to-ontology"

        total_chars = sum(len(d.text) for d in non_empty)
        avg_chars = total_chars / len(non_empty)

        # Check for noun density (proxy for affinity clustering suitability)
        noun_density = 0
        try:
            nouns = _extract_nouns_spacy([d.text[:5000] for d in non_empty[:3]])
            noun_density = len(nouns) / max(1, total_chars / 100)
        except Exception:
            pass

        if len(non_empty) == 1 and avg_chars < 5000:
            return "text-to-ontology"
        elif noun_density > 5 and len(non_empty) >= 3:
            return "affinity-clustering"
        elif avg_chars > 10000 and len(non_empty) <= 3:
            return "ht-relation-to-ontology"
        elif avg_chars > 3000:
            return "relation-to-ontology"
        return "text-to-ontology"

    # ── D-O: Dense Ontology (Graph F1 optimized) ───────────────────────────

    def _build_dense(
        self,
        documents: list[Document],
        samples: str,
        all_text: str,
        existing: Optional[Ontology] = None,
    ) -> Optional[Ontology]:
        """Dense Ontology mode: maximizes Graph F1 with rich hierarchy + cross-relations.

        Three-stage approach:
        1. Extract ALL candidate concepts/types from text (maximize recall)
        2. Organize into 3-4 level hierarchy with parent fields
        3. Create dense cross-type relation network
        """
        from .prompts.system_prompts import SYSTEM_PROMPT_ONTOLOGY_ANALYZER

        agent = self._create_agent("dense_builder", SYSTEM_PROMPT_ONTOLOGY_ANALYZER, max_tokens=16384)

        # Stage 1: Maximal type extraction
        stage1_prompt = f"""Extract EVERY possible concept category and entity type from these texts. Be exhaustive.

## Text
{samples[:6000]}

## Instructions
- List 30-50 distinct entity types / concept categories
- Include both broad categories AND specific sub-types
- Use simple 1-3 word names
- Group related types together

Output JSON: {{"types": [{{"name": "...", "subtypes": ["..."]}}]}}"""

        try:
            stage1 = agent.run_structured(stage1_prompt, {
                "type": "object",
                "properties": {
                    "types": {"type": "array", "items": {"type": "object", "properties": {
                        "name": {"type": "string"}, "subtypes": {"type": "array", "items": {"type": "string"}}
                    }}},
                },
            })
        except Exception:
            stage1 = None

        # Stage 2: Dense hierarchy + relations
        types_context = json.dumps(stage1, ensure_ascii=False)[:3000] if stage1 else samples[:3000]

        stage2_prompt = f"""Build a COMPLETE, DENSE ontology with rich hierarchy and many relations.

## Candidate Types
{types_context}

## Requirements (CRITICAL for evaluation):
1. **20-40 entity_types**: each with name, description, and parent field
2. **80%+ must have parent**: build 3-4 level deep hierarchy
3. **8-15 relation_types**: each with domain AND range
4. **Dense connections**: connect different branches (not just is-a)
5. **Root types**: 4-6 top-level types, each with 3-8 subtypes

Output comprehensive JSON with entity_types and relation_types."""

        try:
            result = agent.run_structured(stage2_prompt, {
                "type": "object",
                "properties": {
                    "ontology_name": {"type": "string"},
                    "entity_types": {"type": "array", "items": {"type": "object", "properties": {
                        "name": {"type": "string"}, "description": {"type": "string"}, "parent": {"type": "string"},
                    }}},
                    "relation_types": {"type": "array", "items": {"type": "object", "properties": {
                        "name": {"type": "string"}, "description": {"type": "string"},
                        "domain": {"type": "string"}, "range": {"type": "string"},
                    }}},
                },
            })

            if not result or not result.get("entity_types"):
                return None

            return self._result_to_ontology(result, "DO")
        except Exception as e:
            if self.log:
                self.log.error(f"Dense build failed: {e}")
            return None

    # ── T-O: Text-to-Ontology ──────────────────────────────────────────────

    def _build_to(
        self,
        documents: list[Document],
        samples: str,
        all_text: str,
        existing: Optional[Ontology] = None,
    ) -> Optional[Ontology]:
        """T-O paradigm: Feed full text to LLM, generate ontology in one step.

        This is KGClaw's existing approach, now extracted as a mode.
        """
        from .prompts.system_prompts import SYSTEM_PROMPT_ONTOLOGY_ANALYZER

        agent = self._create_agent("to_builder", SYSTEM_PROMPT_ONTOLOGY_ANALYZER)

        if existing and existing.raw_definition:
            prompt = f"""请基于以下文档内容，优化和完善现有的本体定义。

## 现有本体
{existing.raw_definition[:2000]}

## 文档内容
{samples[:6000]}

## 要求
1. 识别文档中可能遗漏的实体类型和关系类型
2. 补充缺失的类型定义（特别是 parent 字段）
3. 确保 entity_types 中的 parent 字段正确反映层次关系

输出 JSON 格式。"""
        else:
            # Enhanced T-O prompt: detect if input is type list or raw text
            is_type_list = bool(re.search(r'(?:^|\n)\s*-\s+\S+', samples[:2000]) or
                               (samples.count('\n') > 20 and len(samples.split('\n')[0]) < 100))
            if is_type_list:
                prompt = f"""你是一个本体构建专家。请从以下类型/术语列表中构建完整的本体定义。

## 关键任务
1. **发现类型**：识别列表中的概念，将它们组织成语义类型
2. **构建层次**：使用 parent 字段建立 is-a 层次。分析命名模式推断父子关系
3. **推断关系**：发现类型间的语义关系

## 数据
{samples[:8000]}

输出 JSON 格式（entity_types + relation_types + parent 字段）。"""
            else:
                prompt = f"""你是一个本体构建专家。请从以下文档内容中发现和组织知识结构，构建完整的本体定义。

## 文档内容
{samples[:6000]}

## 关键要求
1. 识别核心概念类型（至少5个），为每个类型填写 parent 字段
2. 发现类型间的语义关系（至少2个），包括 domain/range
3. 类型命名使用简洁的术语

输出 JSON 格式（entity_types + relation_types + parent）。"""

        try:
            result = agent.run_structured(prompt, {
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
            })

            if not result:
                return None

            return self._result_to_ontology(result, "TO")
        except Exception as e:
            if self.log:
                self.log.error(f"T-O build failed: {e}")
            return None

    # ── R-O: Relation-to-Ontology ──────────────────────────────────────────

    def _build_ro(
        self,
        documents: list[Document],
        samples: str,
        all_text: str,
        existing: Optional[Ontology] = None,
    ) -> Optional[Ontology]:
        """R-O paradigm: First identify relations, then build ontology from them.

        Two-stage process:
        1. Extract all candidate relations from text
        2. Cluster and abstract relations into ontology relation types,
           deriving entity types from domain/range analysis
        """
        agent = self._create_agent("ro_builder", SYSTEM_PROMPT_RO_BUILDER)

        # Stage 1: Extract candidate relations from text
        stage1_prompt = f"""## Stage 1: 关系发现 (Relation Discovery)

请从以下文本中识别所有实体之间存在的语义关系。

### 文本
{samples[:6000]}

### 要求
1. 识别文本中的所有实体（如人物、组织、地点、概念等）
2. 找出每对实体之间存在的语义关系
3. 用自然语言描述每个关系
4. 不要使用预定义的关系类型——从具体关系中归纳

输出 JSON:
{{"relations": [
  {{"subject": "实体A", "relation": "关系描述（自然语言短语）", "object": "实体B", "evidence": "文本证据"}}
]}}"""

        try:
            stage1_result = agent.run_structured(stage1_prompt, {
                "type": "object",
                "properties": {
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "relation": {"type": "string"},
                                "object": {"type": "string"},
                                "evidence": {"type": "string"},
                            },
                        },
                    },
                },
            })
        except Exception:
            stage1_result = None

        if not stage1_result:
            return self._build_to(documents, samples, all_text, existing)

        # Handle case where LLM returns a list instead of dict
        if isinstance(stage1_result, list):
            raw_relations = stage1_result
        else:
            raw_relations = stage1_result.get("relations", stage1_result.get("data", []))
        if len(raw_relations) < 2:
            # Too few relations; fall back to T-O
            return self._build_to(documents, samples, all_text, existing)

        # Stage 2: Cluster relations and build ontology
        subjects = list(set(r["subject"] for r in raw_relations))
        objects = list(set(r["object"] for r in raw_relations))
        all_entities = list(set(subjects + objects))
        relation_phrases = list(set(r["relation"] for r in raw_relations))

        # Extract entity type candidates from entity names
        entity_summary = "\n".join(f"- {e}" for e in all_entities[:100])
        relation_summary = "\n".join(f"- {r}" for r in relation_phrases[:50])

        stage2_prompt = f"""## Stage 2: 本体归纳 (Ontology Induction)

基于 Stage 1 发现的实体和关系，归纳出结构化的本体定义。

### 发现的实体 (前100个)
{entity_summary[:1500]}

### 发现的关系短语 (前50个)
{relation_summary[:1500]}

### 关键要求
1. 从实体中归纳类型，**每个类型必须填写 parent 字段**建立层次
2. 从关系短语中归纳关系类型，指定 domain 和 range
3. 分析命名模式推断层次（如 "X Y" → parent: "Y"）

输出 JSON 格式的本体定义（包含 entity_types 的 parent 字段和 relation_types）。"""

        try:
            stage2_result = agent.run_structured(stage2_prompt, {
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
                            },
                        },
                    },
                },
            })

            if not stage2_result:
                return None

            return self._result_to_ontology(stage2_result, "RO")
        except Exception as e:
            if self.log:
                self.log.error(f"R-O build failed: {e}")
            return None

    # ── HT-R-O: Head-Tail-Relation-to-Ontology ─────────────────────────────

    def _build_htro(
        self,
        documents: list[Document],
        samples: str,
        all_text: str,
        existing: Optional[Ontology] = None,
    ) -> Optional[Ontology]:
        """HT-R-O paradigm: Hierarchy-first ontology building with iterative refinement.

        Optimized single-stage approach (based on SchemaOrg Edge-F1=0.4828 success):
        1. Feed all type/text information to LLM with hierarchy-focused prompt
        2. LLM organizes concepts into a hierarchy with parent fields
        3. LLM infers relations between types

        Key improvements over original:
        - Single-stage (faster, avoids error propagation from Stage 1)
        - Uses the enhanced SYSTEM_PROMPT_ONTOLOGY_ANALYZER
        - Iterative refinement: if first pass has < 3 entity types, retry with different approach
        """
        from .prompts.system_prompts import SYSTEM_PROMPT_ONTOLOGY_ANALYZER

        # Strategy detection: is input a type list or raw text?
        # Type lists typically contain many short names separated by newlines
        type_list_pattern = bool(re.search(r'(?:^|\n)\s*-\s+\S+', samples[:2000]) or
                                (samples.count('\n') > 20 and len(samples.split('\n')[0]) < 100))

        agent = self._create_agent("htro_builder", SYSTEM_PROMPT_ONTOLOGY_ANALYZER)

        if type_list_pattern:
            # Input is a type/term list — ask LLM to organize into hierarchy
            prompt = f"""你是一个本体构建专家。以下是领域 "{documents[0].metadata.get('domain', 'unknown') if documents else 'unknown'}" 中的概念/类型列表及其层次关系示例。

## 任务
分析这些概念，发现它们的层次结构（is-a 关系）并构建完整的本体定义。

## 关键要求
1. **识别顶层类型**：找出 3-8 个最顶层的概念类型（如 Thing, Disease, Organization 等）
2. **构建层次**：对每个子类型，使用 `parent` 字段指向其父类型。例如 "lung cancer" 的 parent 是 "cancer"
3. **推断关系**：除了 is-a 层次，发现类型之间的语义关系（如 causes, located_in, part_of）
4. **类型命名**：直接使用输入中的术语作为类型名，不要编造新词

## 输入数据
{samples[:10000]}

## 输出
完整的本体定义 JSON（包含 entity_types 和 relation_types，每个 entity_type 必须有 parent 字段）。"""
        else:
            # Input is raw text — extract concepts first, then organize
            prompt = f"""你是一个本体构建专家。请从以下文本中发现和组织知识结构。

## 任务
1. 从文本中识别所有核心概念和实体类别
2. 将这些概念组织成层次结构（使用 parent 字段）
3. 发现概念之间的关系类型

## 文本内容
{samples[:8000]}

## 输出
完整的本体定义 JSON（entity_types + relation_types，每个 entity_type 必须有 parent 字段）。"""

        # First pass
        try:
            result = agent.run_structured(prompt, {
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
                            },
                        },
                    },
                },
            })
        except Exception:
            result = None

        # Retry with different approach if first pass produced too few types
        if result and len(result.get("entity_types", [])) < 3 and type_list_pattern:
            if self.log:
                self.log.info("HT-R-O: first pass found <3 types, retrying with broader prompt")
            broad_prompt = f"""请从以下概念列表中，识别所有可能的本体类型（不要遗漏！），并构建完整的层次结构。

## 重要提示
- 至少识别 15 个以上的类型（越多越好）
- 每个类型必须有 parent 字段指向其父类型
- 如果概念是另一个概念的子类型（如 "A" 是 "B" 的一种），用 parent 表示

## 概念列表
{samples[:8000]}

## 层次模式提示
分析概念名称中的模式来推断层次：
- "X Y" 通常是 "Y" 的子类型（如 "lung cancer" → parent: "cancer"）
- 高频出现的词通常是顶层类型
- 寻找共享相同后缀/前缀的概念组

输出完整本体 JSON（包含大量 entity_types + relation_types）。"""
            try:
                result = agent.run_structured(broad_prompt, {
                    "type": "object",
                    "properties": {
                        "ontology_name": {"type": "string"},
                        "entity_types": {"type": "array", "items": {"type": "object", "properties": {
                            "name": {"type": "string"}, "description": {"type": "string"}, "parent": {"type": "string"},
                        }}},
                        "relation_types": {"type": "array", "items": {"type": "object", "properties": {
                            "name": {"type": "string"}, "description": {"type": "string"},
                            "domain": {"type": "string"}, "range": {"type": "string"},
                        }}},
                    },
                })
            except Exception:
                pass

        if not result or not result.get("entity_types"):
            return None

        return self._result_to_ontology(result, "HTRO")

    # ── Affinity Clustering ────────────────────────────────────────────────

    def _build_affinity(
        self,
        documents: list[Document],
        samples: str,
        all_text: str,
        existing: Optional[Ontology] = None,
    ) -> Optional[Ontology]:
        """Affinity Clustering paradigm from LLM4Onto.

        1. Extract nouns/noun phrases from text using spaCy
        2. Cluster nouns using Affinity Propagation
        3. Use LLM to name clusters as entity types
        4. Multi-round merging of similar clusters
        5. Infer relation types between discovered entity types
        """
        agent = self._create_agent("ap_builder", SYSTEM_PROMPT_AP_BUILDER)

        # Step 1: Extract nouns
        texts = [d.text for d in documents]
        nouns = _extract_nouns_spacy(texts)
        if self.log:
            self.log.info(f"Affinity Clustering: extracted {len(nouns)} nouns")

        if len(nouns) < 10:
            if self.log:
                self.log.warning("Too few nouns extracted, falling back to T-O")
            return self._build_to(documents, samples, all_text, existing)

        # Step 2: Affinity Propagation clustering
        clusters = _affinity_propagation_clustering(
            nouns,
            min_cluster_size=max(2, int(len(nouns) * 0.01)),  # Dynamic threshold
        )
        if self.log:
            self.log.info(f"Affinity Clustering: {len(clusters)} clusters formed")

        if len(clusters) < 2:
            return self._build_to(documents, samples, all_text, existing)

        # Limit clusters for LLM processing
        clusters = clusters[:30]  # Top 30 largest clusters

        # Step 3: LLM names the clusters
        named_clusters = _name_clusters_with_llm(clusters, samples, agent)
        if not named_clusters:
            return self._build_to(documents, samples, all_text, existing)

        # Step 4: Multi-round merging (up to 3 rounds)
        merge_agent = self._create_agent("merge_clusters", SYSTEM_PROMPT_MERGE_CLUSTERS)

        for round_num in range(3):
            if len(named_clusters) <= 2:
                break

            # Build pairs of adjacent clusters for merge consideration
            merge_candidates = []
            for i in range(len(named_clusters)):
                for j in range(i + 1, min(len(named_clusters), i + 5)):
                    ci = named_clusters[i]
                    cj = named_clusters[j]
                    merge_candidates.append({
                        "cluster_a": ci["name"],
                        "cluster_b": cj["name"],
                        "members_a": ", ".join(ci.get("members", [])[:10]),
                        "members_b": ", ".join(cj.get("members", [])[:10]),
                    })

            if not merge_candidates:
                break

            merge_prompt = f"""## 聚类合并判断

以下是语义相近的聚类对，判断它们是否应该合并：

{json.dumps(merge_candidates[:10], ensure_ascii=False, indent=2)[:4000]}

输出 JSON: {{"merges": [...]}}"""

            try:
                merge_result = merge_agent.run_structured(merge_prompt, {
                    "type": "object",
                    "properties": {
                        "merges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "cluster_a": {"type": "string"},
                                    "cluster_b": {"type": "string"},
                                    "should_merge": {"type": "boolean"},
                                    "merged_name": {"type": "string"},
                                    "reason": {"type": "string"},
                                },
                            },
                        },
                    },
                })

                if merge_result:
                    merges = merge_result.get("merges", [])
                    to_merge = [m for m in merges if m.get("should_merge")]
                    if not to_merge:
                        break

                    # Apply merges
                    merged_names = set()
                    for m in to_merge:
                        a_name = m["cluster_a"]
                        b_name = m["cluster_b"]

                        # Find and merge clusters
                        a_cluster = next((c for c in named_clusters if c["name"] == a_name), None)
                        b_cluster = next((c for c in named_clusters if c["name"] == b_name), None)

                        if a_cluster and b_cluster:
                            merged = {
                                "name": m.get("merged_name", a_name),
                                "description": f"Combined: {a_cluster.get('description', '')}; {b_cluster.get('description', '')}",
                                "members": list(set(a_cluster.get("members", []) + b_cluster.get("members", []))),
                            }
                            named_clusters = [c for c in named_clusters
                                            if c["name"] not in (a_name, b_name)]
                            named_clusters.append(merged)
                            merged_names.add(a_name)
                            merged_names.add(b_name)

                    if self.log:
                        self.log.info(f"Round {round_num + 1}: merged {len(to_merge)} cluster pairs, "
                                     f"{len(named_clusters)} clusters remaining")
            except Exception:
                break

        if self.log:
            self.log.info(f"Final clusters after merging: {len(named_clusters)}")

        # Step 5: Infer relation types
        relation_types_data = _discover_relations_from_types(named_clusters, samples, agent)

        # Build the ontology
        entity_types = []
        for nc in named_clusters:
            entity_types.append(EntityType(
                name=nc.get("name", "Unknown"),
                description=nc.get("description", ""),
                parent=nc.get("parent"),
            ))

        relation_types = []
        for rt_data in relation_types_data:
            relation_types.append(RelationType(
                name=rt_data.get("name", ""),
                description=rt_data.get("description", ""),
                domain=rt_data.get("domain"),
                range=rt_data.get("range"),
                inverse=rt_data.get("inverse"),
            ))

        if not entity_types:
            return None

        # Ensure at least some relations
        if not relation_types:
            # Generate default "related_to" relations
            for i, et1 in enumerate(entity_types):
                for et2 in entity_types[i+1:i+4]:
                    relation_types.append(RelationType(
                        name="related_to",
                        description=f"Relation between {et1.name} and {et2.name}",
                        domain=et1.name,
                        range=et2.name,
                    ))

        raw_def_parts = [
            "# Auto-discovered Ontology (Affinity Clustering)",
            "",
            "## Entity Types",
        ]
        for et in entity_types:
            raw_def_parts.append(f"- **{et.name}**: {et.description}")
        raw_def_parts.append("")
        raw_def_parts.append("## Relation Types")
        for rt in relation_types:
            d = f" (from `{rt.domain}`)" if rt.domain else ""
            r = f" (to `{rt.range}`)" if rt.range else ""
            raw_def_parts.append(f"- **{rt.name}**{d}{r}: {rt.description}")

        return Ontology(
            name=f"auto_discovered_ac_{len(entity_types)}_types",
            description=f"Automatically discovered via Affinity Clustering from {len(documents)} documents",
            entity_types=entity_types,
            relation_types=relation_types,
            raw_definition="\n".join(raw_def_parts),
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _normalize_type_name(self, name: str) -> str:
        """Normalize a type name to standard format."""
        name = name.strip()
        # Remove leading/trailing markers
        name = re.sub(r'^[\d\s\.\-\*\#]+', '', name)
        name = re.sub(r'[\d\s\.\-\*\#]+$', '', name)
        # Normalize spaces and capitalization
        name = ' '.join(name.split())  # Normalize whitespace
        # If it's all lowercase and 1-2 words, capitalize first letter of each word
        if name == name.lower() and len(name.split()) <= 3:
            name = ' '.join(w[0].upper() + w[1:] if len(w) > 1 else w.upper() for w in name.split())
        return name or "Unknown"

    def _result_to_ontology(self, result: dict[str, Any], paradigm: str) -> Ontology:
        """Convert LLM result dict to Ontology object with name normalization."""
        raw_entity_types = result.get("entity_types", [])
        # Deduplicate and normalize names
        seen_names = set()
        entity_types = []
        for et in raw_entity_types:
            name = self._normalize_type_name(et.get("name", ""))
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            entity_types.append(EntityType(
                name=name,
                description=et.get("description", ""),
                parent=et.get("parent"),
            ))

        # Validate parent references: only keep parent if it exists in the entity type set
        et_names = {et.name for et in entity_types}
        et_names_lower = {n.lower(): n for n in et_names}
        for et in entity_types:
            if et.parent and et.parent not in et_names:
                parent_lower = et.parent.lower()
                match = et_names_lower.get(parent_lower)
                if match:
                    et.parent = match
                else:
                    et.parent = None  # Invalid parent, clear it

        # Implicit parent inference: for types without parents, try to assign one
        # based on naming patterns (e.g., "Lung Cancer" → parent "Cancer" if "Cancer" exists)
        without_parent = [et for et in entity_types if not et.parent]
        for et in without_parent:
            name_lower = et.name.lower()
            # Strategy 1: Check if any existing type is a suffix of this type name
            for other_name in sorted(et_names, key=len, reverse=True):
                other_lower = other_name.lower()
                if other_lower != name_lower and name_lower.endswith(" " + other_lower):
                    et.parent = other_name
                    break
            # Strategy 2: Check if last word of name is an existing type
            if not et.parent and " " in et.name:
                last_word = et.name.rsplit(" ", 1)[-1]
                match = et_names_lower.get(last_word.lower())
                if match and match != et.name:
                    et.parent = match

        relation_types = [
            RelationType(
                name=rt.get("name", ""),
                description=rt.get("description", ""),
                domain=rt.get("domain"),
                range=rt.get("range"),
                inverse=rt.get("inverse"),
            )
            for rt in result.get("relation_types", [])
        ]

        raw_def_parts = [
            f"# Ontology: {result.get('ontology_name', 'auto_discovered')}",
            f"Paradigm: {paradigm}",
            "",
            "## Entity Types",
        ]
        for et in entity_types:
            p = f" (subtype of: {et.parent})" if et.parent else ""
            raw_def_parts.append(f"  - **{et.name}**{p}: {et.description}")
        raw_def_parts.append("")
        raw_def_parts.append("## Relation Types")
        for rt in relation_types:
            d = f" from `{rt.domain}`" if rt.domain else ""
            r = f" to `{rt.range}`" if rt.range else ""
            raw_def_parts.append(f"  - **{rt.name}**{d}{r}: {rt.description}")

        return Ontology(
            name=result.get("ontology_name", f"auto_discovered_{paradigm.lower()}"),
            description=result.get("description", result.get("extraction_guide", "")),
            entity_types=entity_types,
            relation_types=relation_types,
            raw_definition="\n".join(raw_def_parts),
        )
