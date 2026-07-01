"""
Helper methods for the Harness class.

Contains text chunking, entity name normalization, fuzzy matching,
fuzzy deduplication, and skill-agent factory methods.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import difflib
import re as _re
from typing import Any, Optional

from ..models import Entity


class _HarnessHelpers:
    """Mixin providing chunking, normalization, fuzzy-matching, and agent factory helpers."""

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize entity name: remove punctuation, whitespace, lowercase. Used for fuzzy matching."""
        if not name:
            return ""
        normalized = _re.sub(r'[\s，,。．.、；;：:！!？?""\'\'「」『』【】（）()\[\]{}《》<>]', '', name)
        return normalized.lower()

    def _fuzzy_match_entity(
        self,
        name: str,
        entity_index: dict[str, list[Entity]],
        norm_index: dict[str, list[tuple[str, Entity]]] = None,
    ) -> tuple[Optional[Entity], float, str]:
        """Fuzzy-match an entity name, returning (entity, confidence_penalty, match_method).

        Matching strategy (tried in order):
        1. exact match → penalty=0
        2. normalized exact match (strip punct/space/lowercase) → penalty=0.05
        3. difflib similarity > 0.85 → penalty=0.1
        4. substring containment → penalty=0.15
        5. all fail → (None, 0, "no_match")
        """
        if not name or not entity_index:
            return (None, 0, "no_match")

        name_stripped = name.strip()
        # Level 1: exact match
        if name_stripped in entity_index:
            return (entity_index[name_stripped][0], 0, "exact")

        # Level 2: normalized exact match
        name_norm = self._normalize_name(name_stripped)
        if norm_index and name_norm in norm_index:
            return (norm_index[name_norm][0][1], 0.05, "normalized")

        # Build normalized index on-the-fly if not provided
        if not norm_index:
            norm_index = {}
            for ename, elist in entity_index.items():
                nkey = self._normalize_name(ename)
                if nkey not in norm_index:
                    norm_index[nkey] = []
                for e in elist:
                    norm_index[nkey].append((ename, e))
            if name_norm in norm_index:
                return (norm_index[name_norm][0][1], 0.05, "normalized")

        # Level 3: difflib similarity
        all_names = list(entity_index.keys())
        best_ratio = 0
        best_name = None
        for ename in all_names:
            ratio = difflib.SequenceMatcher(None, name_norm, self._normalize_name(ename)).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_name = ename
        if best_ratio > 0.85 and best_name:
            return (entity_index[best_name][0], 0.1, f"fuzzy({best_ratio:.2f})")

        # Level 4: substring containment
        name_lower = name_stripped.lower()
        for ename in all_names:
            ename_lower = ename.lower()
            if len(name_lower) >= 2 and len(ename_lower) >= 2:
                if name_lower in ename_lower or ename_lower in name_lower:
                    return (entity_index[ename][0], 0.15, "substring")

        # Level 5: token overlap (Jaccard ≥ 0.5)
        name_tokens = set(name_norm.split())
        for ename in all_names:
            ename_tokens = set(self._normalize_name(ename).split())
            if name_tokens and ename_tokens:
                overlap = len(name_tokens & ename_tokens) / min(len(name_tokens), len(ename_tokens))
                if overlap >= 0.5:
                    return (entity_index[ename][0], 0.18, f"token_overlap({overlap:.2f})")

        # Level 6: character overlap for short strings (Korean, abbreviations, etc.)
        if len(name_norm) <= 15:
            name_chars = set(name_norm)
            for ename in all_names:
                ename_norm = self._normalize_name(ename)
                if len(ename_norm) <= 15:
                    ename_chars = set(ename_norm)
                    if name_chars and ename_chars:
                        char_overlap = len(name_chars & ename_chars) / max(len(name_chars), len(ename_chars))
                        if char_overlap >= 0.7:
                            return (entity_index[ename][0], 0.20, f"char_overlap({char_overlap:.2f})")

        return (None, 0, "no_match")

    @staticmethod
    def entity_match(a: str, b: str) -> bool:
        """Fuzzy entity name matching: token overlap ≥ 0.5, substring, or edit distance ≤ 3."""
        if not a or not b:
            return False
        na = _HarnessHelpers._normalize_name(a)
        nb = _HarnessHelpers._normalize_name(b)
        if not na or not nb:
            return False
        if na == nb:
            return True
        if len(na) >= 2 and na in nb:
            return True
        if len(nb) >= 2 and nb in na:
            return True
        toks_a, toks_b = set(na.split()), set(nb.split())
        if toks_a and toks_b:
            overlap = len(toks_a & toks_b) / min(len(toks_a), len(toks_b))
            if overlap >= 0.5:
                return True
        if len(na) <= 20 and len(nb) <= 20:
            from ..utils import levenshtein_distance
            try:
                dist = levenshtein_distance(na, nb)
                if dist <= 3:
                    return True
            except ImportError:
                pass
            # simple inline edit distance
            m, n = len(na), len(nb)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1,
                                   dp[i-1][j-1] + (0 if na[i-1] == nb[j-1] else 1))
            if dp[m][n] <= 3:
                return True
        return False

    def _fuzzy_dedup_entities(self, entities: list[Entity]) -> list[Entity]:
        """Fuzzy deduplicate entities by merging similar names within the same type.

        Uses difflib similarity > 0.85 as the merge threshold.
        When merging, keeps the entity with higher confidence and merges attributes.
        """
        if len(entities) <= 1:
            return entities

        # Group entities by type
        by_type: dict[str, list[Entity]] = {}
        for e in entities:
            if e.type not in by_type:
                by_type[e.type] = []
            by_type[e.type].append(e)

        result = []
        for etype, ents in by_type.items():
            if len(ents) <= 1:
                result.extend(ents)
                continue

            merged: set[int] = set()
            for i in range(len(ents)):
                if i in merged:
                    continue
                best = ents[i]
                for j in range(i + 1, len(ents)):
                    if j in merged:
                        continue
                    sim = difflib.SequenceMatcher(
                        None,
                        self._normalize_name(best.name),
                        self._normalize_name(ents[j].name),
                    ).ratio()
                    if sim > 0.85:
                        if ents[j].confidence > best.confidence:
                            best = ents[j]
                        if ents[j].attributes:
                            if not best.attributes:
                                best = best.model_copy()
                                best.attributes = {}
                            for k, v in ents[j].attributes.items():
                                if k not in best.attributes:
                                    best.attributes[k] = v
                        merged.add(j)
                        if hasattr(self, 'log') and self.log:
                            self.log.debug(
                                f"Fuzzy merged: '{ents[j].name}' -> '{best.name}' "
                                f"({etype}, sim={sim:.2f})"
                            )
                result.append(best)

        return result

    def _create_skill_agent(self, name: str, skill) -> "Agent":
        """Create an agent configured with a skill's prompt and tool set.

        Extraction skills (entity_extractor, relation_extractor) don't need tools —
        text is already in the prompt and they can return structured results directly.
        Other agents keep necessary tools.
        """
        from ..agent import Agent, AgentConfig

        tools = skill.get_tool_names()
        max_calls = 1 if not tools else 5
        config = AgentConfig(
            name=name,
            system_prompt=skill.get_system_prompt(),
            tools=tools,
            max_tool_calls=max_calls,
        )
        agent = Agent(config, self.memory, self.llm_config)
        agent.on_event(lambda et, d: self._emit(et, d))
        return agent

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks respecting paragraph/sentence boundaries.

        Prefers paragraph (\\n\\n) boundaries; falls back to sentence boundaries
        for Chinese text. Uses sliding overlap to avoid losing entities across chunk boundaries.
        """
        MAX_CHUNK_SIZE = max(self.config.chunk_size, 1000)  # use config directly
        OVERLAP = max(self.config.chunk_overlap, 100)        # use config directly
        MAX_CHUNKS = self.config.max_chunks

        # Single-line text (e.g. PDF extraction): split by character + overlap
        if '\n' not in text.strip('\n'):
            if len(text) <= MAX_CHUNK_SIZE:
                return [text]
            chunks = []
            step = MAX_CHUNK_SIZE - OVERLAP
            for i in range(0, len(text), step):
                chunks.append(text[i:i + MAX_CHUNK_SIZE])
            return chunks[:MAX_CHUNKS]

        # Split by paragraphs first
        paragraphs = _re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Further split long paragraphs by sentence boundaries
        segments = []
        for para in paragraphs:
            if len(para) <= MAX_CHUNK_SIZE:
                segments.append(para)
            else:
                sentences = _re.split(r'(?<=[。！？.!?])\s*', para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) > MAX_CHUNK_SIZE and current:
                        segments.append(current.strip())
                        current = sent
                    else:
                        current += sent
                if current.strip():
                    segments.append(current.strip())

        # Pack segments into chunks, each close to MAX_CHUNK_SIZE
        chunks = []
        current_chunk = ""
        for seg in segments:
            if len(current_chunk) + len(seg) > MAX_CHUNK_SIZE and current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-OVERLAP:] if len(current_chunk) > OVERLAP else current_chunk
                current_chunk = overlap_text + "\n\n" + seg
            else:
                if current_chunk:
                    current_chunk += "\n\n" + seg
                else:
                    current_chunk = seg

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Ensure chunk count doesn't exceed max
        if len(chunks) > MAX_CHUNKS:
            self.log.warning(f"Chunk count {len(chunks)} exceeds max {MAX_CHUNKS}, merging...")
            while len(chunks) > MAX_CHUNKS:
                best_i = 0
                best_len = float('inf')
                for i in range(len(chunks) - 1):
                    combined = len(chunks[i]) + len(chunks[i+1])
                    if combined < best_len:
                        best_len = combined
                        best_i = i
                chunks[best_i] = chunks[best_i] + "\n\n" + chunks[best_i+1]
                chunks.pop(best_i + 1)

        return chunks if chunks else [text]
