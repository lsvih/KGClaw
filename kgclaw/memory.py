"""
Session Memory system for KGClaw.

Manages conversation context, workflow state persistence,
and progressive context loading across agent interactions.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .models import ExtractionResult, Message, Ontology, PhaseResult, Role, WorkflowState
from urllib.parse import quote


def _nt_escape_uri(name: str) -> str:
    """Percent-encode entity names for use in N-Triples URIs."""
    # Encode characters unsafe in URIs: spaces, <, >, #, %, etc.
    return quote(name, safe='')

def _nt_escape_literal(s: str) -> str:
    r"""Escape a string for N-Triples literal (quoted string).

    Per W3C N-Triples spec: must escape \, ", \n, \r, \t.
    """
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\n', '\\n')
    s = s.replace('\r', '\\r')
    s = s.replace('\t', '\\t')
    return s


class Memory:
    """
    Session-level memory that persists workflow state to disk.

    Responsibilities:
    - Store and retrieve conversation messages per agent
    - Persist workflow state (ontology, phases, results)
    - Manage context window: compact old messages when approaching limits
    - Support progressive disclosure (load relevant context only)
    """

    def __init__(self, work_dir: str = ".kgclaw"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._messages: dict[str, list[Message]] = {}  # agent_id → messages
        self._workflow: Optional[WorkflowState] = None
        self._context: dict[str, Any] = {}  # arbitrary key-value context store
        self._loaded = False
        self._msg_lock = threading.Lock()  # protects _messages from concurrent access

    # ── Workflow State ──────────────────────────────────────────────────────

    @property
    def workflow(self) -> Optional[WorkflowState]:
        return self._workflow

    def init_workflow(self, ontology: Optional[Ontology] = None) -> WorkflowState:
        """Initialize a new workflow session (preserving existing state if any)."""
        if self._workflow is None:
            self._workflow = WorkflowState(ontology=ontology)
        elif ontology is not None and self._workflow.ontology is None:
            # Preserve existing workflow but fill in missing ontology
            self._workflow.ontology = ontology
        self._save_workflow()
        return self._workflow

    def load_workflow(self) -> Optional[WorkflowState]:
        """Load workflow state from disk, returning None on corrupt/missing state."""
        state_file = self.work_dir / "workflow_state.json"
        if not state_file.exists():
            return None
        try:
            data = json.loads(state_file.read_text())
            self._workflow = WorkflowState(**data)
            self._loaded = True
            return self._workflow
        except (json.JSONDecodeError, IOError, ValueError, TypeError) as e:
            import logging
            logging.getLogger("kgclaw").warning(
                f"Failed to load workflow state from {state_file}: {e}"
            )
            return None

    def save_workflow(self):
        """Persist current workflow state to disk."""
        if self._workflow:
            self._save_workflow()

    def _save_workflow(self):
        state_file = self.work_dir / "workflow_state.json"
        state_file.write_text(self._workflow.model_dump_json(indent=2))

    # ── Ontology Export ────────────────────────────────────────────────────

    def export_ontology(self, output_path: Optional[str] = None) -> tuple[Path, Path]:
        """Export the ontology as standalone JSON and Markdown files.

        Returns:
            Tuple of (json_path, md_path).
        """
        json_path = Path(output_path) if output_path else (self.work_dir / "ontology.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)

        if self._workflow and self._workflow.ontology:
            onto = self._workflow.ontology
            json_path.write_text(
                onto.model_dump_json(indent=2, exclude_none=True),
            )

            # Also write human-readable markdown
            md_path = json_path.with_suffix(".md")
            md_lines = [
                f"# Ontology: {onto.name}",
                "",
                onto.description or "(no description)",
                "",
                "## Entity Types",
                "",
            ]
            for et in onto.entity_types:
                md_lines.append(f"- **{et.name}**: {et.description or ''}")
            md_lines.append("")
            md_lines.append("## Relation Types")
            md_lines.append("")
            for rt in onto.relation_types:
                domain = f"`{rt.domain}`" if rt.domain else "any"
                range_ = f"`{rt.range}`" if rt.range else "any"
                md_lines.append(f"- **{rt.name}** ({domain} → {range_}): {rt.description or ''}")
            md_lines.append("")
            if onto.raw_definition:
                md_lines.extend([
                    "## Raw Definition",
                    "",
                    "```",
                    onto.raw_definition,
                    "```",
                    "",
                ])
            md_path.write_text("\n".join(md_lines))
            return json_path, md_path
        else:
            # Write empty placeholders
            json_path.write_text("{}")
            md_path = json_path.with_suffix(".md")
            md_path.write_text("# No ontology defined\n")
            return json_path, md_path

    # ── Generated Code Persistence ─────────────────────────────────────────

    def save_generated_code(self, name: str, code: str) -> Path:
        """Save LLM-generated code to the work directory.

        Args:
            name: Descriptive filename (e.g. 'extract_doc_0_20260624T142530.py').
            code: The generated code text.

        Returns:
            Path to the saved file.
        """
        gen_dir = self.work_dir / "generated_code"
        gen_dir.mkdir(parents=True, exist_ok=True)
        file_path = gen_dir / name
        file_path.write_text(code)
        return file_path

    # ── Document Manifest (file change detection) ──────────────────────────

    def save_document_manifest(self, docs: list) -> None:
        """Save document manifest for future change detection.

        Records MD5 hash, mtime, and size for each source file.
        Stored as {work_dir}/document_manifest.json.
        """
        manifest = {}
        for doc in docs:
            if not doc.source:
                continue
            manifest[doc.source] = {
                "content_hash": doc.metadata.get("content_hash", ""),
                "mtime": doc.metadata.get("file_mtime", 0),
                "size": doc.metadata.get("file_size", 0),
                "last_processed_at": datetime.now().isoformat(),
            }
        manifest_file = self.work_dir / "document_manifest.json"
        manifest_file.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
        )

    def load_document_manifest(self) -> dict:
        """Load document manifest from disk.

        Returns:
            Dict mapping filepath -> {content_hash, mtime, size, last_processed_at}.
            Empty dict if no manifest exists.
        """
        manifest_file = self.work_dir / "document_manifest.json"
        if not manifest_file.exists():
            return {}
        try:
            return json.loads(manifest_file.read_text())
        except (json.JSONDecodeError, IOError):
            return {}

    def detect_file_changes(self, current_paths: list[str]) -> dict:
        """Compare current files against stored manifest to detect changes.

        Computes MD5 hash for each current file and compares against
        the stored manifest.

        Args:
            current_paths: List of file paths currently present on disk.

        Returns:
            {
                "unchanged": [path, ...],
                "added": [path, ...],
                "modified": [path, ...],
                "deleted": [path, ...],
            }
        """
        import hashlib

        manifest = self.load_document_manifest()

        unchanged = []
        added = []
        modified = []
        deleted = []

        current_set = set()
        for path_str in current_paths:
            p = Path(path_str)
            if not p.exists() or not p.is_file():
                continue
            current_set.add(path_str)

            try:
                content = p.read_bytes()
                current_hash = hashlib.md5(content, usedforsecurity=False).hexdigest()
            except (IOError, OSError):
                # Can't read file — treat as unavailable, skip
                continue

            if path_str not in manifest:
                added.append(path_str)
            elif manifest[path_str].get("content_hash") != current_hash:
                modified.append(path_str)
            else:
                unchanged.append(path_str)

        # Deleted: in manifest but not on disk
        for path_str in manifest:
            if path_str not in current_set:
                deleted.append(path_str)

        return {
            "unchanged": unchanged,
            "added": added,
            "modified": modified,
            "deleted": deleted,
        }

    # ── Document Management ────────────────────────────────────────────────

    def remove_document(self, source: str) -> bool:
        """Remove a single document from the workflow by its source path.

        Returns True if a document was removed, False if not found.
        """
        if not self._workflow:
            return False
        initial = len(self._workflow.documents)
        self._workflow.documents = [
            d for d in self._workflow.documents if d.source != source
        ]
        if len(self._workflow.documents) < initial:
            self._save_workflow()
            return True
        return False

    def remove_documents(self, sources: list[str]) -> int:
        """Remove multiple documents by their source paths.

        Returns the count of documents actually removed.
        """
        if not self._workflow or not sources:
            return 0
        source_set = set(sources)
        initial = len(self._workflow.documents)
        self._workflow.documents = [
            d for d in self._workflow.documents if d.source not in source_set
        ]
        removed = initial - len(self._workflow.documents)
        if removed > 0:
            self._save_workflow()
        return removed

    def clear_documents(self) -> int:
        """Remove all documents from the workflow.

        Returns the count of documents removed (0 if no workflow exists).
        """
        if not self._workflow:
            return 0
        count = len(self._workflow.documents)
        if count > 0:
            self._workflow.documents = []
            self._save_workflow()
        return count

    def get_loaded_documents(self) -> list[dict]:
        """Return a summary list of all loaded documents.

        Each dict: source, filename, chars, ext, size, is_tabular.
        """
        if not self._workflow:
            return []
        summaries = []
        for doc in self._workflow.documents:
            meta = doc.metadata or {}
            src = doc.source
            summaries.append({
                "source": src,
                "filename": meta.get("filename", Path(src).name if src else ""),
                "chars": len(doc.text),
                "ext": meta.get("ext", Path(src).suffix if src else ""),
                "size": meta.get("file_size", meta.get("size", 0)),
                "is_tabular": meta.get("is_tabular", False),
            })
        return summaries

    def has_document(self, source: str) -> bool:
        """Check if a document with the given source path is loaded."""
        if not self._workflow:
            return False
        return any(d.source == source for d in self._workflow.documents)

    def add_phase_result(self, phase: PhaseResult):
        """Add or update a phase result in the workflow."""
        if not self._workflow:
            return
        # Replace if exists, otherwise append
        for i, p in enumerate(self._workflow.phases):
            if p.phase_name == phase.phase_name:
                self._workflow.phases[i] = phase
                self._save_workflow()
                return
        self._workflow.phases.append(phase)
        self._save_workflow()

    def set_final_result(self, result: ExtractionResult, output_nt: Optional[str] = None):
        """Set the final extraction result."""
        if self._workflow:
            self._workflow.final_result = result
            self._workflow.output_nt = output_nt
            self._workflow.completed_at = datetime.now()
            self._save_workflow()

    # ── Conversation Messages ────────────────────────────────────────────────

    def get_messages(self, agent_id: str = "default") -> list[Message]:
        """Get messages for an agent (thread-safe)."""
        with self._msg_lock:
            return list(self._messages.get(agent_id, []))

    def add_message(self, agent_id: str, message: Message):
        """Add a message to an agent's conversation (thread-safe)."""
        with self._msg_lock:
            if agent_id not in self._messages:
                self._messages[agent_id] = []
            self._messages[agent_id].append(message)

    def add_messages(self, agent_id: str, messages: list[Message]):
        """Add multiple messages at once (thread-safe)."""
        with self._msg_lock:
            if agent_id not in self._messages:
                self._messages[agent_id] = []
            self._messages[agent_id].extend(messages)

    def clear_messages(self, agent_id: str = "default"):
        """Clear messages for an agent (thread-safe)."""
        with self._msg_lock:
            self._messages[agent_id] = []

    def compact_messages(
        self,
        agent_id: str = "default",
        max_messages: int = 50,
        keep_system: bool = True,
    ) -> list[Message]:
        """
        Compact conversation by summarizing old messages.

        Strategy: Keep the system message + last N messages.
        When messages exceed max_messages, summarize the middle portion
        and insert as a single system-like message.

        Thread-safe: acquires _msg_lock for both read and write.
        """
        with self._msg_lock:
            msgs = self._messages.get(agent_id, [])
            if len(msgs) <= max_messages:
                return list(msgs)

            system_msgs = [m for m in msgs if m.role.value == "system"] if keep_system else []
            non_system = [m for m in msgs if m.role.value != "system"]

            # Keep last (max_messages - len(system_msgs) - 1) non-system messages
            keep_count = max_messages - len(system_msgs) - 1
            if keep_count < 1:
                keep_count = 1

            # Summarize removed messages
            removed = non_system[:-keep_count]
            kept = non_system[-keep_count:]

            summary_lines = ["[Context Summary - earlier conversation]", ""]
            for m in removed[:10]:  # take first 10 for summary
                if m.role.value in ("user", "assistant"):
                    preview = m.content[:200].replace("\n", " ")
                    summary_lines.append(f"- [{m.role.value}] {preview}...")
            summary_lines.append(f"\n[{len(removed)} total earlier messages summarized]")

            summary_msg = Message(
                role=Role.SYSTEM,
                content="\n".join(summary_lines),
            )

            compacted = system_msgs + [summary_msg] + kept
            self._messages[agent_id] = compacted
            return compacted

    # ── Context Store ────────────────────────────────────────────────────────

    def set_context(self, key: str, value: Any):
        """Store arbitrary context data."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve context data."""
        return self._context.get(key, default)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_messages_to_disk(self, agent_id: str = "default"):
        """Persist agent messages to disk."""
        msg_file = self.work_dir / f"messages_{agent_id}.json"
        data = [m.model_dump() for m in self._messages.get(agent_id, [])]
        msg_file.write_text(json.dumps(data, indent=2, default=str))

    def load_messages_from_disk(self, agent_id: str = "default") -> list[Message]:
        """Load agent messages from disk."""
        msg_file = self.work_dir / f"messages_{agent_id}.json"
        if msg_file.exists():
            data = json.loads(msg_file.read_text())
            messages = [Message(**m) for m in data]
            self._messages[agent_id] = messages
            return messages
        return []

    # ── Output ───────────────────────────────────────────────────────────────

    def get_ontology_context(self) -> str:
        """Get the ontology as a string for inclusion in prompts."""
        if not self._workflow or not self._workflow.ontology:
            return "No ontology defined."
        return self._workflow.ontology.to_extraction_guide()

    def get_progress_summary(self) -> str:
        """Get a human-readable summary of workflow progress."""
        if not self._workflow:
            return "No workflow active."

        lines = [f"Workflow: {self._workflow.workflow_id}"]
        lines.append(f"Documents: {len(self._workflow.documents)}")
        lines.append(f"Phases: {len(self._workflow.phases)}")
        for phase in self._workflow.phases:
            status_icon = {
                "pending": "⏳",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌",
                "skipped": "⏭️",
            }.get(phase.status.value, "?")
            lines.append(f"  {status_icon} {phase.phase_name}: {phase.status.value}")
        if self._workflow.final_result:
            lines.append(
                f"Result: {len(self._workflow.final_result.entities)} entities, "
                f"{len(self._workflow.final_result.triples)} triples"
            )
        return "\n".join(lines)

    def export_final_nt(self, output_path: Optional[str] = None) -> str:
        """Export the final knowledge graph as N-Triples format.

        Always writes a file (even if empty) so the user knows output was generated.
        """
        if not self._workflow:
            out_path = Path(output_path) if output_path else (self.work_dir / "output.nt")
            out_path.write_text("# No workflow data\n")
            return ""

        lines = []
        seen_entities = set()
        seen_triples = set()

        # Step 1: 为所有实体写入 rdf:type + 名称 + 属性三元组（即使没有关系也能独立存在）
        for entity in self._workflow.all_entities:
            e_key = (entity.name, entity.type)
            if e_key in seen_entities:
                continue
            seen_entities.add(e_key)

            safe_uri_name = _nt_escape_uri(entity.name)
            safe_uri_type = _nt_escape_uri(entity.type)
            safe_literal_name = _nt_escape_literal(entity.name)
            lines.append(f'<{safe_uri_type}/{safe_uri_name}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{safe_uri_type}> .')
            lines.append(f'<{safe_uri_type}/{safe_uri_name}> <名称> "{safe_literal_name}"^^<http://www.w3.org/2001/XMLSchema#string> .')

            # 属性三元组
            for attr_name, attr_val in (entity.attributes or {}).items():
                if attr_val is not None:
                    safe_val = _nt_escape_literal(str(attr_val))
                    lines.append(
                        f'<{safe_uri_type}/{safe_uri_name}> '
                        f'<{_nt_escape_uri(attr_name)}> '
                        f'"{safe_val}"^^<http://www.w3.org/2001/XMLSchema#string> .'
                    )
            lines.append("")

        # Step 2: 写入关系三元组
        for triple in self._workflow.all_triples:
            t_key = (triple.subject.name, triple.subject.type,
                     triple.predicate,
                     triple.object.name, triple.object.type)
            if t_key in seen_triples:
                continue
            seen_triples.add(t_key)

            rel_line = (
                f'<{_nt_escape_uri(triple.subject.type)}/{_nt_escape_uri(triple.subject.name)}> '
                f'<{_nt_escape_uri(triple.predicate)}> '
                f'<{_nt_escape_uri(triple.object.type)}/{_nt_escape_uri(triple.object.name)}> .'
            )
            lines.append(rel_line)

        output = "\n".join(lines)

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(output)
        else:
            out_path = self.work_dir / "output.nt"
            out_path.write_text(output)

        return output
