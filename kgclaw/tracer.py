"""
Structured trace writer for KGClaw.

Writes every LLM interaction (full prompt + response), tool call, and
phase transition to a JSONL file under .kgclaw/traces/ for later
inspection and debugging.

Usage:
    tracer = TraceWriter(work_dir=".kgclaw")
    tracer.start("workflow_abc123")
    tracer.llm_request("entity_extractor_0", "gpt-4o", prompt_text, tool_schemas)
    tracer.llm_response("entity_extractor_0", prompt_tokens, completion_tokens, response_text)
    tracer.tool_call("quality_checker", "validate_against_ontology", {"entities": [...]})
    tracer.tool_result("quality_checker", "validate_against_ontology", True, result_data)
    tracer.phase("entity_extraction", "complete", {"entities": 150, "duration_ms": 4500})
    tracer.close()
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class TraceWriter:
    """Thread-safe structured trace writer.

    Each event is one JSON line in a .jsonl file.  The file is flushed after
    every write so that crash-recovery inspection is possible.
    """

    def __init__(self, work_dir: str = ".kgclaw"):
        self._work_dir = Path(work_dir)
        self._file = None
        self._lock = threading.Lock()
        self._event_count = 0
        self._start_time: Optional[float] = None
        self._closed = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._file is not None and not self._closed

    def start(self, workflow_id: str = "") -> Path:
        """Open the trace file and write a header event.  Returns the file path."""
        trace_dir = self._work_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        wid = workflow_id[:12] if workflow_id else "unknown"
        filename = f"build_{ts}_{wid}.jsonl"
        self._file = (trace_dir / filename).open("w", encoding="utf-8")
        self._start_time = time.time()
        self._event_count = 0
        self._closed = False

        self._write({
            "ts": _now(),
            "event": "trace_start",
            "workflow_id": workflow_id,
            "trace_file": str(self._file.name),
        })
        return trace_dir / filename

    def close(self):
        """Flush and close the trace file."""
        with self._lock:
            if self._file and not self._closed:
                elapsed = time.time() - (self._start_time or time.time())
                self._write({
                    "ts": _now(),
                    "event": "trace_end",
                    "total_events": self._event_count,
                    "elapsed_s": round(elapsed, 1),
                })
                self._file.close()
                self._closed = True

    # ── Event helpers ──────────────────────────────────────────────────────

    def llm_request(
        self,
        agent: str,
        model: str,
        prompt: str,
        tools: Optional[list[dict]] = None,
    ):
        self.event("llm_request", {
            "agent": agent,
            "model": model,
            "prompt_chars": len(prompt),
            "prompt": prompt,
            "tool_schemas": [t.get("function", {}).get("name", "?") for t in (tools or [])],
            "num_tools": len(tools or []),
        })

    def llm_response(
        self,
        agent: str,
        prompt_tokens: int,
        completion_tokens: int,
        content: str,
        tool_calls: Optional[list[dict]] = None,
        finish_reason: str = "",
    ):
        self.event("llm_response", {
            "agent": agent,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "content_chars": len(content),
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        })

    def tool_call(self, agent: str, tool: str, args: dict):
        self.event("tool_call", {
            "agent": agent,
            "tool": tool,
            "args": _safe_serialize(args),
        })

    def tool_result(self, agent: str, tool: str, success: bool, data: Any = None, error: str = ""):
        self.event("tool_result", {
            "agent": agent,
            "tool": tool,
            "success": success,
            "data": _safe_serialize(data) if success else None,
            "error": error if not success else None,
        })

    def phase(self, name: str, status: str, meta: Optional[dict] = None):
        self.event("phase", {
            "phase": name,
            "status": status,  # "start", "complete", "failed"
            "meta": meta or {},
        })

    def workflow(self, event_type: str, data: Optional[dict] = None):
        self.event(f"workflow_{event_type}", data or {})

    # ── Generic event ──────────────────────────────────────────────────────

    def event(self, event_type: str, data: dict):
        """Write a generic trace event."""
        payload = {
            "ts": _now(),
            "event": event_type,
        }
        payload.update(data)
        self._write(payload)

    # ── Internal ───────────────────────────────────────────────────────────

    def _write(self, obj: dict):
        with self._lock:
            if self._file and not self._closed:
                self._file.write(json.dumps(obj, ensure_ascii=False) + "\n")
                self._file.flush()
                self._event_count += 1


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
           f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


def _safe_serialize(obj: Any, max_chars: int = 50000) -> Any:
    """Return a safe-to-serialize version of obj, truncating large strings."""
    if isinstance(obj, str):
        if len(obj) > max_chars:
            return obj[:max_chars] + f"\n...[truncated at {max_chars} chars]..."
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, (int, float, bool, type(None))):
        return obj
    return str(obj)[:max_chars]
