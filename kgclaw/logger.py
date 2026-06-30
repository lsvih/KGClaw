"""
Structured logging for KGClaw with optional trace capture.

Writes logs to .kgclaw/logs/kgclaw.log (rotating, max 10MB × 3 files).
- Normal mode:  INFO+ to file, INFO+ to console (rich output)
- Debug mode:   DEBUG+ to file, INFO+ to console
- Trace mode:   additionally writes full LLM interactions to .kgclaw/traces/
- Quiet mode:   WARNING+ to console (like old default)
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional


# ─── Try importing Rich for nicer console output ─────────────────────────────

try:
    from rich.console import Console
    from rich.text import Text as RichText
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# ─── Console singleton (shared with ui/display.py) ────────────────────────────

if _RICH_AVAILABLE:
    _console = Console(stderr=True, highlight=False)
else:
    _console = None


class KGClawLogger:
    """Singleton logger for KGClaw.

    Three output channels:
    1. **File** — rotating log files (.kgclaw/logs/kgclaw.log)
    2. **Console** — rich-formatted status messages (stderr)
    3. **Trace** — structured JSONL file (.kgclaw/traces/build_*.jsonl)
    """

    _instance: Optional["KGClawLogger"] = None
    _initialized = False

    # Rich style constants
    C_DIM = "dim"
    C_INFO = "bright_black"
    C_OK = "green"
    C_WARN = "yellow"
    C_ERROR = "red"
    C_PHASE = "cyan"
    C_AGENT = "magenta"
    C_TOOL = "blue"
    C_TOKEN = "bright_cyan"
    C_TIME = "bright_black"
    C_COUNT = "bold white"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if KGClawLogger._initialized:
            return
        KGClawLogger._initialized = True

        self._logger = logging.getLogger("kgclaw")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        self._log_dir: Optional[Path] = None
        self._debug_mode = False
        self._quiet_mode = False
        self._file_handler: Optional[RotatingFileHandler] = None
        self._console_handler: Optional[logging.StreamHandler] = None
        self._workflow_start_time: Optional[float] = None

        # Trace writer (lazy-initialized when trace is enabled)
        self._tracer: Any = None  # TraceWriter

    # ── Setup ────────────────────────────────────────────────────────────────

    def setup(
        self,
        work_dir: str = ".kgclaw",
        debug: bool = False,
        trace: bool = False,
        quiet: bool = False,
    ):
        """Initialize logging with the given work directory and flags.

        Args:
            work_dir: Root work directory (.kgclaw)
            debug: Enable DEBUG-level file logging + full prompt/response
            trace: Enable structured JSONL trace file (.kgclaw/traces/)
            quiet: Suppress console output to WARNING+ only
        """
        self._debug_mode = debug
        self._quiet_mode = quiet
        self._log_dir = Path(work_dir) / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # ── File handler ──────────────────────────────────────────────────
        for h in list(self._logger.handlers):
            if hasattr(h, 'baseFilename'):
                self._logger.removeHandler(h)

        log_file = self._log_dir / "kgclaw.log"
        self._file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        self._file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        self._file_handler.setFormatter(_create_formatter(detailed=True))
        self._logger.addHandler(self._file_handler)

        # ── Console handler ────────────────────────────────────────────────
        console_level = logging.WARNING if quiet else logging.INFO
        self._console_handler = logging.StreamHandler(sys.stderr)
        self._console_handler.setLevel(console_level)
        self._console_handler.setFormatter(_create_formatter(detailed=False))
        self._logger.addHandler(self._console_handler)

        # ── Trace writer ───────────────────────────────────────────────────
        if trace:
            self._enable_trace(work_dir)

        self.info("KGClaw logger initialized", extra={
            "log_dir": str(self._log_dir),
            "debug": debug,
            "trace": trace,
            "quiet": quiet,
        })

    def _enable_trace(self, work_dir: str):
        """Lazily import and initialize the trace writer."""
        try:
            from .tracer import TraceWriter
            self._tracer = TraceWriter(work_dir=work_dir)
        except Exception:
            self._tracer = None

    @property
    def debug_mode(self) -> bool:
        return self._debug_mode

    @property
    def trace_enabled(self) -> bool:
        return self._tracer is not None and self._tracer.enabled

    @property
    def log_dir(self) -> Optional[Path]:
        return self._log_dir

    # ── Standard logging methods ───────────────────────────────────────────

    def debug(self, msg: str, **extra):
        self._logger.debug(msg, extra={"extra_data": extra} if extra else {})

    def info(self, msg: str, **extra):
        self._logger.info(msg, extra={"extra_data": extra} if extra else {})

    def warning(self, msg: str, **extra):
        self._logger.warning(msg, extra={"extra_data": extra} if extra else {})

    def error(self, msg: str, **extra):
        self._logger.error(msg, extra={"extra_data": extra} if extra else {})

    # ── Rich console helpers ───────────────────────────────────────────────

    def console(self, msg: str, style: str = "", icon: str = ""):
        """Write a Rich-formatted line to stderr (if Rich is available).

        Used sparingly for important one-off messages.  The main console
        output should still go through the progress callback / harness events.
        """
        if _console is None:
            return
        if icon:
            _console.print(f"  {icon} {msg}", style=style, highlight=False)
        else:
            _console.print(f"  {msg}", style=style, highlight=False)

    # ── Domain-specific logging ────────────────────────────────────────────

    def workflow_start(self, ontology_len: int, doc_count: int, skills: list[str], model: str):
        self._workflow_start_time = time.time()
        self.info(
            "Workflow started",
            ontology_chars=ontology_len,
            doc_count=doc_count,
            skills=skills,
            model=model,
        )
        # Rich console
        self.console(f"工作流启动 · 模型 [bold]{model}[/bold] · {doc_count} 文档",
                     style=self.C_PHASE)
        # Trace
        if self.trace_enabled:
            self._tracer.workflow("start", {
                "doc_count": doc_count,
                "model": model,
                "skills": skills,
            })

    def workflow_end(self, entities: int, relations: int, triples: int):
        elapsed = time.time() - (self._workflow_start_time or time.time())
        self.info(
            "Workflow completed",
            entities=entities,
            relations=relations,
            triples=triples,
            elapsed_seconds=round(elapsed, 1),
        )
        self.console(
            f"工作流完成 · {entities} 实体 · {relations} 关系 · {triples} 三元组 · "
            f"[{self.C_DIM}]{elapsed:.1f}s[/]",
            style=self.C_OK,
        )
        if self.trace_enabled:
            self._tracer.workflow("end", {
                "entities": entities,
                "relations": relations,
                "triples": triples,
                "elapsed_s": round(elapsed, 1),
            })

    def workflow_error(self, error: str):
        self.error("Workflow failed", error=error)

    def phase_start(self, phase: str):
        self.info(f"Phase start: {phase}")
        self.console(f"[{self.C_PHASE}]▸[/] {phase}", style=self.C_PHASE)
        if self.trace_enabled:
            self._tracer.phase(phase, "start")

    def phase_end(self, phase: str, **stats):
        parts = "  ".join(f"[{self.C_COUNT}]{v}[/] {k}" for k, v in stats.items())
        self.info(f"Phase complete: {phase}", **stats)
        self.console(f"[{self.C_OK}]✓[/] {phase}  {parts}", style=self.C_OK)
        if self.trace_enabled:
            self._tracer.phase(phase, "complete", stats)

    def phase_error(self, phase: str, error: str):
        self.error(f"Phase failed: {phase}", error=error)
        self.console(f"[{self.C_ERROR}]✗[/] {phase}: {error}", style=self.C_ERROR)
        if self.trace_enabled:
            self._tracer.phase(phase, "failed", {"error": error})

    def agent_call(self, agent_name: str, prompt_size: int, **kwargs):
        """Log an LLM agent call."""
        level = logging.DEBUG if self._debug_mode else logging.INFO
        self._logger.log(
            level,
            f"Agent call: {agent_name}",
            extra={"extra_data": {"agent": agent_name, "prompt_size": prompt_size, **kwargs}},
        )
        extras = f"  [{self.C_DIM}]{prompt_size:,} chars[/]" if prompt_size else ""
        self.console(
            f"[{self.C_AGENT}]@{agent_name}[/]{extras}",
            style=self.C_DIM,
        )

    def agent_result(self, agent_name: str, duration: float, **kwargs):
        """Log an LLM agent result."""
        level = logging.DEBUG if self._debug_mode else logging.INFO
        self._logger.log(
            level,
            f"Agent result: {agent_name} ({duration:.1f}s)",
            extra={"extra_data": {"agent": agent_name, "duration": round(duration, 1), **kwargs}},
        )
        # Build a compact result line
        parts = []
        if "entities" in kwargs:
            parts.append(f"entities={kwargs['entities']}")
        if "relations" in kwargs:
            parts.append(f"relations={kwargs['relations']}")
        self.console(
            f"  [{self.C_OK}]✓[/] @{agent_name} [{self.C_TIME}]{duration:.1f}s[/]"
            + (f"  {'  '.join(parts)}" if parts else ""),
            style=self.C_DIM,
        )

    def tool_call(self, agent: str, tool: str, args: dict):
        self.debug(f"Tool call: {tool}", agent=agent, tool=tool, args=str(args)[:500])
        if self.trace_enabled:
            self._tracer.tool_call(agent, tool, args)

    def tool_result(self, agent: str, tool: str, success: bool, result_preview: str = ""):
        if success:
            self.debug(f"Tool result: {tool} [OK]", agent=agent, result=result_preview[:200])
        else:
            self.warning(f"Tool result: {tool} [FAIL]", agent=agent, error=result_preview[:200])
        if self.trace_enabled:
            self._tracer.tool_result(agent, tool, success,
                                     data=result_preview if success else None,
                                     error=result_preview if not success else "")

    def llm_prompt(self, agent: str, prompt: str):
        """Log the full prompt sent to LLM (debug + trace only)."""
        if self._debug_mode:
            self.debug(f"LLM prompt for {agent}", prompt=prompt[:5000])
        if self.trace_enabled:
            # Already captured in trace's llm_request — no need to duplicate
            pass

    def llm_response(self, agent: str, response: str):
        """Log the full LLM response (debug + trace only)."""
        if self._debug_mode:
            self.debug(f"LLM response from {agent}", response=response[:5000])

    # ── Trace-specific ─────────────────────────────────────────────────────

    def trace_llm_request(self, agent: str, model: str, prompt: str,
                          tools: Optional[list] = None):
        """Record a full LLM request in the trace file."""
        if self.trace_enabled:
            self._tracer.llm_request(agent, model, prompt, tools)

    def trace_llm_response(self, agent: str, prompt_tokens: int, completion_tokens: int,
                           content: str, tool_calls: Optional[list] = None,
                           finish_reason: str = ""):
        """Record a full LLM response in the trace file."""
        if self.trace_enabled:
            self._tracer.llm_response(
                agent, prompt_tokens, completion_tokens,
                content, tool_calls, finish_reason,
            )

    def trace_start(self, workflow_id: str = "") -> Optional[Path]:
        """Start the trace file. Returns the file path."""
        if self._tracer is None:
            self._enable_trace(".kgclaw")
        if self._tracer:
            self.console(f"跟踪日志已启用 → {self._tracer._work_dir / 'traces'}",
                        style=self.C_DIM)
            return self._tracer.start(workflow_id)
        return None

    def trace_close(self):
        """Close the trace file."""
        if self._tracer:
            self._tracer.close()

    # ─── Token logging (rich console) ─────────────────────────────────────

    def token_usage(self, prompt_tokens: int, completion_tokens: int):
        """Log accumulated token usage (for rich console display)."""
        self.console(
            f"[{self.C_TOKEN}]↓{prompt_tokens:,}[/] [{self.C_DIM}]↑{completion_tokens:,} tokens[/]",
            style=self.C_DIM,
        )

    def chunk_progress(self, current: int, total: int, **kwargs):
        self.debug(f"Chunk progress: {current}/{total}", **kwargs)


# ─── Helper ──────────────────────────────────────────────────────────────────

def _create_formatter(detailed: bool = False) -> logging.Formatter:
    if detailed:
        fmt = "%(asctime)s [%(levelname)-7s] %(message)s  %(extra_data)s"
    else:
        fmt = "%(asctime)s [%(levelname)-7s] %(message)s"
    return _ExtraFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")


class _ExtraFormatter(logging.Formatter):
    """Custom formatter that renders extra_data dict nicely."""

    def format(self, record):
        if not hasattr(record, "extra_data") or not record.extra_data:
            record.extra_data = ""
        elif isinstance(record.extra_data, dict):
            parts = []
            for k, v in record.extra_data.items():
                if isinstance(v, str) and len(v) > 100:
                    v = v[:100] + "..."
                parts.append(f"{k}={v}")
            record.extra_data = "  " + " ".join(parts)
        return super().format(record)


# ─── Global accessor ─────────────────────────────────────────────────────────

def get_logger() -> KGClawLogger:
    """Get the global KGClaw logger instance."""
    return KGClawLogger()


def setup_logging(work_dir: str = ".kgclaw", debug: bool = False,
                  trace: bool = False, quiet: bool = False):
    """Initialize global logging."""
    get_logger().setup(work_dir=work_dir, debug=debug, trace=trace, quiet=quiet)
