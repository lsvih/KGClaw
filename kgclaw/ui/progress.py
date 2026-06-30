"""
Shared weighted progress callback for CLI and interactive REPL.

Provides a factory function that creates Rich-based progress callbacks
with phase-weighted progress bars, ETA estimation, and token tracking.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import sys
import time
from typing import Any, Callable, Optional

from rich.console import Console

from ..harness import PHASE_WEIGHTS, phase_label
from ..i18n import _

console = Console()


def _fmt_tokens(prompt: int, completion: int) -> str:
    """Format token counts for display."""
    def _k(n: int) -> str:
        if n >= 1000:
            return f"{n/1000:.1f}k"
        return str(n)
    return f"↓ {_k(prompt)} tokens ↑ {_k(completion)} tokens"


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return _("剩余 ~{seconds:.0f}s").format(seconds=seconds)
    elif seconds < 3600:
        return _("剩余 ~{seconds:.1f}min").format(seconds=seconds / 60)
    return _("剩余 ~{seconds:.1f}h").format(seconds=seconds / 3600)


def make_progress_callback(
    verbose_toggle: Any = None,
    *,
    show_chunk_detail: bool = True,
) -> tuple[Callable, Callable]:
    """Create a weighted progress callback with ETA and token tracking.

    Returns (callback, stop_fn). Call stop_fn() when the operation completes.

    Phase weight design (total 100%):
    - auto_discover_ontology: 5%
    - ontology_analysis: 5%
    - entity_extraction: 60% (most time-consuming, subdivided by chunk count)
    - relation_extraction: 20%
    - quality_check: 5%
    - triple_construction: 5%
    """

    def _verbose() -> bool:
        if verbose_toggle is None:
            return False
        return getattr(verbose_toggle, 'value', False)

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    progress = Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, style="cyan", complete_style="green"),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[eta]}[/dim]"),
        console=console,
    )
    main_task = None
    started = False
    total_weight = sum(PHASE_WEIGHTS.values())
    completed_weight = 0.0
    current_phase_weight = 0
    current_phase_chunks = 1
    phase_start_time = 0.0
    overall_start_time = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    active_phase_label = ""  # track current phase for display when bar is capped

    def _clamp_completed(w: float) -> float:
        """Cap progress below 100% until workflow_complete fires.

        Caps at 98% of total to leave a visible gap in the progress bar
        during late phases (quality_check, triple_construction). The gap
        prevents the bar from appearing "done" while work is ongoing.
        Rich's TaskProgressColumn rounds up, so we stay well below 100%.
        """
        return min(w, total_weight * 0.98)

    def _refresh_display():
        """Force-refresh the progress display with live timer and token counts."""
        if main_task is None:
            return
        elapsed = time.time() - overall_start_time
        tok = _fmt_tokens(prompt_tokens, completion_tokens)
        display = f"[dim]{tok} | {active_phase_label}[/dim]" if active_phase_label else tok
        if completed_weight < total_weight * 0.98:
            eta_s = elapsed / max(completed_weight, 0.01) * (total_weight - completed_weight)
            display = f"{_format_eta(eta_s)} | {display}"
        else:
            display = _("处理中...") + f" | {display}"
        progress.update(main_task, completed=_clamp_completed(completed_weight), eta=display)

    def _stop():
        nonlocal started
        if started:
            try:
                progress.stop()
            except Exception:
                pass
            started = False
        _ensure_cursor_visible()
        console.print()

    def cb(event_type: str, data: dict):
        nonlocal main_task, started, completed_weight
        nonlocal current_phase_weight, current_phase_chunks
        nonlocal phase_start_time, overall_start_time
        nonlocal prompt_tokens, completion_tokens
        nonlocal active_phase_label

        if event_type == "workflow_start":
            skills = data.get("skills", [])
            console.print(_("  流水线: {skills}").format(skills="[dim]" + ' -> '.join(skills) + "[/dim]"))
            overall_start_time = time.time()
            main_task = progress.add_task(
                _("[cyan]构建进度"), total=total_weight, eta="..."
            )
            progress.start()
            started = True

        elif event_type == "phase_start":
            phase = data.get("phase", "")
            label = phase_label(phase)
            active_phase_label = label
            phase_start_time = time.time()
            current_phase_weight = PHASE_WEIGHTS.get(phase, 10)
            current_phase_chunks = 1
            console.print(_("  [bold]{label}[/bold] [dim]— 开始...[/dim]").format(label=label))
            _refresh_display()

        elif event_type == "chunk_progress":
            status = data.get("status", "")
            total = data.get("total_chunks", 1)
            phase_name = data.get("phase", "entity_extraction")
            if status == "starting":
                current_phase_chunks = max(total, 1)
                console.print(_("  文本分为 [cyan]{total}[/cyan] 分块并行处理").format(total=total))
            elif status == "done":
                current = data.get("current", 0)
                phase_names = list(PHASE_WEIGHTS.keys())
                prefix_weight = 0.0
                for pn in phase_names:
                    if pn == phase_name:
                        break
                    prefix_weight += PHASE_WEIGHTS[pn]
                phase_weight = PHASE_WEIGHTS.get(phase_name, 60)
                completed_weight = prefix_weight + phase_weight * (current / max(current_phase_chunks, 1))
                _refresh_display()

                if show_chunk_detail:
                    elapsed = data.get("elapsed", 0)
                    is_relation = phase_name == "relation_extraction"
                    new_label = _("关系") if is_relation else _("实体")
                    new_count = data.get('new_relations', 0) if is_relation else data.get('new_entities', 0)
                    total_count = data.get('total_relations', 0) if is_relation else data.get('total_entities', 0)
                    console.print(
                        _("  [dim]分块 [{current}/{total}][/dim] +{new_count} {new_label} (累计 {total_count}) {elapsed:.0f}s").format(
                            current=current, total=total,
                            new_count=new_count, new_label=new_label,
                            total_count=total_count, elapsed=elapsed,
                        )
                    )

        elif event_type == "phase_complete":
            phase = data.get("phase", "")
            label = phase_label(phase)
            info = ""
            if "entities" in data:
                info = _("找到 {n} 实体").format(n=data['entities'])
            elif "relations" in data:
                info = _("识别 {n} 关系").format(n=data['relations'])
            elif "triples" in data:
                info = _("构建 {n} 三元组").format(n=data['triples'])
            elif "entity_types" in data:
                info = _("识别 {et} 实体类型, {rt} 关系类型").format(
                    et=data['entity_types'], rt=data['relation_types'])
            elapsed = time.time() - phase_start_time if phase_start_time > 0 else 0
            console.print(f"  [green][OK][/green] {info} [dim]({elapsed:.0f}s)[/dim]")

            if current_phase_weight > 0:
                phase_names = list(PHASE_WEIGHTS.keys())
                cum = 0.0
                for pn in phase_names:
                    if phase_label(pn) == label:
                        cum += PHASE_WEIGHTS[pn]
                        break
                    cum += PHASE_WEIGHTS[pn]
                if cum > completed_weight:
                    completed_weight = cum
            _refresh_display()

        elif event_type == "subagent_start":
            if _verbose():
                subagent = data.get("subagent", "?")
                preview = data.get("task_preview", "")[:120]
                console.print(_("  [dim]┌─ 子 Agent: [bold]{name}[/bold][/dim]").format(name=subagent))
                console.print(_("  [dim]│ 输入: {preview}...[/dim]").format(preview=preview))
            else:
                console.print(
                    _("  [dim]生成子 Agent: {name}[/dim]").format(name=data.get('subagent', '?')),
                )

        elif event_type == "tool_call":
            if _verbose():
                agent = data.get("agent", "?")
                tool = data.get("tool", "?")
                args = data.get("args", {})
                args_preview = str(args)[:150]
                console.print(f"  [dim]│ [yellow]>> {tool}[/yellow] {args_preview}[/dim]")

        elif event_type == "tool_result":
            if _verbose():
                success = data.get("success", False)
                result_preview = str(data.get("result", ""))[:200]
                icon = "[green]OK[/green]" if success else "[red]FAIL[/red]"
                console.print(f"  [dim]│ {icon} {result_preview}[/dim]")

        elif event_type == "agent_call_start":
            if _verbose():
                agent = data.get("agent", "?")
                chunk_i = data.get("chunk_index", 0)
                prompt_size = data.get("prompt_size", 0)
                thread = data.get("thread", "")
                console.print(
                    _("  [dim]│ [{agent}] LLM 调用开始 (prompt: {size:,} 字符, {thread})...[/dim]").format(
                        agent=agent, size=prompt_size, thread=thread)
                )
            else:
                chunk_i = data.get("chunk_index", 0)
                thread = data.get("thread", "")
                thread_info = f" [{thread}]" if thread else ""
                console.print(_("  [dim]  分块 {chunk} 处理中{thread}...[/dim]").format(
                    chunk=chunk_i + 1, thread=thread_info))
            _refresh_display()

        elif event_type == "agent_call_end":
            if _verbose():
                duration = data.get("duration", 0)
                entities_found = data.get("entities_found", 0)
                console.print(
                    _("  [dim]│ [{agent}] 完成 ({duration:.1f}s, {n} 实体)[/dim]").format(
                        agent=data.get('agent', '?'), duration=duration, n=entities_found)
                )
            _refresh_display()

        elif event_type == "token_usage":
            prompt_tokens = data.get("prompt_tokens", prompt_tokens)
            completion_tokens = data.get("completion_tokens", completion_tokens)
            _refresh_display()

        elif event_type == "workflow_complete":
            if main_task is not None:
                tok = _fmt_tokens(prompt_tokens, completion_tokens)
                progress.update(main_task, completed=total_weight, eta=_("完成!") + f" | {tok}")
            if started:
                progress.stop()
            console.print(_("  [green][OK][/green] 构建完成"))

        elif event_type == "phase_failed":
            console.print(
                _("  [red][FAIL][/red] {phase}: {error}").format(
                    phase=data.get('phase', '?'), error=data.get('error', '?'))
            )

    return cb, _stop


def _ensure_cursor_visible():
    """Restore terminal cursor visibility."""
    try:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
    except Exception:
        pass
