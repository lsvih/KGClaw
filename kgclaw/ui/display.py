"""
Shared display helpers for CLI and interactive REPL output.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

from typing import Any, Optional

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

from ..i18n import _

console = Console()

# ─── Shared styles ──────────────────────────────────────────────────────────

STYLE = {
    "banner": Style(color="bright_cyan", bold=True),
    "primary": Style(color="cyan"),
    "success": Style(color="green"),
    "warning": Style(color="yellow"),
    "error": Style(color="red"),
    "dim": Style(dim=True),
    "muted": Style(color="bright_black"),
    "accent": Style(color="magenta"),
    "heading": Style(color="bright_yellow", bold=True),
    "logo": Style(color="#FF6B6B", bold=True),
}

ICON = {
    "logo": "[kgclaw]",
    "ok": "[OK]",
    "fail": "[FAIL]",
    "warn": "[!]",
    "doc": "[docs]",
    "onto": "[onto]",
    "model": "[AI]",
    "output": "[out]",
    "time": "[time]",
}


def print_banner():
    """Print the KGClaw banner."""
    logo = Text()
    logo.append(f"{ICON['logo']} ", style=STYLE["logo"])
    logo.append("KGClaw", style=STYLE["banner"])
    logo.append(_(" — 知识图谱构建引擎"), style=STYLE["dim"])
    console.print()
    console.print(Align.center(logo))
    console.print(
        Align.center(
            Text(_("基于 AI Agent Harness 的本体驱动知识图谱构建系统"), style=STYLE["muted"])
        )
    )
    console.print("  v0.1.0", style=STYLE["dim"], justify="center")
    console.print()


def print_section(title: str, icon: str = "gear"):
    console.print()
    console.print(f"  {ICON.get(icon, icon)} [bold yellow]{title}[/bold yellow]")


def print_success(msg: str):
    console.print(f"  {ICON['ok']} [green]{msg}[/green]")


def print_error(msg: str):
    console.print(f"  {ICON['fail']} [red]{msg}[/red]")


def print_warning(msg: str):
    console.print(f"  {ICON['warn']} [yellow]{msg}[/yellow]")


def print_ontology_summary(ontology: Any) -> None:
    """Print a one-line summary of the structured ontology."""
    if not ontology or not getattr(ontology, 'entity_types', None):
        console.print(_("  [dim]本体: 未结构化[/dim]"))
        return
    et_names = ", ".join(et.name for et in ontology.entity_types[:5])
    rt_names = ", ".join(rt.name for rt in ontology.relation_types[:5])
    etc = _("…等{n}个").format(n=len(ontology.entity_types)) if len(ontology.entity_types) > 5 else ""
    rtc = _("…等{n}个").format(n=len(ontology.relation_types)) if len(ontology.relation_types) > 5 else ""
    console.print(
        _("  [dim]实体类型:[/dim] {et_names}{etc}  [dim]关系类型:[/dim] {rt_names}{rtc}").format(
            et_names=et_names, etc=etc, rt_names=rt_names, rtc=rtc
        )
    )


def print_ontology(ontology: Any) -> None:
    """Display structured ontology in formatted Rich tables.

    Shows entity types (name, description, parent) and relation types
    (name, description, domain, range, inverse) in separate panels.
    """
    if not ontology:
        return

    console.print()
    console.print(Rule(_("  本体分析结果"), style=STYLE["muted"]))
    console.print()

    # Name and description
    if getattr(ontology, 'name', None) and ontology.name != "unnamed":
        console.print(_("  [bold]本体名称:[/bold] {name}").format(name=ontology.name))
    if getattr(ontology, 'description', None):
        console.print(f"  [dim]{ontology.description}[/dim]")
        console.print()

    # Entity types table
    entity_types = getattr(ontology, 'entity_types', None) or []
    if entity_types:
        et_table = Table(box=box.SIMPLE, show_header=True, border_style="cyan")
        et_table.add_column(_("实体类型"), style="bold cyan", width=16)
        et_table.add_column(_("描述"), style="white")
        et_table.add_column(_("父类型"), style="dim", width=10)
        for et in entity_types:
            et_table.add_row(
                et.name,
                getattr(et, 'description', '') or '-',
                getattr(et, 'parent', None) or '-',
            )
        console.print(Panel(et_table, title=_("实体类型 (Entity Types)"), border_style="cyan"))
        console.print()

    # Relation types table
    relation_types = getattr(ontology, 'relation_types', None) or []
    if relation_types:
        rt_table = Table(box=box.SIMPLE, show_header=True, border_style="yellow")
        rt_table.add_column(_("关系类型"), style="bold yellow", width=16)
        rt_table.add_column(_("描述"), style="white")
        rt_table.add_column("Domain", style="dim", width=12)
        rt_table.add_column("Range", style="dim", width=12)
        rt_table.add_column(_("逆关系"), style="dim", width=10)
        for rt in relation_types:
            rt_table.add_row(
                rt.name,
                getattr(rt, 'description', '') or '-',
                getattr(rt, 'domain', None) or '-',
                getattr(rt, 'range', None) or '-',
                getattr(rt, 'inverse', None) or '-',
            )
        console.print(Panel(rt_table, title=_("关系类型 (Relation Types)"), border_style="yellow"))
        console.print()


def print_extraction_result(result: Any):
    """Print a summary table of extraction results."""
    if not result or (not result.entities and not result.triples):
        console.print(_("  [!] 未抽取到结果"), style="yellow")
        return

    attr_total = sum(len(e.attributes) for e in result.entities if hasattr(e, 'attributes'))
    entities_with_attrs = sum(1 for e in result.entities if hasattr(e, 'attributes') and e.attributes)

    table = Table(box=box.ROUNDED, border_style="cyan")
    table.add_column(_("指标"), style="bold cyan")
    table.add_column(_("数量"), style="bold white", justify="right")
    table.add_row(_("实体"), str(len(result.entities)))
    table.add_row(_("关系"), str(len(result.relations)))
    table.add_row(_("三元组"), str(len(result.triples)))
    if attr_total > 0:
        table.add_row(_("属性"), _("{total} (分布在 {n} 个实体)").format(
            total=attr_total, n=entities_with_attrs))
    console.print(table)

    if result.triples:
        console.print()
        console.print(_("  [bold]三元组样例:[/bold]"))
        for t in result.triples[:5]:
            console.print(
                f"  [cyan]{t.subject.type}/{t.subject.name}[/cyan] "
                f"[yellow]{t.predicate}[/yellow] "
                f"[cyan]{t.object.type}/{t.object.name}[/cyan]"
            )


def print_stats(result: Any, ontology: Any = None):
    """Print detailed extraction stats with Rich Table."""
    if not result or (not result.entities and not result.triples):
        console.print()
        console.print(_("  [yellow][!]  未抽取到任何实体或关系[/yellow]"))
        console.print(_("  [dim]可能的原因和解决方法:[/dim]"))
        console.print(_("  [dim]  1. 本体类型与文档内容不匹配 → 尝试用更宽泛的类型定义本体[/dim]"))
        console.print(_("  [dim]  2. 文档格式有问题 → 用 --verbose 查看详细日志[/dim]"))
        console.print(_("  [dim]  3. LLM 响应解析失败 → 检查 .kgclaw/logs/kgclaw.log 中的 WARNING[/dim]"))
        console.print(_("  [dim]  4. 试试其他策略: --strategy fast 或 --strategy code[/dim]"))
        return

    console.print()
    console.print(Rule(_("  抽取结果"), style=STYLE["muted"]))
    console.print()

    summary = Table(box=box.ROUNDED, border_style="cyan", show_header=True)
    summary.add_column(_("指标"), style="bold cyan", width=20)
    summary.add_column(_("数量"), style="bold white", justify="right", width=12)
    summary.add_column(_("说明"), style=STYLE["dim"])

    if ontology:
        summary.add_row(_("实体类型（本体定义）"), str(len(ontology.entity_types)), _("用户本体中定义的实体类别"))
        summary.add_row(_("关系类型（本体定义）"), str(len(ontology.relation_types)), _("用户本体中定义的关系类别"))
    summary.add_row(_("抽取实体数"), str(len(result.entities)), _("从文本中识别出的实体实例"))
    summary.add_row(_("抽取关系数"), str(len(result.relations)), _("实体之间的语义关系"))
    summary.add_row(_("构建三元组数"), str(len(result.triples)), _("最终 SPO 三元组"))
    attr_total = sum(len(e.attributes) for e in result.entities if hasattr(e, 'attributes'))
    if attr_total > 0:
        entities_with = sum(1 for e in result.entities if hasattr(e, 'attributes') and e.attributes)
        summary.add_row(_("属性数"), _("{total} (分布在 {n} 个实体)").format(
            total=attr_total, n=entities_with), _("实体的键值属性"))

    console.print(summary)

    # Entity type distribution
    if result.entities:
        type_counts: dict[str, int] = {}
        for e in result.entities:
            type_counts[e.type] = type_counts.get(e.type, 0) + 1

        if type_counts:
            console.print()
            console.print(_("  [bold]实体类型分布[/bold]"))
            max_count = max(type_counts.values())
            for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                bar_len = max(1, int(count / max_count * 25)) if max_count > 0 else 0
                bar = "█" * bar_len
                console.print(f"  [cyan]{etype:<16}[/cyan] {count:>4}  {bar}", style="green")

    # Sample triples
    if result.triples:
        console.print()
        console.print(_("  [bold]三元组样例[/bold]（前 10 个）"))
        console.print()
        sample = result.triples[:10]
        triples_table = Table(box=box.SIMPLE, show_header=True, border_style=STYLE["muted"])
        triples_table.add_column("Subject", style="cyan", max_width=20)
        triples_table.add_column("Predicate", style="bold yellow", max_width=16)
        triples_table.add_column("Object", style="cyan", max_width=20)
        triples_table.add_column("Conf", style="dim", width=6)
        for t in sample:
            triples_table.add_row(
                f"{t.subject.type}/{t.subject.name}",
                t.predicate,
                f"{t.object.type}/{t.object.name}",
                f"{t.confidence:.0%}" if (t.confidence is not None and t.confidence < 1.0) else "✓",
            )
        console.print(triples_table)
        if len(result.triples) > 10:
            console.print(_("  [dim]... 共 {n} 个三元组[/dim]").format(n=len(result.triples)))
