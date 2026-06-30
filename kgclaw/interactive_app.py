"""
Claude Code 风格的交互式 REPL 界面。

提供:
- 对话式交互: 用户输入自然语言, LLM 流式回复
- 斜杠命令: /load, /run, /ontology, /status, /help, /clear, /quit
- 实时流式输出: token-by-token 显示 LLM 生成内容
- 工具调用可视化: 展示 Agent 调用工具的过程
- 会话持久化: 保存和恢复对话上下文
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import os
import sys
import termios
import threading
import time
import traceback
import tty
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PTKStyle

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .agent import Agent, AgentConfig
from .config import UserConfig
from .git_manager import GitManager
from .harness import Harness
from .memory import Memory
from .models import HarnessConfig, LLMConfig
from .prompts.system_prompts import SYSTEM_PROMPT_ORCHESTRATOR
from .ui import make_progress_callback, print_extraction_result
from .i18n import _


console = Console()

# 模块级常量：扫描/加载时跳过的目录
SKIP_DIRS = {'.kgclaw', '.git', '__pycache__', '.venv', 'venv', 'node_modules'}


# ─── 工具函数 ────────────────────────────────────────────────────────────────

# 保存 REPL 启动时的终端设置，用于全局恢复
_initial_terminal_settings = None


def _save_initial_terminal():
    """保存 REPL 启动时的终端设置。在 run_interactive 开始时调用。"""
    global _initial_terminal_settings
    try:
        _initial_terminal_settings = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass


def _restore_terminal():
    """恢复到 REPL 启动时的终端设置 + 显示光标。"""
    try:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
    except Exception:
        pass
    if _initial_terminal_settings is not None:
        try:
            termios.tcsetattr(
                sys.stdin.fileno(),
                termios.TCSANOW,
                _initial_terminal_settings,
            )
        except Exception:
            pass


def _ensure_cursor_visible():
    """恢复终端到已知良好状态。"""
    _restore_terminal()


def _flush_stdin():
    """清空 stdin 缓冲区中的残留数据，并恢复终端可见状态。"""
    import select
    _ensure_cursor_visible()
    try:
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.buffer.read(4096)
    except (OSError, ValueError):
        pass


# ─── Ctrl+O 热键监听（实时切换 verbose 模式）────────────────────────────────

class VerboseToggle:
    """线程安全的热键切换器。

    在后台线程中监听 Ctrl+O (ASCII 15)，实时切换 verbose 标志。
    用于在 /run 执行过程中动态开关详细消息流。
    """

    def __init__(self):
        self._value = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._listener_thread: Optional[threading.Thread] = None
        self._old_terminal_settings = None

    @property
    def value(self) -> bool:
        with self._lock:
            return self._value

    @value.setter
    def value(self, v: bool):
        with self._lock:
            self._value = v

    def toggle(self):
        with self._lock:
            self._value = not self._value
            state = "[ON] 详细流" if self._value else "[OFF] 简洁模式"
            # 使用 print 直接写入 stdout，绕过 Rich console
            sys.stdout.write(f"\r\033[K  {state}                          \n")
            sys.stdout.flush()

    def start_listener(self):
        """启动后台键盘监听线程。"""
        if self._listener_thread and self._listener_thread.is_alive():
            return  # 已经在运行

        self._stop_event.clear()

        def _listen():
            try:
                # 保存终端设置，切换到 cbreak 模式（逐字节读取）
                fd = sys.stdin.fileno()
                self._old_terminal_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)

                while not self._stop_event.is_set():
                    try:
                        ch = sys.stdin.buffer.read(1)
                        if not ch:
                            break
                        # Ctrl+O = ASCII 15 (0x0f)
                        if ch == b'\x0f':
                            self.toggle()
                    except (OSError, ValueError):
                        break
            except Exception:
                pass
            finally:
                # 恢复终端到初始状态（而非 cbreak 之前的状态，因为 cbreak 之前可能已被 Rich 修改）
                _restore_terminal()

        self._listener_thread = threading.Thread(target=_listen, daemon=True)
        self._listener_thread.start()

    def stop_listener(self):
        """停止后台键盘监听线程并恢复终端。"""
        self._stop_event.set()
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=0.5)
        self._listener_thread = None
        _ensure_cursor_visible()

# ─── 配色 ────────────────────────────────────────────────────────────────────

STYLE = {
    "banner": Style(color="bright_cyan", bold=True),
    "user": Style(color="bright_green"),
    "assistant": Style(color="bright_cyan"),
    "tool": Style(color="yellow"),
    "dim": Style(dim=True),
    "muted": Style(color="bright_black"),
    "error": Style(color="red"),
    "success": Style(color="green"),
    "highlight": Style(color="magenta", bold=True),
    "heading": Style(color="bright_yellow", bold=True),
}

COMMANDS_HELP = {
    "/load <path>": "加载文档文件或目录",
    "/docs": "显示已加载的文档列表（源路径、字符数、类型）",
    "/files": "显示已加载的文档列表（同 /docs）",
    "/unload <path|filename>": "移除已加载的某个文档（支持路径或文件名）",
    "/clear-docs": "清除所有已加载的文档",
    "/ontology <text|file>": "设置本体定义（自然语言或文件路径）",
    "/run": "运行完整的 KG 构建流水线",
    "/rebuild": "使用更新后的本体重新构建图谱",
    "/refine <feedback>": "根据反馈分析并优化本体和构建策略",
    "/rollback <hash>": "回滚工作目录到之前的版本",
    "/extract-entities": "仅运行实体抽取",
    "/strategy <auto|fast|standard|code>": "设置工作流策略",
    "/cooccur": "切换共现图谱构建（开启/关闭）",
    "/output <path>": "设置输出文件路径",
    "/format <nt|json|jsonl>": "设置输出格式",
    "/template <1|2|3>": "使用内置本体模板（1=人物关系, 2=企业, 3=法律法规）",
    "/verbose": "切换详细消息流模式（显示 LLM 交互细节）",
    "/debug": "切换调试日志模式（全部细节写入 .kgclaw/logs/）",
    "/status": "显示当前会话状态",
    "/chat <message>": "与 AI 助手自由对话",
    "/history": "显示构建和对话历史",
    "/examples": "查看内置示例",
    "/config": "查看当前 LLM 配置",
    "/clear": "清除对话历史（保留文档和本体）",
    "/reset": "完全重置会话",
    "/help": "显示此帮助",
    "/quit": "退出",
}

# prompt_toolkit 补全
COMMAND_COMPLETER = WordCompleter(
    [k.split()[0] for k in COMMANDS_HELP] + ["@"],
    ignore_case=True,
    sentence=True,
)


class Session:
    """管理交互式会话状态。"""

    def __init__(self, api_key: str, api_base: str, model: str, work_dir: str = ".kgclaw"):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.work_dir = work_dir

        self.memory = Memory(work_dir=self.work_dir)
        self.ontology_raw: Optional[str] = None
        self.previous_ontology_raw: Optional[str] = None  # for diff on ontology update
        self.ontology_updated: bool = False  # True when ontology changed since last build
        self.ontology_is_structured: bool = False  # True when eager analysis succeeded
        self.doc_paths: list[str] = []
        self.doc_texts: list[str] = []
        self.documents_loaded = False
        self.last_result: Any = None
        self.accumulated_context: str = ""  # chat 中积累的本体/实体/关系发现
        self.discovered_files: list[str] = []  # chat agent 读取过的数据文件路径
        self.pre_extracted_entities: list[dict] = []  # extract_with_code 预提取的实体
        self.pre_extracted_relations: list[dict] = []  # extract_with_code 预提取的关系
        self.strategy: str = "auto"  # workflow strategy
        self.enable_co_occurrence: bool = True  # build co-occurrence graph
        self.output_path: str = ".kgclaw/output.nt"  # output file path
        self.output_format: str = "nt"  # output format: nt, json, jsonl
        self.refinement_tips: dict = {}  # tips from /refine for next build
        self.resumed_from: Optional[str] = None  # workflow_id if resumed
        self.file_changes: Optional[dict] = None  # change detection result
        self.needs_rebuild: bool = False          # whether rebuild is needed
        self.rebuild_reason: str = ""             # human-readable reason for rebuild

        self._harness: Optional[Harness] = None
        self._chat_agent: Optional[Agent] = None
        self._git: Optional[GitManager] = None
        self.verbose_toggle = VerboseToggle()

    @property
    def verbose(self) -> bool:
        return self.verbose_toggle.value

    @verbose.setter
    def verbose(self, v: bool):
        self.verbose_toggle.value = v

    @property
    def harness(self) -> Harness:
        if self._harness is None:
            llm_cfg = UserConfig.get_llm_config()
            config = HarnessConfig(
                llm=LLMConfig(
                    model=self.model,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    temperature=float(llm_cfg.get("temperature", 0.3)),
                ),
                work_dir=self.work_dir,
                verbose=True,
            )
            self._harness = Harness(config)
        return self._harness

    @property
    def chat_agent(self) -> Agent:
        """获取或创建 chat agent，每次动态注入当前会话上下文。

        Agent 能看到已加载的文档、已定义的本体、以及之前对话中积累的发现。
        """
        # 构建上下文感知的 system prompt
        context_parts = [SYSTEM_PROMPT_ORCHESTRATOR, ""]
        if self.ontology_raw:
            context_parts.append("## 用户已定义的本体")
            context_parts.append(self.ontology_raw[:2000])
            context_parts.append("")
        if self.documents_loaded and self.doc_paths:
            context_parts.append(f"## 已加载的文档 ({len(self.doc_paths)} 个)")
            for p in self.doc_paths[:5]:
                context_parts.append(f"  - {p}")
            if len(self.doc_paths) > 5:
                context_parts.append(f"  ... 等 {len(self.doc_paths)} 个文件")
            context_parts.append("你可以使用 read_file 工具查看文档内容。")
            context_parts.append("")
        if self.accumulated_context:
            context_parts.append("## 之前对话中积累的发现")
            context_parts.append(self.accumulated_context[-3000:])
            context_parts.append("")

        cfg = AgentConfig(
            name="chat",
            system_prompt="\n".join(context_parts),
            tools=["read_file", "list_files", "search_in_text", "propose_action",
                   "run_python", "analyze_file_format", "extract_with_llm_prompt", "extract_with_code"],
            max_tool_calls=10,
        )
        # 每次都重建以确保上下文是最新的
        self._chat_agent = Agent(cfg, self.memory, self.harness.llm_config)
        return self._chat_agent

    @property
    def git(self) -> GitManager:
        if self._git is None:
            self._git = GitManager(Path(self.work_dir))
            self._git.init()
        return self._git

    def load_docs(self, paths: list[str]):
        self.doc_paths = paths
        self.harness.load_documents(paths)
        self.documents_loaded = True

    def set_ontology(self, raw: str):
        """Set ontology, tracking changes for rebuild detection."""
        if self.ontology_raw and self.ontology_raw.strip() != raw.strip():
            self.previous_ontology_raw = self.ontology_raw
            self.ontology_updated = True
        self.ontology_raw = raw
        result = self.harness.set_ontology(raw)

        # Check if eager analysis produced structured results
        workflow_onto = self.harness.memory.workflow.ontology
        self.ontology_is_structured = (
            workflow_onto.is_structured if workflow_onto else False
        )

        # Git: commit ontology update
        self.memory.export_ontology()
        git = self.git
        short_preview = raw[:80].replace("\n", " ")
        git.commit_ontology_update(short_preview)

        return result

    def status_summary(self) -> str:
        lines = []
        lines.append(f"  Model:  {self.model}")
        lines.append(f"  API:    {self.api_base}")
        onto_status = '已设置 (' + str(len(self.ontology_raw or '')) + ' 字符)'
        if self.ontology_is_structured:
            wf = self.harness.memory.workflow
            if wf and wf.ontology:
                onto = wf.ontology
                onto_status += f" [{len(onto.entity_types)}实体, {len(onto.relation_types)}关系]"
        if self.ontology_updated:
            onto_status += ' [已更新]'
        lines.append(f"  Onto:   {onto_status if self.ontology_raw else '未设置'}")
        live_docs = self.harness.list_documents()
        doc_count = len(live_docs) if live_docs else len(self.doc_paths)
        doc_chars = sum(d.get("chars", 0) for d in live_docs) if live_docs else 0
        is_loaded = bool(live_docs) if live_docs is not None else self.documents_loaded
        lines.append(f"  Docs:   {doc_count} 文件"
                     f"{', ' + f'{doc_chars:,} 字符' if doc_chars else ''}"
                     f"{' (已加载)' if is_loaded else ''}")
        lines.append(f"  Stgy:   {self.strategy}  输出: {self.output_format} → {self.output_path}")
        lines.append(f"  CoOcc:  {'开启' if self.enable_co_occurrence else '关闭'}")
        if self.last_result:
            lines.append(f"  Last:   {getattr(self.last_result, 'entities', []) and len(self.last_result.entities)} entities, "
                         f"{getattr(self.last_result, 'triples', []) and len(self.last_result.triples)} triples")
        if self.resumed_from:
            lines.append(f"  Resume: 来自 workflow {self.resumed_from[:12]}")
        if self.file_changes:
            fc = self.file_changes
            parts = []
            if fc.get("unchanged"): parts.append(f"{len(fc['unchanged'])} 未变")
            if fc.get("added"): parts.append(f"{len(fc['added'])} 新增")
            if fc.get("modified"): parts.append(f"{len(fc['modified'])} 已修改")
            if fc.get("deleted"): parts.append(f"{len(fc['deleted'])} 已删除")
            if parts:
                lines.append(f"  Files:  {', '.join(parts)}")
        if self.needs_rebuild:
            lines.append(f"  Rebuild: [yellow]需要重建[/yellow] — {self.rebuild_reason}")
        # Git version info
        git_hash = self.git.get_current_hash()
        if git_hash:
            lines.append(f"  Git:    {git_hash}")
        return "\n".join(lines)

    @classmethod
    def restore_from_workflow(cls, api_key: str, api_base: str, model: str,
                               work_dir: str = ".kgclaw") -> "Session":
        """Restore a Session from existing workflow state on disk."""
        session = cls(api_key, api_base, model, work_dir=work_dir)
        wf = session.memory.load_workflow()
        if not wf:
            return session

        session.resumed_from = wf.workflow_id

        # Restore ontology
        if wf.ontology:
            if wf.ontology.raw_definition:
                session.ontology_raw = wf.ontology.raw_definition
            else:
                # Reconstruct raw from structured
                session.ontology_raw = wf.ontology.to_extraction_guide()

        # Restore document paths
        session.doc_paths = [d.source for d in wf.documents if d.source]
        if session.doc_paths:
            session.documents_loaded = True
            # Reload into harness
            existing = [p for p in session.doc_paths if Path(p).exists()]
            if existing:
                session.load_docs(existing)

        # Restore last result
        if wf.final_result:
            session.last_result = wf.final_result
            # Update output path if output was generated
            output_nt = session.memory.work_dir / "output.nt"
            if output_nt.exists():
                session.output_path = str(output_nt)

        return session


def _scan_data_files(base_dir: Path) -> list[str]:
    """Scan directory for supported data files, excluding tool directories.

    Used for file change detection when resuming a session.
    """
    from .loaders import supported_extensions
    data_files = []
    skip = SKIP_DIRS
    try:
        for ext in supported_extensions():
            for f in base_dir.glob(f"**/*{ext}"):
                parts = set(f.parts)
                if not parts.intersection(skip):
                    data_files.append(str(f))
    except Exception:
        pass
    return data_files


def print_welcome(session: Session):
    """打印欢迎界面。"""
    console.print()
    banner = Panel(
        Text(_("KGClaw 交互式会话"), style=STYLE["banner"], justify="center"),
        subtitle="基于 AI Agent Harness 的知识图谱构建工具",
        subtitle_align="center",
        box=box.ROUNDED,
        border_style="cyan",
    )
    console.print(banner)

    console.print()
    info = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    info.add_column(style="cyan", width=10)
    info.add_column(style="white")
    info.add_row("Model", session.model)
    info.add_row("API", session.api_base)
    console.print(info)

    console.print()
    console.print(_("  [bold]快速开始:[/bold]"))
    console.print(_("    [dim]/ontology 人物: 自然人  \\\\n 关系: 生父, 儿子, 老师[/dim]"))
    console.print(_("    [dim]/load examples/人物图谱/人物关系图谱原始数据.txt[/dim]"))
    console.print("    [dim]/run[/dim]")
    console.print()
    console.print(f"  输入 [bold]/help[/bold] 查看所有命令, [bold]/quit[/bold] 退出")
    console.print()


def run_interactive(api_key: str, api_base: str, model: str, work_dir: str = ".kgclaw"):
    """主 REPL 循环。"""
    from .cli import _fix_stdin_encoding
    from .i18n import init_locale
    _fix_stdin_encoding()  # ensure stdin encoding is UTF-8 for Chinese input
    init_locale()  # ensure i18n is initialized (no-op if already done by CLI main())
    _save_initial_terminal()  # 保存启动时的终端状态，供后续恢复

    # ── Session Resume Detection ─────────────────────────────────────────
    work_path = Path(work_dir)
    state_file = work_path / "workflow_state.json"
    resumed = False

    if state_file.exists():
        # Existing session found — prompt user
        console.print()
        console.print(Panel(
            Group(
                Text(_("检测到之前的构建会话"), style=Style(color="yellow", bold=True), justify="center"),
                Text(""),
                Text(f"工作目录: {work_path.resolve()}", justify="center"),
                Text(f"状态文件: workflow_state.json", justify="center"),
            ),
            box=box.ROUNDED,
            border_style="yellow",
        ))
        from rich.prompt import Confirm as _Confirm
        if _Confirm.ask("  是否继续上次的会话？", default=True):
            session = Session.restore_from_workflow(api_key, api_base, model, work_dir=work_dir)
            resumed = True
            console.print()
            if session.last_result:
                console.print(f"  [OK] 已恢复会话 [bold]{session.resumed_from[:12]}[/bold]", style="green")
                console.print(f"  [dim]上次构建: {len(session.last_result.entities)} entities, "
                             f"{len(session.last_result.triples)} triples[/dim]")
            else:
                console.print(f"  [OK] 已恢复会话 [bold]{session.resumed_from[:12]}[/bold] (无构建结果)", style="green")
            if session.ontology_raw:
                console.print(f"  [dim]本体已加载 ({len(session.ontology_raw)} 字符)[/dim]")
            if session.doc_paths:
                console.print(f"  [dim]文档已加载 ({len(session.doc_paths)} 文件)[/dim]")

            # ── File Change Detection ─────────────────────────────────
            # Scan current directory for data files and compare against manifest
            current_files = _scan_data_files(work_path)
            if current_files:
                changes = session.memory.detect_file_changes(current_files)
                session.file_changes = changes

                has_file_changes = bool(
                    changes.get("added") or changes.get("modified") or changes.get("deleted")
                )
                has_onto_change = session.ontology_updated

                # Display change summary
                console.print()
                change_parts = []
                if changes.get("unchanged"):
                    change_parts.append(f"📄 {len(changes['unchanged'])} 个文件未变化")
                if changes.get("added"):
                    change_parts.append(f"➕ {len(changes['added'])} 个新文件")
                if changes.get("modified"):
                    change_parts.append(f"✏️  {len(changes['modified'])} 个文件已修改")
                if changes.get("deleted"):
                    change_parts.append(f"🗑️  {len(changes['deleted'])} 个文件已删除")
                if has_onto_change:
                    change_parts.append("📝 本体已更新")

                if not has_file_changes and not has_onto_change:
                    session.needs_rebuild = False
                    session.rebuild_reason = ""
                    console.print(f"  [green]✅ 文件与本体均无变化，无需重建[/green]")
                    if session.last_result:
                        console.print(f"  [dim]使用缓存结果: {len(session.last_result.entities)} entities, "
                                     f"{len(session.last_result.triples)} triples[/dim]")
                else:
                    session.needs_rebuild = True
                    reasons = []
                    if has_file_changes:
                        reasons.append("文件有变化")
                    if has_onto_change:
                        reasons.append("本体已更新")
                    session.rebuild_reason = " + ".join(reasons)

                    console.print(f"  [yellow]检测到变化: {', '.join(change_parts)}[/yellow]")
                    console.print(f"  [yellow]建议: 完全重建（{session.rebuild_reason}）[/yellow]")

                # Show details for added/modified/deleted files
                for label, style, paths in [
                    ("新增", "green", changes.get("added", [])),
                    ("修改", "yellow", changes.get("modified", [])),
                    ("删除", "red", changes.get("deleted", [])),
                ]:
                    for fp in paths[:3]:
                        console.print(f"    [{style}]{label}[/{style}]: {Path(fp).name}")
                    if len(paths) > 3:
                        console.print(f"    [dim]... 等 {len(paths)} 个文件[/dim]")

            # Show git history
            history = session.git.get_history(5)
            if history:
                console.print(f"  [dim]最近的构建版本:[/dim]")
                for h in history[:3]:
                    console.print(f"    [dim]{h['hash']} {h['date']} {h['message'][:60]}[/dim]")
        else:
            session = Session(api_key, api_base, model, work_dir=work_dir)
            console.print(_("  [dim]已开始全新会话[/dim]"))
    else:
        session = Session(api_key, api_base, model, work_dir=work_dir)

    # Initialize git for version management
    session.git.init()

    # prompt_toolkit 配置
    history_file = Path.home() / ".kgclaw" / "repl_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    ptk_style = PTKStyle.from_dict({
        "prompt": "bold green",
        "": "",
    })

    kb = KeyBindings()

    @kb.add("c-d")
    def _(event):
        """Ctrl+D 退出。"""
        event.app.exit()

    session_obj = PromptSession(
        history=FileHistory(str(history_file)),
        completer=COMMAND_COMPLETER,
        style=ptk_style,
        key_bindings=kb,
    )

    print_welcome(session)

    # ── REPL 循环 ──────────────────────────────────────────────────────────
    while True:
        _ensure_cursor_visible()  # 每次 prompt 前确保光标可见
        try:
            raw = session_obj.prompt(
                [("class:prompt", "\n> ")],
                multiline=False,
            )
        except EOFError:
            _ensure_cursor_visible()
            console.print(_("\n  再见！"))
            break
        except KeyboardInterrupt:
            console.print("^C")
            continue

        # Ctrl+D 通过 keybinding 返回 None
        if raw is None:
            _ensure_cursor_visible()
            console.print(_("\n  再见！"))
            break

        user_input = raw.strip()

        if not user_input:
            continue

        # ── 命令分发（内部捕获 Ctrl+C 中断当前操作）──────────────────────
        try:
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                    _ensure_cursor_visible()
                    console.print(_("  再见！"))
                    break

                elif cmd == "/help":
                    _cmd_help()

                elif cmd == "/config":
                    _cmd_config(session)

                elif cmd == "/status":
                    _cmd_status(session)

                elif cmd == "/verbose":
                    session.verbose = not session.verbose
                    status = "[ON] 开启" if session.verbose else "[OFF] 关闭"
                    console.print(f"  {status} 详细消息流模式", style="green")
                    if session.verbose:
                        console.print(_("  [dim]现在 /run 和 /chat 将显示完整的 LLM 交互过程[/dim]"))

                elif cmd == "/strategy":
                    valid_strategies = ["auto", "fast", "standard", "code"]
                    if arg.strip() in valid_strategies:
                        session.strategy = arg.strip()
                        console.print(f"  [OK] 工作流策略: [bold]{session.strategy}[/bold]", style="green")
                        desc = {"auto": "自动选择", "fast": "快速合并抽取", "standard": "标准多阶段", "code": "代码沙盒抽取"}
                        console.print(f"  [dim]{desc[session.strategy]}[/dim]")
                    else:
                        console.print(f"  [!] 无效策略。可选: {', '.join(valid_strategies)}", style="yellow")

                elif cmd == "/cooccur":
                    session.enable_co_occurrence = not session.enable_co_occurrence
                    status = "[ON] 开启" if session.enable_co_occurrence else "[OFF] 关闭"
                    console.print(f"  {status} 共现图谱构建", style="green")

                elif cmd == "/output":
                    if arg.strip():
                        session.output_path = str(Path(arg.strip()).resolve())
                        console.print(f"  [OK] 输出路径: [bold]{session.output_path}[/bold]", style="green")
                    else:
                        console.print(f"  [!] 用法: /output <文件路径>", style="yellow")

                elif cmd == "/format":
                    valid_formats = ["nt", "json", "jsonl"]
                    if arg.strip() in valid_formats:
                        session.output_format = arg.strip()
                        console.print(f"  [OK] 输出格式: [bold]{session.output_format}[/bold]", style="green")
                    else:
                        console.print(f"  [!] 无效格式。可选: {', '.join(valid_formats)}", style="yellow")

                elif cmd == "/template":
                    from .cli import ONTOLOGY_TEMPLATES
                    if arg.strip() in ONTOLOGY_TEMPLATES:
                        tmpl = ONTOLOGY_TEMPLATES[arg.strip()]
                        session.ontology_raw = tmpl["template"]
                        console.print(f"  [OK] 本体模板: [bold]{tmpl['name']}[/bold]", style="green")
                        console.print(f"  [dim]提示: 输入 /run 开始构建知识图谱[/dim]")
                    else:
                        console.print(f"  [!] 无效模板编号。可选: 1=人物关系, 2=企业, 3=法律法规", style="yellow")

                elif cmd == "/debug":
                    from .logger import get_logger
                    log = get_logger()
                    import logging as _logging
                    # 检查当前 handler 的实际级别（不是 logger 级别）
                    is_debug = any(
                        hasattr(h, 'level') and h.level <= _logging.DEBUG
                        for h in log._logger.handlers
                        if hasattr(h, 'baseFilename')  # 只检查文件 handler
                    )
                    if is_debug:
                        # 关闭 debug: 文件 handler → INFO, logger → INFO
                        for h in log._logger.handlers:
                            if hasattr(h, 'baseFilename'):
                                h.setLevel(_logging.INFO)
                        log._logger.setLevel(_logging.INFO)
                        console.print(_("  [OFF] 调试日志已关闭 (仅记录 INFO+)"), style="green")
                    else:
                        # 开启 debug: 文件 handler → DEBUG, logger → DEBUG
                        log._logger.setLevel(_logging.DEBUG)
                        for h in log._logger.handlers:
                            if hasattr(h, 'baseFilename'):
                                h.setLevel(_logging.DEBUG)
                        log_dir = log.log_dir or ".kgclaw/logs"
                        console.print(f"  [ON] 调试日志已开启", style="green")
                        console.print(f"  [dim]日志文件: {log_dir}/kgclaw.log[/dim]")

                elif cmd == "/clear":
                    session.memory.clear_messages()
                    session._chat_agent = None
                    console.print(_("  [OK] 对话历史已清除"), style="green")

                elif cmd == "/reset":
                    session = Session(api_key, api_base, model)
                    console.print(_("  [OK] 会话已完全重置"), style="green")

                elif cmd == "/history":
                    _cmd_history(session)

                elif cmd == "/examples":
                    _cmd_examples(session)

                elif cmd == "/load":
                    _cmd_load(session, arg)

                elif cmd == "/docs" or cmd == "/files":
                    _cmd_docs(session)

                elif cmd == "/unload":
                    _cmd_unload(session, arg)

                elif cmd == "/clear-docs":
                    _cmd_clear_docs(session)

                elif cmd == "/ontology":
                    _cmd_ontology(session, arg)

                elif cmd == "/run":
                    _cmd_run(session)

                elif cmd == "/extract-entities":
                    _cmd_extract_entities(session)

                elif cmd == "/chat":
                    _cmd_chat(session, arg)

                elif cmd == "/rebuild":
                    _cmd_rebuild(session)

                elif cmd == "/refine":
                    _cmd_refine(session, arg)

                elif cmd == "/rollback":
                    _cmd_rollback(session, arg)

                else:
                    console.print(f"  [!] 未知命令: [bold]{cmd}[/bold]。输入 /help 查看可用命令。", style="yellow")

            elif user_input.startswith("@"):
                # @ 快捷方式：加载文件
                path = user_input[1:].strip()
                _cmd_load(session, path)

            else:
                # 自由对话模式
                _cmd_chat(session, user_input)

        except KeyboardInterrupt:
            # Ctrl+C 中断当前操作
            session.verbose_toggle.stop_listener()
            _ensure_cursor_visible()
            _flush_stdin()
            console.print()
            console.print(_("  [yellow]^C 已中断当前操作，回到待机模式[/yellow]"))
            console.print(_("  [dim]提示: 按 [bold]Ctrl+D[/bold] 或输入 [bold]/quit[/bold] 退出[/dim]"))
            # 继续 REPL 循环


# ─── 命令实现 ────────────────────────────────────────────────────────────────

def _cmd_help():
    console.print()
    console.print(_("  [bold]可用命令:[/bold]"))
    console.print()
    for cmd, desc in COMMANDS_HELP.items():
        console.print(f"  [bold cyan]{cmd:<28}[/bold cyan] {desc}")
    console.print()
    console.print(_("  [dim]以 @ 开头的输入会被视为文件路径（快捷加载）[/dim]"))
    console.print(_("  [dim]不以 / 或 @ 开头的输入会进入 AI 对话模式[/dim]"))
    console.print()


def _cmd_config(session: Session):
    saved = UserConfig.load()
    llm = saved.get("llm", {})
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column(style="cyan")
    table.add_column(style="white")
    table.add_row("API Base", llm.get("api_base", "N/A"))
    table.add_row("Model", llm.get("model", "N/A"))
    key = llm.get("api_key", "")
    table.add_row("API Key", key[:8] + "..." + key[-4:] if len(key) > 12 else "***")
    console.print(table)


def _cmd_status(session: Session):
    console.print()
    console.print(Panel(
        Text(session.status_summary()),
        title="会话状态",
        border_style="cyan",
    ))
    console.print()


def _cmd_history(session: Session):
    """显示对话历史和构建版本历史。"""
    console.print()

    # Git build history
    git_history = session.git.get_history(10)
    if git_history:
        console.print(_("  [bold]构建版本历史 (git):[/bold]"))
        for h in git_history:
            icon = "📦" if h["message"].startswith("build:") else "📝"
            console.print(f"  {icon} [cyan]{h['hash']}[/cyan] {h['date']} {h['message'][:70]}")
        console.print()

    # Chat history
    msgs = session.memory.get_messages("chat")
    if msgs:
        console.print(_("  [bold]对话历史:[/bold]"))
        for i, msg in enumerate(msgs[-20:], 1):
            role_style = STYLE["user"] if msg.role.value == "user" else STYLE["assistant"]
            preview = msg.content[:120].replace("\n", " ")
            console.print(f"  [{i}] [{role_style}]{msg.role.value[:4]:>4}[/{role_style}] {preview}...")
        console.print(f"  [dim]共 {len(msgs)} 条消息[/dim]")
    elif not git_history:
        console.print(_("  [dim]暂无历史记录[/dim]"))


def _cmd_examples(session: Session):
    examples_dir = Path(__file__).parent.parent / "examples"
    console.print()
    if examples_dir.exists():
        for d in sorted(examples_dir.iterdir()):
            if d.is_dir():
                files = list(d.rglob("*"))
                data_files = [f for f in files if f.suffix in ('.txt', '.jsonl', '.nt', '.docx')]
                console.print(f"  [bold cyan]{d.name}[/bold cyan] — {len(data_files)} 文件")
                # 找到第一个 txt 文件作为快捷路径
                txt_files = [f for f in data_files if f.suffix == '.txt']
                if txt_files:
                    console.print(f"    [dim]快捷加载: @{txt_files[0].relative_to(Path.cwd())}[/dim]")
    console.print()


def _cmd_load(session: Session, arg: str):
    if not arg:
        console.print(_("  [!] 用法: /load <文件路径或目录>"), style="yellow")
        console.print(_("  [dim]例: /load examples/人物图谱/人物关系图谱原始数据.txt[/dim]"))
        return

    path = Path(arg.strip())
    if not path.exists():
        console.print(f"  [FAIL] 路径不存在: {path}", style="red")
        return

    if path.is_dir():
        from .loaders import supported_extensions
        doc_paths = []
        skip = SKIP_DIRS
        for ext in supported_extensions():
            for p in path.rglob(f"*{ext}"):
                if not set(p.parts).intersection(skip):
                    doc_paths.append(str(p))
        if not doc_paths:
            console.print(f"  [!] 目录中无支持的文档文件", style="yellow")
            return
        session.load_docs(doc_paths)
        console.print(f"  [OK] 已加载 {len(doc_paths)} 个文件", style="green")
        for p in doc_paths[:5]:
            console.print(f"    [dim]{p}[/dim]")
        if len(doc_paths) > 5:
            console.print(f"    [dim]... 等 {len(doc_paths)} 个文件[/dim]")
    else:
        session.load_docs([str(path)])
        size = path.stat().st_size
        lines = path.read_text().count('\n') + 1
        console.print(f"  [OK] 已加载 [bold]{path.name}[/bold] ({lines} 行, {size:,} bytes)", style="green")


def _cmd_docs(session: Session):
    """Display all loaded documents with metadata."""
    docs = session.harness.list_documents()
    if not docs:
        console.print()
        console.print(_("  [dim]当前未加载任何文档[/dim]"))
        console.print(_("  [dim]使用 /load <路径> 加载文档[/dim]"))
        console.print()
        return

    console.print()
    table = Table(
        title=f"已加载文档 ({len(docs)} 个)",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column(_("文件名"), style="white", no_wrap=True)
    table.add_column(_("类型"), style="yellow", width=6)
    table.add_column(_("字符数"), style="green", justify="right", width=10)
    table.add_column(_("大小"), style="dim", justify="right", width=10)
    table.add_column(_("源路径"), style="dim")

    for i, d in enumerate(docs, 1):
        ext = d.get("ext", Path(d["source"]).suffix if d["source"] else "?")
        chars = d.get("chars", 0)
        size_bytes = d.get("size", 0)
        if size_bytes >= 1024 * 1024:
            size_str = f"{size_bytes / (1024*1024):.1f} MB"
        elif size_bytes >= 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes} B" if size_bytes else "-"
        filename = d.get("filename", Path(d["source"]).name if d["source"] else "(unknown)")
        table.add_row(
            str(i),
            filename,
            ext.lstrip(".").upper() if ext else "?",
            f"{chars:,}",
            size_str,
            d.get("source", ""),
        )

    console.print(table)
    total_chars = sum(d.get("chars", 0) for d in docs)
    console.print(f"  [dim]{_('总计')}: {len(docs)} {_('个文件')}, {total_chars:,} {_('字符')}[/dim]")
    console.print()


def _cmd_unload(session: Session, arg: str):
    """Remove a loaded document by source path or filename."""
    if not arg:
        console.print(_("  [!] 用法: /unload <文件路径或文件名>"), style="yellow")
        console.print(_("  [dim]使用 /docs 查看已加载的文件[/dim]"))
        return

    target = arg.strip()
    docs = session.harness.list_documents()

    if not docs:
        console.print(_("  [dim]当前未加载任何文档[/dim]"))
        return

    # Strategy 1: exact match by full source path
    matches = [d for d in docs if d["source"] == target]

    # Strategy 2: match by basename
    if not matches:
        target_basename = Path(target).name
        matches = [d for d in docs if Path(d["source"]).name == target_basename]

    # Strategy 3: match by substring in source path
    if not matches:
        matches = [d for d in docs if target in d["source"]]

    if not matches:
        console.print(f"  [FAIL] {_('未找到已加载的文件')}: [bold]{target}[/bold]", style="yellow")
        console.print(_("  [dim]使用 /docs 查看所有已加载的文件[/dim]"))
        return

    if len(matches) > 1:
        console.print(f"  [yellow]{_('找到多个匹配文件')}:[/yellow]")
        for m in matches:
            console.print(f"    [dim]{m['source']}[/dim]")
        console.print(_("  [dim]请使用完整路径以精确指定要移除的文件[/dim]"))
        return

    source = matches[0]["source"]
    removed = session.harness.unload_document(source)

    if removed:
        # Update session state
        session.doc_paths = [p for p in session.doc_paths if p != source]
        if not session.doc_paths:
            session.documents_loaded = False
        chars = matches[0].get("chars", 0)
        console.print(f"  [OK] {_('已移除')}: [bold]{Path(source).name}[/bold] ({chars:,} {_('字符')})", style="green")
    else:
        console.print(f"  [FAIL] {_('移除失败')}: {source}", style="red")


def _cmd_clear_docs(session: Session):
    """Remove all loaded documents."""
    docs = session.harness.list_documents()
    if not docs:
        console.print(_("  [dim]当前未加载任何文档[/dim]"))
        return

    count = len(docs)
    total_chars = sum(d.get("chars", 0) for d in docs)

    from rich.prompt import Confirm as _ConfirmClear
    console.print()
    if not _ConfirmClear.ask(f"  {_('确认清除所有')} {count} {_('个已加载的文档')}？"):
        console.print(_("  [dim]已取消[/dim]"))
        return

    removed = session.harness.clear_documents()
    if removed > 0:
        session.doc_paths = []
        session.documents_loaded = False
        console.print(f"  [OK] {_('已清除')} [bold]{removed}[/bold] {_('个文档')} ({total_chars:,} {_('字符')})", style="green")
    else:
        console.print(_("  [FAIL] 清除失败"), style="red")


def _cmd_ontology(session: Session, arg: str):
    if not arg:
        console.print(_("  [!] 用法: /ontology <本体定义文本或文件路径>"), style="yellow")
        console.print(_("  [dim]例: /ontology Entity Types: 人物 \\\\n Relation Types: 生父, 儿子[/dim]"))
        console.print(_("  [dim]例: /ontology @my_ontology.yaml[/dim]"))
        return

    arg = arg.strip()
    if arg.startswith("@"):
        # 从文件加载
        onto_path = Path(arg[1:].strip())
        if not onto_path.exists():
            console.print(f"  [FAIL] 文件不存在: {onto_path}", style="red")
            return
        raw = onto_path.read_text()
    else:
        # 将 \\n 还原为换行符
        raw = arg.replace("\\n", "\n")

    session.set_ontology(raw)

    if session.ontology_updated and session.previous_ontology_raw:
        console.print(f"  [OK] 本体已更新 ([dim]{len(raw)} 字符[/dim])", style="green")
    else:
        console.print(f"  [OK] 本体已设置 ([dim]{len(raw)} 字符[/dim])", style="green")

    # Display structured analysis result, or fall back to raw text preview
    if session.ontology_is_structured:
        onto = session.harness.memory.workflow.ontology
        if onto:
            from .ui.display import print_ontology
            print_ontology(onto)
    else:
        preview = raw[:200].replace("\n", " ")
        console.print(f"  [dim]{preview}...[/dim]")
        console.print(_("  [dim]（LLM 分析将在运行 /run 时进行）[/dim]"))

    if session.ontology_updated and session.previous_ontology_raw:
        console.print(f"  [dim]提示: 使用 /rebuild 基于新本体重新构建图谱[/dim]")

def _cmd_run(session: Session):
    # ── Skip build if nothing changed ──────────────────────────────────
    if (session.file_changes is not None
            and not session.needs_rebuild
            and not session.ontology_updated
            and session.last_result):
        console.print()
        console.print(_("  [green]✅ 文件与本体均无变化，使用缓存结果[/green]"))
        console.print(f"  [dim]上次构建: {len(session.last_result.entities)} entities, "
                     f"{len(session.last_result.triples)} triples[/dim]")
        console.print()
        print_extraction_result(session.last_result)
        return

    # Show ontology update warning if applicable
    if session.ontology_updated and session.last_result:
        console.print()
        console.print(_("  [yellow]⚠ 本体已更新，将基于新本体重新构建图谱[/yellow]"))
        if session.previous_ontology_raw:
            old_lines = set(session.previous_ontology_raw.strip().split("\n"))
            new_lines = set((session.ontology_raw or "").strip().split("\n"))
            added = new_lines - old_lines
            removed = old_lines - new_lines
            if added:
                added_preview = ", ".join(list(added)[:3])
                console.print(f"  [dim]+ 新增: {added_preview}[/dim]")
            if removed:
                removed_preview = ", ".join(list(removed)[:3])
                console.print(f"  [dim]- 移除: {removed_preview}[/dim]")

    # 自动加载 chat 中发现的数据文件
    # 智能文档加载: chat中发现 → 当前目录扫描
    if not session.documents_loaded:
        loaded = False
        # Step 1: 从 chat 中发现的文件
        if session.discovered_files:
            existing = [p for p in session.discovered_files if Path(p).exists()]
            if existing:
                console.print(f"  [dim]自动加载 chat 中发现的 {len(existing)} 个文件...[/dim]")
                session.load_docs(existing)
                session.doc_paths = existing
                loaded = True
        # Step 2: 如果什么都没发现，扫描当前目录下的所有数据文件
        if not loaded:
            from pathlib import Path as _Path
            from .loaders import supported_extensions
            cwd = _Path.cwd()
            data_files = []
            skip_dirs = SKIP_DIRS
            for ext in supported_extensions():
                for f in cwd.glob(f"**/*{ext}"):
                    # 跳过隐藏目录和工具目录
                    parts = set(f.parts)
                    if not parts.intersection(skip_dirs):
                        data_files.append(str(f))
            if data_files:
                # 分类统计
                from collections import Counter
                ext_counts = Counter(_Path(f).suffix for f in data_files)
                ext_summary = ", ".join(f"{cnt} {ext}" for ext, cnt in ext_counts.most_common(5))
                console.print(
                    f"  [dim]自动扫描当前目录，发现 {len(data_files)} 个数据文件 ({ext_summary})...[/dim]"
                )
                session.load_docs(data_files)
                session.doc_paths = data_files
                loaded = True

    if not session.documents_loaded:
        console.print(_("  [!] 请先加载文档: /load <路径>"), style="yellow")
        return

    console.print()
    if not session.ontology_raw:
        console.print(_("  [dim]未设置本体，将从文档中自动推断实体和关系类型...[/dim]"))
    console.print(Rule(_("开始构建知识图谱"), style=STYLE["muted"]))

    # 重新创建 harness 以确保事件回调正确绑定
    llm_cfg = UserConfig.get_llm_config()
    config = HarnessConfig(
        llm=LLMConfig(
            model=session.model, api_key=session.api_key, api_base=session.api_base,
            temperature=float(llm_cfg.get("temperature", 0.3)),
        ),
        work_dir=session.work_dir,
        verbose=True,
    )
    harness = Harness(config)
    progress_cb, progress_stop = make_progress_callback(verbose_toggle=session.verbose_toggle)
    harness.on_event(progress_cb)

    # 提示 Ctrl+O 快捷键
    console.print(
        "  [dim]提示: 按 [bold]Ctrl+O[/bold] 可实时切换详细消息流[/dim]"
    )

    # 重新加载文档和本体
    if session.doc_paths:
        harness.load_documents(session.doc_paths)

    # 合并 chat 中积累的本体发现
    ontology_to_use = session.ontology_raw
    if not ontology_to_use and session.accumulated_context.strip():
        console.print(
            "  [dim]从之前的对话中发现了以下信息，将用作本体推断的提示:[/dim]"
        )
        console.print(f"  [dim]{session.accumulated_context[-500:]}[/dim]")
        # 将累积上下文作为 user_notes 传给 ontology_analyzer
        ontology_to_use = (
            "请根据文档内容自动推断合理的知识图谱本体。\n"
            f"提示: {session.accumulated_context[-2000:]}"
        )
    if ontology_to_use:
        harness.set_ontology(ontology_to_use)

    # 启动 Ctrl+O 键盘监听
    session.verbose_toggle.start_listener()

    try:
        result = harness.run(
            ontology_raw=ontology_to_use,
            strategy=session.strategy,
            enable_co_occurrence=session.enable_co_occurrence,
        )

        # 合并 chat 中 extract_with_code 预提取的结果
        if session.pre_extracted_entities or session.pre_extracted_relations:
            from .models import Entity as _E, Relation as _R, Triple as _T
            for e_data in session.pre_extracted_entities:
                result.entities.append(_E(
                    name=e_data.get("name", ""),
                    type=e_data.get("type", e_data.get("entity_type", "")),
                    confidence=e_data.get("confidence", 0.95),
                ))
            for r_data in session.pre_extracted_relations:
                pred = r_data.get("predicate", r_data.get("relation", ""))
                subj = r_data.get("subject", "")
                obj = r_data.get("object", "")
                result.relations.append(_R(
                    subject=subj, predicate=pred, object=obj,
                    confidence=r_data.get("confidence", 0.90),
                ))
                # 也为预提取的关系生成三元组
                # 找到或创建 subject/object 实体
                subj_e = _E(name=subj, type="Entity", confidence=0.90)
                obj_e = _E(name=obj, type="Entity", confidence=0.90)
                result.triples.append(_T(
                    subject=subj_e, predicate=pred, object=obj_e,
                    confidence=r_data.get("confidence", 0.90),
                ))
            console.print(
                f"  [dim]已合并预提取: +{len(session.pre_extracted_entities)} 实体, "
                f"+{len(session.pre_extracted_relations)} 关系[/dim]"
            )
        session.last_result = result

        # 显示结果
        console.print()
        print_extraction_result(result)

        # 导出
        output_base = str(Path(session.output_path).with_suffix(''))
        if session.output_format == "nt":
            harness.export_nt(f"{output_base}.nt")
            harness.export_json(f"{output_base}.json")
            console.print(f"  [OK] 输出: [bold]{output_base}.nt[/bold] [dim]+ .json[/dim]", style="green")
        elif session.output_format == "json":
            harness.export_json(f"{output_base}.json")
            console.print(f"  [OK] 输出: [bold]{output_base}.json[/bold]", style="green")
        elif session.output_format == "jsonl":
            harness.export_jsonl(f"{output_base}.jsonl")
            console.print(f"  [OK] 输出: [bold]{output_base}.jsonl[/bold]", style="green")

        # Export standalone ontology files
        session.memory.export_ontology()

        # Save document manifest for future change detection
        session.memory.save_document_manifest(harness.memory.workflow.documents)

        # Clear file change state (current state is now the baseline)
        session.file_changes = None
        session.needs_rebuild = False
        session.rebuild_reason = ""

        # Git: commit this build run
        git_hash = session.git.commit_build(
            harness.memory.workflow.workflow_id if harness.memory.workflow else "unknown",
            {
                "entities": len(result.entities),
                "relations": len(result.relations),
                "triples": len(result.triples),
            },
        )
        if git_hash:
            console.print(f"  [dim]Git: {git_hash}[/dim]")

        # Reset ontology update flag after successful build
        session.ontology_updated = False
        session.previous_ontology_raw = None

    except KeyboardInterrupt:
        # Ctrl+C 中断构建：已在 REPL 层处理，这里只确保清理
        console.print()
        console.print(_("  [yellow]^C 构建已中断[/yellow]"))
    except Exception as e:
        console.print(f"  [FAIL] 构建失败: {e}", style="red")
        traceback.print_exc()
    finally:
        # 停止进度条、键盘监听、清空 stdin 缓冲区
        progress_stop()
        session.verbose_toggle.stop_listener()
        _flush_stdin()


def _cmd_rebuild(session: Session):
    """使用更新后的本体重新构建知识图谱。

    与 /run 不同，/rebuild 明确表示本体已更新，会显示本体变化差异。
    """
    if not session.documents_loaded:
        console.print(_("  [!] 请先加载文档: /load <路径>"), style="yellow")
        return
    if not session.ontology_raw:
        console.print(_("  [!] 请先设置本体: /ontology <定义>"), style="yellow")
        return

    console.print()
    console.print(Rule(_("重新构建知识图谱（本体已更新）"), style=STYLE["muted"]))

    if session.previous_ontology_raw:
        old_lines = set(session.previous_ontology_raw.strip().split("\n"))
        new_lines = set(session.ontology_raw.strip().split("\n"))
        added = new_lines - old_lines
        removed = old_lines - new_lines
        if added or removed:
            console.print(_("  [bold]本体变更:[/bold]"))
            for a in sorted(added)[:5]:
                console.print(f"  [green]+ {a.strip()[:80]}[/green]")
            for r in sorted(removed)[:5]:
                console.print(f"  [red]- {r.strip()[:80]}[/red]")
            if len(added) > 5 or len(removed) > 5:
                console.print(f"  [dim]... 等 {len(added)} 新增, {len(removed)} 移除[/dim]")
        console.print()

    # Delegate to _cmd_run (which handles the actual build)
    _cmd_run(session)


def _cmd_refine(session: Session, arg: str):
    """Analyze the last build result with user feedback and propose improvements.

    Uses the RefinementEngine to analyze what went wrong in the last build
    and suggest concrete changes to ontology, strategy, and prompts.
    """
    if not arg or not arg.strip():
        console.print(_("  [!] 用法: /refine <反馈意见>"), style="yellow")
        console.print(_("  [dim]例: /refine 实体类型太少，需要增加作者和编辑者类型[/dim]"))
        console.print(_("  [dim]例: /refine 关系抽取遗漏了很多跨句子的关系[/dim]"))
        console.print(_("  [dim]例: /refine 人物应该细分为作者和编辑者，关系抽取时扩大上下文[/dim]"))
        return

    if not session.last_result:
        console.print(_("  [!] 没有可分析的上次构建结果。请先运行 /run。"), style="yellow")
        return

    if not session.ontology_raw:
        console.print(_("  [!] 没有设置本体定义。请先使用 /ontology 设置。"), style="yellow")
        return

    feedback = arg.strip()

    # Get the structured ontology from the harness
    harness = session.harness
    wf = harness.memory.workflow
    ontology = wf.ontology if wf else None
    if not ontology:
        console.print(_("  [!] 无法获取当前本体定义"), style="red")
        return

    docs = wf.documents if wf else []

    console.print()
    console.print(f"  [bold cyan]分析反馈中...[/bold cyan]")
    console.print(f"  [dim]反馈: {feedback[:200]}[/dim]")

    # Create refinement engine and run analysis
    from .refinement import RefinementEngine
    llm_cfg = harness.llm_config
    engine = RefinementEngine(llm_cfg, harness.memory)

    with console.status("[cyan]Refinement Agent 分析中...", spinner="dots"):
        plan = engine.analyze(
            last_result=session.last_result,
            ontology=ontology,
            docs=docs,
            user_feedback=feedback,
            strategy=session.strategy,
        )

    if not plan.has_changes:
        console.print()
        console.print(_("  [yellow][!] 分析完成，但未发现需要修改的地方。[/yellow]"))
        if plan.rationale:
            console.print(f"  [dim]分析说明: {plan.rationale[:500]}[/dim]")
        return

    # Display the refinement plan
    console.print()
    console.print(Rule(_("  优化方案"), style="cyan"))
    console.print()

    if plan.rationale:
        console.print(f"  [bold]分析:[/bold]")
        for line in plan.rationale.split("\n"):
            if line.strip():
                console.print(f"  [dim]{line.strip()}[/dim]")
        console.print()

    # Ontology changes
    if plan.ontology_changes:
        console.print(_("  [bold cyan]本体变更:[/bold cyan]"))
        for oc in plan.ontology_changes:
            icon = {"add": "+", "remove": "-", "modify": "~"}.get(oc.action, "?")
            color = {"add": "green", "remove": "red", "modify": "yellow"}.get(oc.action, "white")
            target_cn = "实体类型" if oc.target == "entity_type" else "关系类型"
            console.print(
                f"    [{color}]{icon}[/{color}] [{color}]{oc.name}[/{color}] "
                f"[dim]({target_cn})[/dim]"
            )
            if oc.description:
                console.print(f"       {oc.description}")
            if oc.reason:
                console.print(f"       [dim]原因: {oc.reason}[/dim]")
        console.print()

    if plan.updated_ontology_raw:
        console.print(_("  [bold cyan]更新后的本体定义:[/bold cyan]"))
        preview = plan.updated_ontology_raw[:400].replace("\n", "\n  ")
        console.print(f"  [dim]{preview}[/dim]")
        if len(plan.updated_ontology_raw) > 400:
            console.print(f"  [dim]... ({len(plan.updated_ontology_raw)} 字符)[/dim]")
        console.print()

    # Strategy
    if plan.suggested_strategy:
        console.print(
            f"  [bold yellow]策略:[/bold yellow] {session.strategy} → "
            f"[bold]{plan.suggested_strategy}[/bold]"
        )
        console.print()

    # Tips
    if plan.extraction_tips:
        console.print(f"  [bold magenta]抽取提示:[/bold magenta]")
        console.print(f"  [dim]{plan.extraction_tips[:400]}[/dim]")
        console.print()

    if plan.prompt_additions:
        console.print(f"  [bold green]Prompt 增强:[/bold green] (+{len(plan.prompt_additions)} 条)")
        for tip in plan.prompt_additions[:3]:
            console.print(f"    [dim]+ {tip[:120]}[/dim]")
        console.print()

    # Toggles
    toggles = []
    if plan.enable_gleaning is not None:
        toggles.append(f"Gleaning: {'开启' if plan.enable_gleaning else '关闭'}")
    if plan.enable_co_occurrence is not None:
        toggles.append(f"共现图谱: {'开启' if plan.enable_co_occurrence else '关闭'}")
    if plan.suggested_chunk_size > 0:
        toggles.append(f"Chunk size: {plan.suggested_chunk_size}")
    if toggles:
        console.print(f"  [bold]参数调整:[/bold] {', '.join(toggles)}")
        console.print()

    # Ask for confirmation
    from rich.prompt import Confirm
    console.print()
    if Confirm.ask("  [bold]应用以上修改?[/bold]", default=True):
        changes = engine.apply(plan, session)
        console.print()
        console.print(_("  [green]✓ 优化方案已应用:[/green]"))
        if changes.get("ontology_updated"):
            console.print(_("    • 本体定义已更新"))
        if changes.get("strategy_changed"):
            console.print(f"    • 策略已切换为 {changes.get('new_strategy')}")
        if changes.get("gleaning_toggled"):
            console.print(f"    • Gleaning 已{changes.get('gleaning') and '开启' or '关闭'}")
        if changes.get("co_occurrence_toggled"):
            console.print(f"    • 共现图谱已{changes.get('co_occurrence') and '开启' or '关闭'}")
        if changes.get("chunk_size_changed"):
            console.print(f"    • Chunk size 已设为 {changes.get('chunk_size')}")
        if changes.get("tips_added"):
            console.print(_("    • 抽取提示已记录（下次构建生效）"))
        console.print()
        console.print(_("  [bold]输入 /run 使用优化后的配置重新构建。[/bold]"))
    else:
        console.print(_("  [dim]已取消。修改未应用。[/dim]"))


def _cmd_rollback(session: Session, arg: str):
    """回滚工作目录到之前的 git 版本。"""
    if not arg or not arg.strip():
        console.print(_("  [!] 用法: /rollback <commit_hash>"), style="yellow")
        console.print(_("  [dim]使用 /history 查看可用的版本哈希[/dim]"))
        return

    commit_hash = arg.strip()

    if not session.git.has_commits():
        console.print(_("  [!] 没有可回滚的版本"), style="yellow")
        return

    # Show what we're rolling back to
    history = session.git.get_history(20)
    target = next((h for h in history if h["hash"].startswith(commit_hash)), None)
    if not target:
        console.print(f"  [!] 未找到版本: {commit_hash}", style="yellow")
        console.print(_("  [dim]使用 /history 查看可用的版本[/dim]"))
        return

    console.print()
    console.print(f"  [yellow]⚠ 即将回滚到版本:[/yellow]")
    console.print(f"  [cyan]{target['hash']}[/cyan] {target['date']} {target['message']}")
    from rich.prompt import Confirm as _Confirm
    if not _Confirm.ask("  确认回滚？", default=False):
        console.print(_("  [dim]已取消[/dim]"))
        return

    if session.git.rollback(commit_hash):
        console.print(f"  [OK] 已回滚到 [bold]{commit_hash}[/bold]", style="green")
        console.print(_("  [dim]提示: 重新启动会话以加载回滚后的状态[/dim]"))
    else:
        console.print(f"  [FAIL] 回滚失败", style="red")


def _cmd_extract_entities(session: Session):
    # 自动加载（同 _cmd_run 逻辑）
    if not session.documents_loaded:
        loaded = False
        if session.discovered_files:
            existing = [p for p in session.discovered_files if Path(p).exists()]
            if existing:
                session.load_docs(existing)
                session.doc_paths = existing
                loaded = True
        if not loaded:
            from pathlib import Path as _Path
            from .loaders import supported_extensions
            cwd = _Path.cwd()
            data_files = []
            skip_dirs = SKIP_DIRS
            for ext in supported_extensions():
                for f in cwd.glob(f"**/*{ext}"):
                    parts = set(f.parts)
                    if not parts.intersection(skip_dirs):
                        data_files.append(str(f))
            if data_files:
                session.load_docs(data_files)
                session.doc_paths = data_files
                loaded = True

    if not session.documents_loaded:
        console.print(_("  [!] 请先加载文档: /load <路径>"), style="yellow")
        return

    console.print()
    console.print(_("  仅运行实体抽取..."), style=STYLE["dim"])
    console.print(
        "  [dim]提示: 按 [bold]Ctrl+O[/bold] 可实时切换详细消息流[/dim]"
    )

    llm_cfg2 = UserConfig.get_llm_config()
    config = HarnessConfig(
        llm=LLMConfig(
            model=session.model, api_key=session.api_key, api_base=session.api_base,
            temperature=float(llm_cfg2.get("temperature", 0.3)),
        ),
        work_dir=session.work_dir,
        verbose=True,
    )
    harness = Harness(config)
    progress_cb2, progress_stop2 = make_progress_callback(verbose_toggle=session.verbose_toggle)
    harness.on_event(progress_cb2)

    if session.doc_paths:
        harness.load_documents(session.doc_paths)
    if session.ontology_raw:
        harness.set_ontology(session.ontology_raw)

    session.verbose_toggle.start_listener()

    ontology_to_use2 = session.ontology_raw
    if not ontology_to_use2 and session.accumulated_context.strip():
        ontology_to_use2 = (
            "请根据文档内容自动推断合理的知识图谱本体。\n"
            f"提示: {session.accumulated_context[-2000:]}"
        )
    if ontology_to_use2:
        harness.set_ontology(ontology_to_use2)

    try:
        result = harness.run(
            ontology_raw=ontology_to_use2,
            skills=["ontology_analyzer", "entity_extractor"],
            strategy=session.strategy,
            enable_co_occurrence=session.enable_co_occurrence,
        )
        session.last_result = result
        print_extraction_result(result)
    except KeyboardInterrupt:
        console.print()
        console.print(_("  [yellow]^C 抽取已中断[/yellow]"))
    except Exception as e:
        console.print(f"  [FAIL] 抽取失败: {e}", style="red")
    finally:
        progress_stop2()
        session.verbose_toggle.stop_listener()
        _flush_stdin()


def _cmd_chat(session: Session, message: str):
    """流式 AI 对话，自动积累本体相关的上下文，并响应系统操作提案。"""
    if not message.strip():
        return

    console.print()
    proposed = None
    try:
        proposed = _stream_agent_response(session, message)
    finally:
        _flush_stdin()

    # 处理 propose_action 提案
    if proposed:
        action = proposed.get("action", "")
        reason = proposed.get("reason", "")
        if action:
            _handle_proposed_action(session, proposed)
        return  # 提案已处理，不继续积累上下文

    # 将用户的 chat 消息积累到上下文
    _accumulate_chat_context(session, message)


def _accumulate_chat_context(session: Session, message: str):
    """将 chat 消息积累到上下文。"""
    onto_keywords = ["实体", "关系", "类型", "本体", "entity", "relation", "type",
                     "图谱", "知识", "抽取", "人物", "公司", "组织", "地点", "事件",
                     "属性", "属性值", "三元组", "spo", "ner", "re"]
    is_relevant = (
        len(message) > 20 or
        any(kw in message.lower() for kw in onto_keywords)
    )
    if is_relevant:
        if session.accumulated_context:
            session.accumulated_context += "\n---\n"
        session.accumulated_context += f"[用户] {message[:500]}"


def _handle_proposed_action(session: Session, proposed: dict):
    """处理 Agent 提出的系统操作提案，询问用户确认后执行。"""
    action = proposed.get("action", "")
    path = proposed.get("path", "")
    definition = proposed.get("definition", "")
    reason = proposed.get("reason", "")

    action_labels = {
        "run": "运行完整 KG 构建流水线",
        "extract_entities": "仅运行实体抽取",
        "load": f"加载文档: {path}" if path else "加载文档",
        "ontology": f"设置本体: {definition[:60]}..." if definition else "设置本体",
        "quit": "退出程序",
    }
    label = action_labels.get(action, action)

    console.print()
    if reason:
        console.print(f"  [dim]原因: {reason}[/dim]")
    console.print(f"  [bold magenta]Agent 提议: {label}[/bold magenta]")

    _flush_stdin()  # 清空 stdin 防止残留字节干扰确认提示
    from rich.prompt import Confirm
    if Confirm.ask("  [bold]是否执行?[/bold]", default=True):
        console.print()
        if action == "run":
            _cmd_run(session)
        elif action == "extract_entities":
            _cmd_extract_entities(session)
        elif action == "load" and path:
            _cmd_load(session, path)
        elif action == "ontology" and definition:
            _cmd_ontology(session, definition)
        elif action == "quit":
            console.print(_("  再见！"))
            _ensure_cursor_visible()
            console.print()
            sys.exit(0)
        else:
            console.print(f"  [yellow][!] 无法执行: 缺少必要参数[/yellow]")
    else:
        console.print(_("  [dim]已取消[/dim]"))


def _stream_agent_response(session: Session, message: str):
    """流式输出 Agent 回复，使用 Rich Live + Markdown 实时渲染。

    在等待 LLM 回复/工具执行时显示旋转动画指示器。
    拦截 propose_action 工具调用，转为系统操作提案。

    返回: Optional[dict] — 如果是 propose_action，返回操作信息；否则返回 None
    """
    agent = session.chat_agent

    current_text = ""
    tool_calls_log: list[str] = []
    is_thinking = True
    proposed_action: Optional[dict] = None
    last_tool_name: str = ""  # track last tool for result interception

    from rich.markdown import Markdown
    from rich.live import Live
    from rich.padding import Padding
    from rich.spinner import Spinner

    spinner = Spinner("dots", text="思考中...", style="dim cyan")

    def _render():
        parts = []
        for tc in tool_calls_log:
            # Text.from_markup 解析 Rich 标记语法 ([bold], [dim], [green] 等)
            parts.append(Text.from_markup(tc))
        if is_thinking:
            parts.append(Padding(spinner, (0, 2)))
        if current_text.strip():
            parts.append(Padding(Markdown(current_text, code_theme="monokai"), (0, 2)))
        return Group(*parts) if parts else Text("")

    with Live(_render(), console=console, refresh_per_second=10, transient=False) as live:
        try:
            for event_type, data in agent.run_stream(message):
                if event_type == "thinking":
                    pass

                elif event_type == "token":
                    if is_thinking:
                        is_thinking = False
                    current_text += data["text"]

                elif event_type == "tool_call":
                    tool_name = data.get("tool", "?")
                    args = data.get("args", {})

                    # 记录 read_file 读取的数据文件路径（后续 /run 时自动加载）
                    if tool_name == "read_file":
                        path = args.get("path", "")
                        data_exts = ('.txt', '.jsonl', '.docx', '.md', '.csv')
                        if path and any(path.endswith(ext) for ext in data_exts):
                            if path not in session.discovered_files:
                                session.discovered_files.append(path)

                    # 拦截 propose_action — 不执行工具，转为操作提案
                    if tool_name == "propose_action":
                        proposed_action = {
                            "action": args.get("action", ""),
                            "path": args.get("path", ""),
                            "definition": args.get("definition", ""),
                            "reason": args.get("reason", ""),
                        }
                        tool_calls_log.append(
                            f"  [bold magenta]>> 提议: {proposed_action['action']}[/bold magenta]"
                            f" [dim]— {proposed_action.get('reason', '')[:80]}[/dim]"
                        )
                        is_thinking = False
                        live.update(_render())
                        # 立即返回，不再继续流式输出
                        live.stop()
                        console.print()
                        return proposed_action

                    from rich.markup import escape
                    args_str = escape(str(args)[:120])
                    tool_calls_log.append(
                        f"  [bold yellow]>> {tool_name}[/bold yellow] [dim]{args_str}[/dim]"
                    )
                    is_thinking = True
                    last_tool_name = tool_name

                elif event_type == "tool_result":
                    success = data.get("success", False)
                    from rich.markup import escape
                    result_preview = escape(str(data.get("result", ""))[:200])
                    icon = "[green]OK[/green]" if success else "[red]FAIL[/red]"
                    tool_calls_log.append(
                        f"  [dim]>> {icon} {result_preview}[/dim]"
                    )
                    is_thinking = False

                    # 拦截 extract_with_code 结果，存入 session 供 /run 使用
                    if last_tool_name in ("extract_with_code", "extract_with_llm_prompt") and success:
                        result_data = data.get("result", "")
                        try:
                            import json as _json
                            parsed = _json.loads(result_data) if isinstance(result_data, str) else result_data
                            ents = parsed.get("entities", [])
                            rels = parsed.get("relations", [])
                            if ents or rels:
                                session.pre_extracted_entities.extend(ents)
                                session.pre_extracted_relations.extend(rels)
                                console.print(
                                    f"  [dim]>> 预提取: {len(ents)} 实体, {len(rels)} 关系 "
                                    f"(已缓存，/run 时将合并)[/dim]"
                                )
                        except Exception:
                            pass

                elif event_type == "error":
                    from rich.markup import escape
                    tool_calls_log.append(
                        f"  [red][FAIL] {escape(str(data.get('message', ''))[:200])}[/red]"
                    )
                    is_thinking = False

                elif event_type == "done":
                    is_thinking = False

                live.update(_render())

        except Exception as e:
            live.update(Text(f"\n  [red]流式输出错误: {e}[/red]"))

    _ensure_cursor_visible()
    console.print()
    return None


# ─── 入口 ────────────────────────────────────────────────────────────────────

def start_interactive(api_key: str, api_base: str, model: str, work_dir: str = ".kgclaw"):
    """启动交互式 REPL。"""
    from .logger import get_logger
    get_logger().info("Interactive session started", model=model, api_base=api_base, work_dir=work_dir)
    try:
        run_interactive(api_key, api_base, model, work_dir=work_dir)
    finally:
        _restore_terminal()
        console.print()
    get_logger().info("Interactive session ended")
