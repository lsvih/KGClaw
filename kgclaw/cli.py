"""
KGClaw CLI — 基于 Agent Harness 的知识图谱构建命令行工具

使用方式:
    kgclaw setup                   # 首次运行：交互式配置 LLM 连接
    kgclaw run -t "本体定义" -d 文档.txt   # 命令行模式
    kgclaw interactive             # 交互式引导模式
    kgclaw examples                # 查看内置示例

配置文件: ~/.kgclaw/config.yaml
环境变量覆盖: OPENAI_API_KEY, KGCLAW_MODEL, KGCLAW_API_BASE
"""

from __future__ import annotations

import io
import json
import os
import traceback
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich import box

# ── 修复 stdin 编码，确保中文输入正常 ─────────────────────────────────────────

def _fix_stdin_encoding():
    """修复 stdin 编码问题，确保中文输入正常。

    在某些 Linux 终端环境下（如 Docker 容器、musl 系统、最小化安装），
    sys.stdin.encoding 可能不是 UTF-8（如 'ascii'、'ANSI_X3.4-1968' 或 None），
    导致 input() 无法解码中文输入，抛出 UnicodeDecodeError。

    此函数：
    1. 检测编码是否为 UTF-8
    2. 如果不是，用 locale 编码重新包装 stdin
    3. 如果 locale 也不是 UTF-8，则强制使用 UTF-8
    """
    current = (sys.stdin.encoding or '').lower()
    if current in ('utf-8', 'utf8'):
        return  # already OK

    # 优先尝试 locale 的编码
    import locale
    try:
        loc_enc = locale.getpreferredencoding()
        if loc_enc.lower() in ('utf-8', 'utf8'):
            loc_enc = 'utf-8'
    except Exception:
        loc_enc = 'utf-8'

    try:
        sys.stdin = io.TextIOWrapper(
            sys.stdin.buffer,
            encoding=loc_enc,
            errors='replace',
        )
    except (AttributeError, OSError, ValueError):
        # 极其边缘的情况：没有 buffer 或编码不支持
        # 尝试强制 UTF-8
        try:
            sys.stdin = io.TextIOWrapper(
                sys.stdin.buffer,
                encoding='utf-8',
                errors='surrogateescape',
            )
        except Exception:
            pass

# NOTE: _fix_stdin_encoding() is called explicitly from main() and the
# interactive entry point, not at module import time, to avoid side effects
# when importing kgclaw as a library (e.g., `from kgclaw import Harness`).
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .config import UserConfig
from .harness import Harness
from .models import HarnessConfig, LLMConfig
from .i18n import _
from .ui import (
    ICON,
    STYLE,
    make_progress_callback,
    print_banner,
    print_error,
    print_section,
    print_stats,
    print_success,
    print_warning,
)

console = Console()


# ─── 配置解析 ────────────────────────────────────────────────────────────────

def resolve_llm_config(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
) -> dict[str, str]:
    """解析 LLM 配置：CLI 参数 > 环境变量 > 配置文件 > 默认值。"""
    saved = UserConfig.get_llm_config()
    return {
        "api_key": api_key or saved.get("api_key", ""),
        "model": model or saved.get("model", "deepseek-v4-flash"),
        "api_base": api_base or saved.get("api_base", "https://api.deepseek.com/v1"),
    }


def ensure_configured(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
) -> tuple[str, str, str]:
    """确保已完成配置。如果未配置 API Key，引导用户进入 setup 流程。"""
    resolved = resolve_llm_config(api_key, model, api_base)

    if not resolved["api_key"]:
        console.print()
        panel = Panel(
            Group(
                Text(_(" 欢迎使用 KGClaw！"), style=STYLE["banner"], justify="center"),
                Text(""),
                Text(_("看起来你是第一次使用，需要先配置 LLM 服务连接。"), justify="center"),
                Text(_("这只需要 1 分钟，我会一步步引导你完成。"), style=STYLE["dim"], justify="center"),
            ),
            box=box.ROUNDED,
            border_style="cyan",
            padding=(1, 2),
        )
        console.print(panel)

        if Confirm.ask("\n   是否现在开始配置?", default=True):
            return run_setup_wizard()
        else:
            console.print()
            console.print(f"  {ICON['bullet']} 你可以稍后运行 [bold]kgclaw setup[/bold] 来配置", style=STYLE["dim"])
            console.print(f"  {ICON['bullet']} 也可以通过环境变量设置: [bold]OPENAI_API_KEY[/bold], [bold]KGCLAW_MODEL[/bold], [bold]KGCLAW_API_BASE[/bold]", style=STYLE["dim"])
            console.print()
            sys.exit(1)

    return resolved["api_key"], resolved["api_base"], resolved["model"]


# ─── Setup 向导 ──────────────────────────────────────────────────────────────

# 模型的示例 / 推荐场景，方便用户参考
MODEL_SUGGESTIONS = {
    "openai": [
        ("gpt-4o", "综合能力最强，适合复杂的知识抽取任务"),
        ("gpt-4o-mini", "性价比首选，适合大批量文本处理"),
        ("gpt-4.1", "最新旗舰模型"),
    ],
    "deepseek": [
        ("deepseek-chat", "DeepSeek-V3，中文能力强，性价比极高"),
        ("deepseek-reasoner", "DeepSeek-R1，推理增强，适合复杂的本体分析"),
    ],
    "ollama": [
        ("llama3.1:8b", "Meta Llama 3.1 8B，本地运行，无需联网"),
        ("qwen2.5:14b", "通义千问 2.5 14B，中文能力优秀"),
        ("qwen2.5:32b", "通义千问 2.5 32B，中文能力更强，需要更多显存"),
    ],
    "custom": [
        ("gpt-4o", "OpenAI 兼容 API 的通用选择"),
        ("deepseek-chat", "如果你的 API 支持 DeepSeek 模型"),
    ],
}

# 常用中文知识图谱本体模板
ONTOLOGY_TEMPLATES = {
    "1": {
        "name": "人物关系图谱",
        "template": """## Entity Types（实体类型）
- 人物 (Person): 自然人或虚构角色

## Relation Types（关系类型）
- 生父 (biological_father): 人物的生物学父亲
- 生母 (biological_mother): 人物的生物学母亲
- 儿子 (son): 人物的儿子
- 女儿 (daughter): 人物的女儿
- 哥哥 (elder_brother): 人物的兄长
- 弟弟 (younger_brother): 人物的弟弟
- 姐姐 (elder_sister): 人物的姐姐
- 现妻 (current_wife): 人物的现任妻子
- 现夫 (current_husband): 人物的现任丈夫
- 老师 (teacher): 人物的老师/师傅
- 朋友 (friend): 人物的朋友""",
        "example_text": "赵铁蛋是赵本山的儿子，是个聋哑儿。\n马志明，相声名家，相声泰斗马三立先生长子。\n洪剑涛的妻子是冉志娟，92年，他们的儿子洪洋出生。",
    },
    "2": {
        "name": "企业知识图谱",
        "template": """## Entity Types（实体类型）
- 公司 (Company): 企业法人
- 人物 (Person): 自然人
- 产品 (Product): 公司生产或提供的产品/服务
- 行业 (Industry): 公司所属行业领域

## Relation Types（关系类型）
- 创始人 (founder): 人物 → 公司
- CEO (ceo): 人物 → 公司
- 投资方 (investor): 公司 → 公司
- 竞争对手 (competitor): 公司 → 公司
- 生产 (produces): 公司 → 产品
- 属于行业 (in_industry): 公司 → 行业""",
        "example_text": "2024年，字节跳动创始人张一鸣卸任CEO，由梁汝波接任。\nOpenAI发布了GPT-4o模型，与Google的Gemini展开竞争。",
    },
    "3": {
        "name": "法律法规知识图谱",
        "template": """## Entity Types（实体类型）
- 法案 (Bill): 立法文件
- 议员 (Member): 立法机构成员
- 委员会 (Committee): 立法委员会
- 议题 (Topic): 法案涉及的政策议题

## Relation Types（关系类型）
- 发起人 (sponsor): 议员 → 法案
- 支持者 (cosponsor): 议员 → 法案
- 反对者 (opponent): 议员 → 法案
- 提交至 (referred_to): 法案 → 委员会
- 涉及议题 (about_topic): 法案 → 议题""",
        "example_text": "H.R.649 号法案由众议员 Thompson 发起，旨在修改《儿童营养法案》。\n该法案获得了来自两党的 15 名议员联署支持。",
    },
    # ─── Evaluation dataset presets (4-8) ───────────────────────────────────
    "4": {
        "name": "WebNLG (DBpedia)",
        "template": """## Entity Types
- Entity: Any named entity, concept, location, person, organization, number, or date

## Relation Types
- creator: The subject entity is the creator/author of the object
- broadcastedBy: The subject entity is broadcasted or aired by the object entity
- firstAired: The subject entity is first aired on the date specified by the object entity
- established: The subject entity was established or created in the year specified by the object entity
- location: The subject entity is located in the place specified by the object entity
- municipality: The subject entity is in the municipality specified by the object entity
- category: The subject entity belongs to the category specified by the object entity
- country: The subject entity is located in the country specified by the object entity
- governmentType: The subject entity has the government type specified by the object entity
- architect: The subject entity was designed/architected by the object entity
- owner: The subject entity is owned by the object entity
- operator: The subject entity is operated by the object entity
- runwayLength: The subject entity has a runway length of the object value
- elevationAboveTheSeaLevel: The subject entity's elevation above sea level is the object value
- populationTotal: The subject entity has a total population of the object value
- leaderTitle: The subject entity's leader title is the object value
- (and 143 more DBpedia relation types — see webnlg_schema.csv)""",
        "example_text": "The Enaire operated Adolfo Suarez Madrid-Barajas Airport is based in Paracuellos de Jarama.",
    },
    "5": {
        "name": "REBEL (Wikidata relations)",
        "template": """## Entity Types
- Entity: Any named entity, concept, location, person, organization, number, or date

## Relation Types
- country: The subject entity is located in the country specified by the object entity
- contains administrative territorial entity: The subject contains the administrative territorial entity specified by the object
- contains settlement: The subject contains the settlement specified by the object
- located in the administrative territorial entity: The subject is located in the administrative territorial entity
- date of birth: The subject was born on the date specified by the object
- place of birth: The subject was born in the place specified by the object
- date of death: The subject died on the date specified by the object
- place of death: The subject died in the place specified by the object
- start time: The subject started at the time specified by the object
- location: The subject is located at the object
- (and 186 more Wikidata property-based relation types — see rebel_schema.csv)""",
        "example_text": "Spodnje Hoce is a settlement in the Municipality of Hoce-Slivnica in northeastern Slovenia.",
    },
    "6": {
        "name": "Wiki-NRE (Wikidata relations)",
        "template": """## Entity Types
- Entity: Any named entity, concept, location, person, organization, number, or date

## Relation Types
- located in the administrative territorial entity: Subject located within an administrative area
- country: Subject located in a country
- director: Subject was directed by object
- cast member: Subject includes object as a cast member
- screenwriter: Subject was written by object
- part of: Subject is part of object
- located on terrain feature: Subject located on a terrain feature
- original language of film or TV show: Subject's original language is object
- employer: Subject is employed by object
- place served by transport hub: Subject is served by transport hub object
- named after: Subject is named after object
- owned by: Subject is owned by object
- country of citizenship: Subject has citizenship of object
- participant in: Subject participated in object
- place of birth: Subject was born at object
- (and 30 more Wikidata relation types — see wiki-nre_schema.csv)""",
        "example_text": "Lyseren is a lake in the municipalities of Enebakk in Akershus county and Spydeberg in Ostfold county, Norway.",
    },
    "7": {
        "name": "SREDFM (Multilingual RE)",
        "template": """## Entity Types
- Concept: Abstract concept (taxon, species, etc.)
- LOC: Geographic location
- DATE: Date or time value
- NUMBER: Numeric value
- PER: Person
- ORG: Organization or institution
- MEDIA: Media work (film, book, album, etc.)
- EVE: Event
- MISC: Miscellaneous entity
- CEL: Celestial body
- TIME: Time duration or interval
- DIS: Disease or medical condition
- UNK: Unknown entity type

## Relation Types
- country: Subject is located in a country
- located in the administrative territorial entity: Subject is in an administrative area
- instance of: Subject is an instance of a class
- sport: Subject is a sport
- point in time: Subject occurred at a point in time
- part of: Subject is part of a larger whole
- date of birth: Subject's date of birth
- publication date: Subject's publication date
- inception: Subject's inception/founding date
- cast member: Subject features object as a cast member
- (and 190 more Wikidata property relation types)""",
        "example_text": "Miss South Africa 2011 was held on 11 December 2011 in Sun City, South Africa.",
    },
    "8": {
        "name": "KoCHET (Korean Cultural Heritage)",
        "template": """## Entity Types
- ARTIFACTS: Building, craft, documents, historical sites, monument, musical instrument, etc.
- PERSON: Mythical figure, name, position
- TERM: Color, mark, shape, technique
- DATE: Day, duration, dynasty, geo-age, month, season, year
- POLITICAL_LOCATION: Capital city, city, country, county, province
- CIVILIZATION: Building type, clothing, currency, drink, food, language, law, etc.
- MATERIAL: Bone, fiber, grass, jewelry, metal, paper, rock, wood, etc.
- LOCATION: Space, others
- ANIMAL: Amphibian, bird, fish, insect, mammal, reptile
- PLANT: Flower, fruit, grass, tree
- GEOGRAPHICAL_LOCATION: Bay, continent, island, mountain, ocean, river
- EVENT: Activity, festival, sports, war/revolution

## Relation Types (Korean text, use these EXACT relation names)
- OriginatedIn: Subject originated in/from the object (origin/location provenance)
- consistsOf: Subject consists of the object (composition/part-whole)
- depicts: Subject depicts the object (representation)
- documents: Subject documents the object
- fallsWithin: Subject falls within the object (spatial/temporal containment)
- hasCarriedOut: Subject has carried out the object (performer/agent of action)
- hasCreated: Subject has created the object
- hasDestroyed: Subject has destroyed the object
- hasSection: Subject has the object as a section
- hasTime: Subject has the object as its time
- isConnectedWith: Subject is connected with the object
- isUsedIn: Subject is used in the object
- servedAs: Subject served as the object (function/role)
- wears: Subject wears the object (attire/adornment)""",
        "example_text": "1906년 4월 11일 이생원댁 노비 정금이 정생원댁 노비 결득에게 홍산군 해안면 망하리 서쌍동에 있는 논을 팔고 작성한 문서.",
    },
}


def run_setup_wizard() -> tuple[str, str, str]:
    """交互式首次配置向导。引导用户完成 4 步 LLM 配置。"""

    # ── 欢迎 ──
    console.print()
    welcome = Panel(
        Group(
            Text(_("欢迎使用 KGClaw！"), style=Style(color="bright_cyan", bold=True), justify="center"),
            Text(""),
            Text(_("KGClaw 是一个基于 AI Agent Harness 的知识图谱构建工具。"), justify="center"),
            Text('你可以像与 AI 编程助手对话一样，「自然语言描述本体 + 提供文档」，', justify="center"),
            Text(_("它就会自动抽取并构建结构化的知识图谱。"), justify="center"),
            Text(""),
            Text(_("在开始之前，我们先花 1 分钟配置 LLM 服务连接。"), style=STYLE["muted"], justify="center"),
        ),
        box=box.ROUNDED,
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(welcome)

    # ── Step 1: 选择服务商 ──
    console.print()
    console.print(Rule(_("Step 1/4 · 选择 LLM 服务商"), style=STYLE["muted"]))
    console.print()

    providers = [
        ("OpenAI", "api.openai.com", "全球领先的 AI 服务商，GPT-4o 综合能力最强"),
        ("DeepSeek", "api.deepseek.com", "国产高性价比 API，中文能力强，价格实惠"),
        ("Ollama (本地)", "localhost:11434", "完全本地运行，无需联网，零成本，隐私安全"),
        ("自定义", "任意 OpenAI 兼容 API", "支持 vLLM、通义千问、智谱等任何兼容接口"),
    ]

    for i, (name, endpoint, desc) in enumerate(providers, 1):
        idx = f"[{i}]"
        console.print(
            f"  {idx} [bold cyan]{name}[/bold cyan]"
        )
        console.print(f"       {desc}", style=STYLE["dim"])
        console.print(f"       {endpoint}", style=STYLE["muted"])

    console.print()
    console.print(f"  [5] [dim]跳过（使用环境变量 OPENAI_API_KEY 配置）[/dim]")
    console.print()

    choice = Prompt.ask(
        "  请选择服务商",
        choices=["1", "2", "3", "4", "5"],
        default="1",
    )

    if choice == "5":
        console.print(_("  [skip]️  已跳过。请通过环境变量 OPENAI_API_KEY 等配置后重新运行。"), style=STYLE["dim"])
        return ("", os.environ.get("KGCLAW_API_BASE", "https://api.deepseek.com/v1"), os.environ.get("KGCLAW_MODEL", "deepseek-v4-flash"))

    provider_map = {
        "1": ("openai", "https://api.openai.com/v1"),
        "2": ("deepseek", "https://api.deepseek.com/v1"),
        "3": ("ollama", "http://localhost:11434/v1"),
        "4": ("custom", ""),
    }
    provider_key, api_base = provider_map[choice]

    if choice == "4":
        api_base = Prompt.ask("  请输入 API Base URL", default="https://api.openai.com/v1")

    provider_names = {"openai": "OpenAI", "deepseek": "DeepSeek", "ollama": "Ollama (本地)", "custom": "自定义"}
    console.print(f"  [OK] 已选择 [bold]{provider_names[provider_key]}[/bold]", style="green")
    console.print(f"     API 地址: [dim]{api_base}[/dim]", style=STYLE["muted"])

    # ── Step 2: 输入 API Key ──
    console.print()
    console.print(Rule(_("Step 2/4 · 输入 API Key"), style=STYLE["muted"]))
    console.print()

    if provider_key == "ollama":
        console.print(_("  Ollama 本地服务通常不需要 API Key。"), style=STYLE["dim"])
        console.print(_("  如果你在 Ollama 中配置了认证，请输入你的 Key；否则直接回车即可。"), style=STYLE["dim"])
        console.print()
        api_key = Prompt.ask("  API Key", default="ollama", password=False)
    else:
        console.print(f"  密钥将被安全保存在 [bold]~/.kgclaw/config.yaml[/bold]", style=STYLE["dim"])
        console.print(f"  获取地址: ", style=STYLE["dim"], end="")
        if provider_key == "openai":
            console.print("https://platform.openai.com/api-keys", style="cyan underline")
        elif provider_key == "deepseek":
            console.print("https://platform.deepseek.com/api_keys", style="cyan underline")
        console.print()
        api_key = Prompt.ask("  API Key", password=True)

    if not api_key:
        console.print(_("  [!]  未输入 API Key，将尝试使用环境变量 OPENAI_API_KEY"), style="yellow")
    else:
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        console.print(f"  [OK] API Key 已设置 ([dim]{masked}[/dim])", style="green")

    # ── Step 3: 输入模型名称 ──
    console.print()
    console.print(Rule(_("Step 3/4 · 输入默认模型名称"), style=STYLE["muted"]))
    console.print()

    # Sensible defaults per provider
    _default_models = {
        "openai": "deepseek-v4-flash",
        "deepseek": "deepseek-v4-flash",
        "ollama": "qwen2.5:14b",
        "custom": "deepseek-v4-flash",
    }
    default_model = _default_models.get(provider_key, "deepseek-v4-flash")
    console.print(_("  输入你想使用的模型名称（如 deepseek-v4-flash, deepseek-chat, qwen2.5:14b 等）。"), style=STYLE["dim"])
    console.print(_("  运行时可通过 [bold]-m[/bold] 参数临时切换。"), style=STYLE["dim"])
    console.print()
    model = Prompt.ask("  模型名称", default=default_model)

    console.print(f"  [OK] 默认模型: [bold]{model}[/bold]", style="green")

    # ── Step 4: 语言选择 + 确认保存 ──
    console.print()
    console.print(Rule(_("Step 4/4 · 语言设置与确认"), style=STYLE["muted"]))
    console.print()

    console.print(_("  请选择界面语言 / Choose UI language:"))
    console.print(f"  [1] 中文")
    console.print(f"  [2] English")
    console.print()
    lang_choice = Prompt.ask(_("  语言 / Language"), choices=["1", "2"], default="2")
    lang = "en" if lang_choice == "2" else "zh"
    lang_display = "English" if lang == "en" else "中文"
    console.print(f"  [OK] 界面语言: [bold]{lang_display}[/bold]", style="green")
    console.print()

    # 确认保存
    masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    preview = Table(box=box.ROUNDED, show_header=False, border_style="cyan")
    preview.add_column(_("项目"), style="bold cyan", width=14)
    preview.add_column("值", style="white")
    preview.add_row(_("服务商"), provider_names[provider_key])
    preview.add_row(_("API 地址"), api_base)
    preview.add_row(_("默认模型"), model)
    preview.add_row("API Key", masked)
    preview.add_row(_("界面语言"), lang_display)
    console.print(preview)
    console.print()

    if not Confirm.ask("   确认保存以上配置?", default=True):
        console.print()
        console.print(_("  [yellow]已取消。可稍后运行 [bold]kgclaw setup[/bold] 重新配置。[/yellow]"))
        sys.exit(0)

    # 保存到文件
    config = UserConfig.load()
    config["llm"]["api_key"] = api_key
    config["llm"]["api_base"] = api_base
    config["llm"]["model"] = model
    config["llm"]["provider"] = provider_key
    config["preferences"]["lang"] = lang
    UserConfig.save(config)

    console.print()
    console.print(f"  [OK] 配置已保存到 [bold]{UserConfig.config_path()}[/bold]", style="green")
    console.print()

    # 提示下一步
    next_steps = Panel(
        Group(
            Text(_("  配置完成！现在你可以："), style=STYLE["success"]),
            Text(""),
            Text(_("  • 运行 [bold]kgclaw run[/bold] 开始构建知识图谱"), style="white"),
            Text(_("  • 运行 [bold]kgclaw interactive[/bold] 进入交互式引导模式"), style="white"),
            Text(_("  • 运行 [bold]kgclaw examples[/bold] 查看内置示例数据"), style="white"),
            Text(_("  • 运行 [bold]kgclaw setup[/bold] 随时修改配置"), style=STYLE["dim"]),
            Text(""),
            Text(_("Tip: 示例命令（可以直接复制运行）："), style=STYLE["heading"]),
            Text(""),
            Text(
                "  kgclaw run "
                "--ontology \"Entity Types: 人物\\nRelation Types: 生父, 儿子, 老师\" "
                "--docs examples/人物图谱/人物关系图谱原始数据.txt",
                style="bright_black",
            ),
        ),
        box=box.ROUNDED,
        border_style="green",
        padding=(1, 2),
    )
    console.print(next_steps)
    console.print()

    return api_key, api_base, model




# ─── CLI 命令 ────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="kgclaw")
@click.option(
    "--lang", default=None,
    help="界面语言 / UI language (zh, en). Default: en.",
    metavar="LANG",
)
def main(lang: str | None = None):
    """[kgclaw] KGClaw — 基于 AI Agent Harness 的知识图谱构建工具。

    通过自然语言描述本体，结合非结构化文档，自动抽取并构建知识图谱。
    """
    _fix_stdin_encoding()
    from .i18n import init_locale
    init_locale(lang)


# ── setup ────────────────────────────────────────────────────────────────────

@main.command()
def setup():
    """  交互式首次配置向导。

    引导你配置 LLM 服务连接（服务商、API Key、默认模型），
    配置保存在 ~/.kgclaw/config.yaml。

    支持的服务商：
      - OpenAI (GPT-4o 等)
      - DeepSeek (V3 / R1)
      - Ollama (本地 Llama / Qwen)
      - 任意 OpenAI 兼容 API
    """
    print_banner()
    run_setup_wizard()


# ── config ───────────────────────────────────────────────────────────────────

@main.command()
@click.option("--show-key", is_flag=True, help="显示完整的 API Key（默认隐藏）")
def config(show_key: bool):
    """ 查看当前配置。

    显示已保存的 LLM 连接配置和用户偏好设置。
    API Key 默认隐藏，使用 --show-key 可显示完整密钥。
    """
    from .config import UserConfig

    print_banner()

    if not UserConfig.exists():
        panel = Panel(
            Group(
                Text(_("还没有配置文件"), style=STYLE["warning"]),
                Text(""),
                Text(_("运行 [bold]kgclaw setup[/bold] 完成首次配置"), style="white"),
            ),
            box=box.ROUNDED,
            border_style="yellow",
        )
        console.print(panel)
        return

    saved = UserConfig.load()
    llm = saved.get("llm", {})
    prefs = saved.get("preferences", {})

    table = Table(
        title="当前配置",
        title_style=STYLE["heading"],
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
    )
    table.add_column(_("设置项"), style="bold cyan", width=16)
    table.add_column(_("当前值"), style="white")
    table.add_column(_("说明"), style=STYLE["dim"])

    table.add_row(_("配置文件"), str(UserConfig.config_path()), _("配置存储位置"))
    table.add_row(_("服务商"), llm.get("provider", "openai"), _("LLM 服务提供商"))
    table.add_row(_("API 地址"), llm.get("api_base", "N/A"), _("API 端点 URL"))

    if show_key:
        table.add_row("API Key", llm.get("api_key", "N/A"), _("[!] 敏感信息"))
    else:
        key = llm.get("api_key", "")
        masked = key[:8] + "..." + key[-4:] if len(key) > 12 else ("***" if key else "未设置")
        table.add_row("API Key", masked, _("使用 --show-key 显示完整密钥"))

    table.add_row(_("默认模型"), llm.get("model", "N/A"), _("可通过 -m 参数临时切换"))
    table.add_row("Temperature", str(llm.get("temperature", 0.3)), _("生成随机性（0-2）"))
    table.add_row(_("输出格式"), prefs.get("output_format", "nt"), "nt / json / jsonl")
    table.add_row(_("分块大小"), str(prefs.get("chunk_size", 2000)), _("文本分块字符数"))

    console.print(table)
    console.print()
    console.print(f"  运行 [bold]kgclaw setup[/bold] 修改配置", style=STYLE["dim"])


# ── run ──────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--ontology", "-t", help="本体定义文本。自然语言或结构化描述实体类型和关系类型")
@click.option("--ontology-file", "-O", type=click.Path(exists=True), help="本体定义文件路径（支持 .yaml / .json / .txt）")
@click.option("--docs", "-d", multiple=True, type=click.Path(exists=True), help="待处理的文档文件（可多次指定）")
@click.option("--docs-dir", "-D", type=click.Path(exists=True), help="待处理的文档目录（递归搜索 .txt / .md / .jsonl / .docx）")
@click.option("--output", "-o", "output_path", type=click.Path(), default=".kgclaw/output.nt", help="输出文件路径（默认 .kgclaw/output.nt）")
@click.option("--format", "-f", "output_format", type=click.Choice(["nt", "json", "jsonl"]), default="nt", help="输出格式（默认 nt）")
@click.option("--model", "-m", default=None, help="覆盖已保存的默认模型")
@click.option("--api-key", default=None, help="覆盖已保存的 API Key")
@click.option("--api-base", default=None, help="覆盖已保存的 API Base URL")
@click.option("--verbose", "-v", is_flag=True, help="详细输出（显示完整工具调用和 SubAgent 日志）")
@click.option("--debug", is_flag=True, help="调试模式：所有细节写入 .kgclaw/logs/kgclaw.log")
@click.option("--trace", is_flag=True, help="跟踪模式：所有 LLM 交互完整记录到 .kgclaw/traces/（JSONL 格式）")
@click.option("--quiet", "-q", is_flag=True, help="静默模式：减少控制台输出")
@click.option("--work-dir", default=".kgclaw", help="中间文件工作目录（默认 .kgclaw）")
@click.option("--template", "-T", "template_key", default=None, help="使用内置本体模板（1=人物关系, 2=企业, 3=法律法规, 4=WebNLG, 5=REBEL, 6=Wiki-NRE, 7=SREDFM, 8=KoCHET）")
@click.option("--strategy", "-s", "strategy", type=click.Choice(["auto", "fast", "standard", "code"]), default="auto", help="工作流策略：auto=自动选择, fast=快速合并抽取, standard=标准多阶段流水线, code=代码沙盒抽取")
@click.option("--concurrency", "-c", "concurrency", type=int, default=None, help="最大并行 Agent 数（默认 8，范围 1-64）")
@click.option("--chunk-size", "chunk_size_opt", type=int, default=None, help="文本分块大小（默认 4000 字符）")
@click.option("--co-occurrence/--no-co-occurrence", default=True, help="是否构建共现图谱（默认开启）")
@click.option("--resume", is_flag=True, help="继续之前中断的构建会话")
@click.option("--force", "force_run", is_flag=True, help="强制重建，跳过变更检测（与 --resume 配合使用）")
def run(
    ontology: Optional[str],
    ontology_file: Optional[str],
    docs: tuple[str, ...],
    docs_dir: Optional[str],
    output_path: str,
    output_format: str,
    model: Optional[str],
    api_key: Optional[str],
    api_base: Optional[str],
    verbose: bool,
    debug: bool,
    trace: bool,
    quiet: bool,
    work_dir: str,
    template_key: Optional[str],
    strategy: str = "auto",
    concurrency: Optional[int] = None,
    chunk_size_opt: Optional[int] = None,
    co_occurrence: bool = True,
    resume: bool = False,
    force_run: bool = False,
):
    """ 运行知识图谱构建流水线。

    根据给定的本体定义，从非结构化文档中抽取实体和关系，
    经过质量审核后，输出结构化的知识图谱。

    \\b
    示例：
      # 使用内置模板快速开始
      kgclaw run -T 1 -d examples/人物图谱/人物关系图谱原始数据.txt

      # 命令行直接指定本体
      kgclaw run -t "Entity Types: 人物\\\\nRelation Types: 生父, 儿子, 老师" -d docs.txt

      # 从文件加载本体，处理整个目录
      kgclaw run -O ontology.yaml -D my_documents/ -f json

      # 使用本地 Ollama 模型
      kgclaw run -t "实体: 人物, 地点\\\\n关系: 居住于" -d docs.txt -m qwen2.5:14b
    """
    print_banner()

    # 确保已配置
    api_key, api_base, model = ensure_configured(api_key, model, api_base)

    # 处理内置模板
    if template_key and not ontology and not ontology_file:
        # First try numeric template keys
        tmpl = ONTOLOGY_TEMPLATES.get(template_key)
        if tmpl:
            ontology = tmpl["template"]
            console.print(f"  ONT 使用内置本体模板: [bold]{tmpl['name']}[/bold]")
            console.print(f"     提示: 你也可以用自然语言描述自己的本体", style=STYLE["dim"])
        else:
            # Try dataset preset from kgclaw.presets
            try:
                from kgclaw.presets import get_preset
                preset = get_preset(template_key)
                if preset:
                    from kgclaw.models import Ontology
                    onto = Ontology(
                        name=preset.name,
                        entity_types=preset.entity_types,
                        relation_types=preset.relation_types,
                    )
                    ontology = onto.to_extraction_guide()
                    console.print(f"  ONT 使用数据集预设: [bold]{preset.display_name}[/bold]")
                    console.print(f"     [{len(preset.entity_types)} 实体类型, {len(preset.relation_types)} 关系类型]", style=STYLE["dim"])
            except ImportError:
                pass
            if not ontology:
                console.print(f"  [yellow][!]  未知模板 '{template_key}'，可用: 1-8 或 preset 名称[/yellow]")

    # 验证输入
    if not ontology and not ontology_file:
        console.print()
        panel = Panel(
            Group(
                Text(_("[!] 未指定本体定义"), style=STYLE["warning"], justify="center"),
                Text(""),
                Text(_("可以通过以下方式指定："), style="white"),
                Text(_("  * -t 参数直接传入本体文本"), style="white"),
                Text(_("  * -O 参数指定本体文件路径"), style="white"),
                Text(_("  * -T 参数使用内置模板 (1=人物关系, 2=企业, 3=法律法规, 4=WebNLG, 5=REBEL, 6=Wiki-NRE, 7=SREDFM, 8=KoCHET)"), style="white"),
                Text(""),
                Text(_("例: kgclaw run -T 1 -d examples/人物图谱/人物关系图谱原始数据.txt"), style="bright_black"),
            ),
            box=box.ROUNDED,
            border_style="yellow",
            padding=(1, 2),
        )
        console.print(panel)
        console.print(_("  继续运行将尝试从文档内容自动推断实体和关系类型...\n"), style=STYLE["dim"])

    if not docs and not docs_dir:
        console.print()
        console.print(_("  [FAIL] [red]未指定文档。请使用 --docs 或 --docs-dir 指定。[/red]"))
        console.print(_("  Tip: 例: kgclaw run -T 1 -d examples/人物图谱/人物关系图谱原始数据.txt"), style=STYLE["dim"])
        console.print()
        sys.exit(1)

    # 加载本体
    ontology_raw = ontology
    if ontology_file:
        onto_path = Path(ontology_file)
        if onto_path.suffix in ('.yaml', '.yml'):
            import yaml as _yaml
            with open(onto_path) as f:
                ontology_data = _yaml.safe_load(f)
            ontology_raw = _yaml.dump(ontology_data, allow_unicode=True)
        else:
            ontology_raw = onto_path.read_text()

    # 收集文档
    doc_paths = list(docs)
    if docs_dir:
        doc_dir_path = Path(docs_dir)
        supported_exts = (
            '*.txt', '*.md', '*.markdown', '*.text',
            '*.jsonl', '*.docx', '*.pdf', '*.html', '*.htm',
            '*.csv', '*.tsv', '*.xlsx', '*.xls',
        )
        for ext in supported_exts:
            doc_paths.extend(str(p) for p in doc_dir_path.rglob(ext))

    if not doc_paths:
        console.print(f"  {ICON['fail']} [red]未找到任何文档文件。[/red]")
        console.print(f"  支持的格式: .txt, .md, .jsonl, .docx, .pdf, .html, .csv, .tsv, .xlsx, .xls 等", style=STYLE["dim"])
        sys.exit(1)

    # 初始化日志
    from .logger import setup_logging
    setup_logging(work_dir=work_dir, debug=debug, trace=trace, quiet=quiet)

    # Start trace file if enabled
    if trace:
        from .logger import get_logger
        get_logger().trace_start()

    # 创建 Harness（从配置文件读取 temperature 等偏好设置）
    llm_cfg = UserConfig.get_llm_config()
    config = HarnessConfig(
        llm=LLMConfig(
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=float(llm_cfg.get("temperature", 0.3)),
        ),
        output_format=output_format,
        verbose=verbose,
        work_dir=work_dir,
        max_concurrent_agents=concurrency if concurrency else 8,
        chunk_size=chunk_size_opt if chunk_size_opt else 4000,
    )

    harness = Harness(config)
    progress_cb, progress_stop = make_progress_callback()
    harness.on_event(progress_cb)

    # Handle --resume: load existing workflow state
    if resume:
        state_file = Path(work_dir) / "workflow_state.json"
        if state_file.exists():
            import json as _json
            try:
                state_data = _json.loads(state_file.read_text())
                wf_id = state_data.get("workflow_id", "unknown")[:12]
                console.print(f"  [OK] 恢复会话 [bold]{wf_id}[/bold]", style="green")

                # Restore ontology if not overridden
                onto_restored = False
                if not ontology_raw and not ontology_file:
                    onto_data = state_data.get("ontology", {})
                    if onto_data.get("raw_definition"):
                        ontology_raw = onto_data["raw_definition"]
                        onto_restored = True
                        console.print(f"  [dim]已恢复本体定义[/dim]")
                    elif onto_data.get("entity_types"):
                        guide_parts = [f"# Ontology: {onto_data.get('name', 'resumed')}"]
                        for et in onto_data.get("entity_types", []):
                            guide_parts.append(f"Entity Type: {et['name']}" + (f" ({et.get('description', '')})" if et.get('description') else ''))
                        for rt in onto_data.get("relation_types", []):
                            guide_parts.append(f"Relation Type: {rt['name']}" + (f" (domain={rt.get('domain', '')}, range={rt.get('range', '')})" if rt.get('domain') or rt.get('range') else ''))
                        ontology_raw = "\n".join(guide_parts)
                        onto_restored = True
                        console.print(f"  [dim]已从结构化本体恢复定义[/dim]")

                # Restore document paths if not overridden
                if not doc_paths:
                    restored_docs = [d.get("source", "") for d in state_data.get("documents", []) if d.get("source")]
                    existing = [p for p in restored_docs if Path(p).exists()]
                    if existing:
                        doc_paths = existing
                        console.print(f"  [dim]已恢复 {len(existing)} 个文档路径[/dim]")
                    else:
                        console.print(f"  [yellow]警告: 原文档路径已失效，请使用 -d 重新指定[/yellow]")

                # ── File change detection (unless --force) ──────────────
                if not force_run and doc_paths:
                    from .memory import Memory as _Mem
                    tmp_mem = _Mem(work_dir=work_dir)
                    changes = tmp_mem.detect_file_changes(doc_paths)

                    has_file_changes = bool(
                        changes.get("added") or changes.get("modified") or changes.get("deleted")
                    )

                    if not has_file_changes and not onto_restored:
                        console.print(f"  [green] 文件与本体均无变化[/green]")
                    else:
                        console.print()
                        console.print(f"  [bold]变更检测:[/bold]")
                        if changes.get("unchanged"):
                            console.print(f"      {len(changes['unchanged'])} 个文件未变化")
                        if changes.get("added"):
                            console.print(f"    + {len(changes['added'])} 个新文件")
                            for fp in changes["added"][:3]:
                                console.print(f"       {Path(fp).name}")
                        if changes.get("modified"):
                            console.print(f"    * {len(changes['modified'])} 个文件已修改")
                            for fp in changes["modified"][:3]:
                                console.print(f"       {Path(fp).name}")
                        if changes.get("deleted"):
                            console.print(f"    - {len(changes['deleted'])} 个文件已删除")
                            for fp in changes["deleted"][:3]:
                                console.print(f"       {Path(fp).name}")
                        if onto_restored:
                            console.print(f"    本体已恢复（可能与之前不同）")
                        console.print()
            except Exception as e:
                console.print(f"  [yellow]警告: 无法加载会话状态: {e}[/yellow]")
        else:
            console.print(f"  [yellow]警告: 未找到会话状态文件 {state_file}，将开始全新构建[/yellow]")

    # 加载
    harness.load_documents(doc_paths)
    if ontology_raw:
        ontology = harness.set_ontology(ontology_raw)
        # Display structured ontology analysis result
        if ontology.is_structured:
            from .ui.display import print_ontology
            print_ontology(ontology)

    # 显示运行信息
    console.print()
    info_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    info_table.add_column(style="cyan")
    info_table.add_column(style="white")
    info_table.add_row(f"{ICON['model']} 模型", model)
    info_table.add_row(f"{ICON['doc']} 文档", f"{len(doc_paths)} 个文件")
    if ontology_raw:
        onto = harness.memory.workflow.ontology if harness.memory.workflow else None
        if onto and onto.is_structured:
            summary = f"{len(onto.entity_types)} 实体类型, {len(onto.relation_types)} 关系类型"
        else:
            preview = ontology_raw.replace('\n', ' ')[:80]
            summary = f"{preview}..."
        info_table.add_row(f"{ICON['onto']} 本体", summary)
    console.print(info_table)

    # 执行
    start_time = time.time()
    try:
        result = harness.run(ontology_raw=ontology_raw, strategy=strategy, enable_co_occurrence=co_occurrence)
    except Exception:
        harness._emit("workflow_error", {"error": traceback.format_exc()})
        raise
    finally:
        progress_stop()
        if trace:
            from .logger import get_logger
            get_logger().trace_close()
    elapsed = time.time() - start_time

    # 导出
    if output_format == "nt":
        output = harness.export_nt(output_path)
    elif output_format == "json":
        output = harness.export_json(output_path)
    elif output_format == "jsonl":
        output = harness.export_jsonl(output_path)
    else:
        output = ""

    # 恢复光标（Progress 可能隐藏了光标）
    console.show_cursor(True)

    # 展示结果
    print_stats(result, harness.memory.workflow.ontology if harness.memory.workflow else None)

    console.print()
    console.print(f"  {ICON['time']} 耗时: {elapsed:.1f} 秒")
    console.print(f"  {ICON['output']} 输出: [bold]{output_path}[/bold]")

    # 详细模式下展示输出预览
    if output and verbose:
        console.print()
        preview_syntax = output_format if output_format != "nt" else "turtle"
        console.print(
            Panel(
                Syntax(output[:3000], preview_syntax, theme="monokai", word_wrap=True),
                title="输出预览",
                border_style="green",
            )
        )

    # 结束提示
    if not verbose and result.triples:
        console.print(f"  Tip: 使用 [bold]-v[/bold] 查看详细输出和完整三元组列表", style=STYLE["dim"])


# ── refine ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--feedback", "-f", "feedback_text", default=None, help="用户反馈意见（自然语言）")
@click.option("--work-dir", default=".kgclaw", help="工作目录（默认 .kgclaw）")
@click.option("--model", "-m", default=None, help="覆盖默认模型")
@click.option("--api-key", default=None, help="覆盖 API Key")
@click.option("--api-base", default=None, help="覆盖 API Base URL")
@click.option("--apply", "auto_apply", is_flag=True, help="自动应用优化方案（跳过确认）")
def refine(
    feedback_text: Optional[str],
    work_dir: str,
    model: Optional[str],
    api_key: Optional[str],
    api_base: Optional[str],
    auto_apply: bool,
):
    """根据用户反馈优化知识图谱构建方案。

    分析上次构建的结果，结合用户反馈，提出本体、策略和 Prompt 的优化方案。
    需要先运行过一次 kgclaw run 才能使用。

    \\b
    示例：
      kgclaw refine -f "实体类型太少，需要增加作者和编辑者"
      kgclaw refine -f "关系抽取遗漏了很多跨句子的关系" --apply
    """
    from .refinement import RefinementEngine
    from .logger import setup_logging, get_logger
    from rich.prompt import Confirm as _Confirm

    print_banner()

    # Ensure configured
    api_key, api_base, model = ensure_configured(api_key, model, api_base)

    # Init logging
    setup_logging(work_dir=work_dir)

    # Load last workflow state
    from .memory import Memory
    mem = Memory(work_dir=work_dir)
    wf = mem.load_workflow()
    if not wf or not wf.final_result:
        console.print()
        console.print(_("  [FAIL] [red]未找到上次构建结果。请先运行 kgclaw run。[/red]"))
        console.print(f"  [dim]工作目录: {Path(work_dir).absolute()}[/dim]")
        sys.exit(1)

    if not wf.ontology or not wf.ontology.raw_definition:
        console.print(_("  [FAIL] [red]上次构建没有本体定义。[/red]"))
        sys.exit(1)

    # Get feedback (from option or prompt)
    if feedback_text:
        feedback = feedback_text.strip()
    else:
        console.print()
        console.print(_("  [bold]请输入你的反馈意见[/bold]（描述上次构建结果哪里不满意）"))
        console.print(_("  [dim]例: 实体类型太少，需要增加作者和编辑者类型[/dim]"))
        console.print(_("  [dim]例: 关系抽取遗漏了很多跨句子的关系[/dim]"))
        console.print()
        feedback = Prompt.ask("  反馈")
        if not feedback.strip():
            console.print(_("  [!] 未输入反馈，已取消。"), style="yellow")
            sys.exit(0)

    # Build LLM config
    llm_cfg_data = resolve_llm_config(api_key, model, api_base)
    llm_config = LLMConfig(
        api_key=llm_cfg_data["api_key"],
        model=llm_cfg_data["model"],
        api_base=llm_cfg_data["api_base"],
    )

    # Run refinement
    console.print()
    console.print(f"  [bold cyan]分析反馈中...[/bold cyan]")
    console.print(f"  [dim]反馈: {feedback[:200]}[/dim]")

    engine = RefinementEngine(llm_config, mem)
    with console.status("[cyan]Refinement Agent 分析中...", spinner="dots"):
        plan = engine.analyze(
            last_result=wf.final_result,
            ontology=wf.ontology,
            docs=wf.documents,
            user_feedback=feedback,
        )

    if not plan.has_changes:
        console.print()
        console.print(_("  [yellow][!] 分析完成，但未发现需要修改的地方。[/yellow]"))
        if plan.rationale:
            console.print(f"  [dim]{plan.rationale[:500]}[/dim]")
        return

    # Display plan
    console.print()
    console.print(Rule(_("  优化方案"), style="cyan"))
    console.print()

    if plan.rationale:
        console.print(f"  [dim]{plan.rationale}[/dim]")
        console.print()

    if plan.ontology_changes:
        console.print(_("  [bold cyan]本体变更:[/bold cyan]"))
        for oc in plan.ontology_changes:
            icon = {"add": "[green]+[/green]", "remove": "[red]-[/red]", "modify": "[yellow]~[/yellow]"}.get(oc.action, "?")
            target_cn = "实体类型" if oc.target == "entity_type" else "关系类型"
            console.print(f"    {icon} {oc.name} [dim]({target_cn})[/dim]")
            if oc.description:
                console.print(f"       {oc.description}")
            if oc.reason:
                console.print(f"       [dim]原因: {oc.reason}[/dim]")
        console.print()

    if plan.updated_ontology_raw:
        preview = plan.updated_ontology_raw[:300]
        console.print(f"  [bold cyan]更新后的本体:[/bold cyan]")
        console.print(f"  [dim]{preview}[/dim]")
        console.print()

    if plan.suggested_strategy:
        console.print(f"  [bold yellow]建议策略:[/bold yellow] {plan.suggested_strategy}")
        console.print()

    if plan.extraction_tips:
        console.print(f"  [bold magenta]抽取提示:[/bold magenta] {plan.extraction_tips[:300]}")
        console.print()

    # Apply (auto or confirm)
    if auto_apply:
        apply_changes = True
    else:
        console.print()
        apply_changes = _Confirm.ask("  [bold]应用以上修改?[/bold]", default=True)

    if not apply_changes:
        console.print(_("  [dim]已取消。修改未应用。[/dim]"))
        return

    # Apply ontology changes directly to the workflow
    if plan.ontology_changes:
        from .models import apply_ontology_changes
        new_onto = apply_ontology_changes(wf.ontology, plan.ontology_changes)
        wf.ontology = new_onto
        mem.save_workflow()
        mem.export_ontology()

        console.print()
        n_et = len(new_onto.entity_types)
        n_rt = len(new_onto.relation_types)
        console.print(f"  [green]✓ 本体已更新 ({n_et} 实体类型, {n_rt} 关系类型)[/green]")

    if plan.updated_ontology_raw:
        wf.ontology.raw_definition = plan.updated_ontology_raw
        mem.save_workflow()
        mem.export_ontology()
        console.print(f"  [green]✓ 本体定义已更新[/green]")

    # Commit the refinement
    git = GitManager(mem.work_dir)
    if git.init():
        git.commit_ontology_update(f"refine: {feedback[:60]}")

    console.print()
    console.print(f"  [bold]运行 [cyan]kgclaw run -d <docs> --resume[/cyan] 使用优化后的配置重新构建。[/bold]")


# ── interactive ──────────────────────────────────────────────────────────────

@main.command()
@click.option("--model", "-m", default=None, help="覆盖默认模型")
@click.option("--api-key", default=None, help="覆盖 API Key")
@click.option("--api-base", default=None, help="覆盖 API Base URL")
@click.option("--debug", is_flag=True, help="调试模式：所有细节写入 .kgclaw/logs/kgclaw.log")
@click.option("--trace", is_flag=True, help="跟踪模式：所有 LLM 交互完整记录到 .kgclaw/traces/（JSONL 格式）")
@click.option("--quiet", "-q", is_flag=True, help="静默模式：减少控制台输出")
@click.option("--work-dir", default=".kgclaw", help="中间文件工作目录（默认 .kgclaw）")
def interactive(model: Optional[str], api_key: Optional[str], api_base: Optional[str], debug: bool, trace: bool = False, quiet: bool = False, work_dir: str = ".kgclaw"):
    """Claude Code 风格交互式 REPL。

    启动交互式会话，支持:
    - /load <path>     加载文档
    - /ontology <text> 设置本体定义
    - /run             运行完整 KG 构建流水线
    - /chat <message>  与 AI 自由对话（流式输出）
    - /status          查看当前状态
    - /help            查看所有命令
    """
    print_banner()

    # 确保已配置
    api_key, api_base, model = ensure_configured(api_key, model, api_base)

    # 初始化日志
    from .logger import setup_logging
    setup_logging(work_dir=work_dir, debug=debug, trace=trace, quiet=quiet)

    # Start trace file if enabled
    if trace:
        from .logger import get_logger
        get_logger().trace_start()

    # 启动 REPL
    from .interactive_app import start_interactive
    start_interactive(api_key, api_base, model, work_dir=work_dir)


# ── list-skills ──────────────────────────────────────────────────────────────

@main.command()
def list_skills():
    """ 列出所有可用的 Skill（技能）。

    每个 Skill 封装了完成特定 KG 构建任务所需的
    System Prompt、工具集和处理逻辑。
    """
    from .skills import SkillRegistry
    print_banner()

    console.print(_("  [bold]可用的 Skill（知识图谱构建技能）[/bold]"))
    console.print()

    for i, meta in enumerate(SkillRegistry.list_all(), 1):
        skill = SkillRegistry.get(meta.name)
        tools = skill.get_tool_names() if skill else []

        panel_content = Group(
            Text(f"{meta.description}", style="white"),
            Text(""),
            Text(f"适用工具: {', '.join(tools)}", style=STYLE["muted"]),
            Text(f"输出: {', '.join(meta.produces)}", style=STYLE["muted"]),
            Text(f"需要本体: {'是' if meta.requires_ontology else '否'}", style=STYLE["dim"]),
        )
        console.print(
            Panel(
                panel_content,
                title=f"[{i}] {meta.name}",
                title_align="left",
                border_style="cyan",
                padding=(0, 1),
            )
        )
        console.print()

    console.print(f"  Tip: Skill 按需自动加载，你也可以编写自定义 Skill 扩展功能。", style=STYLE["dim"])


# ── examples ─────────────────────────────────────────────────────────────────

@main.command()
@click.argument("example_name", required=False)
def examples(example_name: Optional[str] = None):
    """ 查看或运行内置示例。

    不指定名称时，列出所有可用的示例数据集。
    指定名称时，展示该示例的详细信息。

    \\b
    示例：
      kgclaw examples           # 列出所有示例
      kgclaw examples 人物图谱   # 查看人物图谱示例详情
    """
    print_banner()
    examples_dir = Path(__file__).parent.parent / "examples"

    if not example_name:
        console.print(_("  [bold]内置示例数据集[/bold]"))
        console.print()
        if examples_dir.exists():
            for d in sorted(examples_dir.iterdir()):
                if d.is_dir():
                    files = list(d.rglob("*"))
                    data_files = [f for f in files if f.suffix in ('.txt', '.jsonl', '.nt', '.docx', '.csv')]
                    py_files = [f for f in files if f.suffix == '.py']

                    panel_content = Group(
                        Text(f"数据文件: {len(data_files)} 个", style="white"),
                        Text(f"Python 处理脚本: {len(py_files)} 个", style=STYLE["dim"]),
                        Text(f"路径: examples/{d.name}/", style=STYLE["muted"]),
                    )
                    console.print(
                        Panel(
                            panel_content,
                            title=f"[bold cyan]{d.name}[/bold cyan]",
                            title_align="left",
                            border_style="cyan",
                        )
                    )
                    console.print()
        else:
            console.print(_("  [dim]未找到示例目录[/dim]"))
            console.print()

        console.print(_("  使用 [bold]kgclaw examples <名称>[/bold] 查看详情"))
        console.print()
        console.print(_("  [bold]Tip: 快速开始：[/bold]"))
        console.print(f"    kgclaw run -T 1 -d examples/人物图谱/人物关系图谱原始数据.txt", style="bright_black")
        return

    example_dir = examples_dir / example_name
    if not example_dir.exists():
        console.print(f"  {ICON['fail']} [red]示例 '{example_name}' 未找到[/red]")
        console.print()
        console.print(_("  可用的示例："))
        if examples_dir.exists():
            for d in sorted(examples_dir.iterdir()):
                if d.is_dir():
                    console.print(f"    {ICON['bullet']} [cyan]{d.name}[/cyan]")
        sys.exit(1)

    console.print(f"  [bold]示例详情: {example_name}[/bold]")
    console.print()

    # 遍历文件
    file_tree = Tree(f"examples/{example_name}/", style="cyan")
    for f in sorted(example_dir.rglob("*")):
        if f.is_file() and f.suffix not in ('.xml', '.iml') and '.idea' not in str(f):
            rel = f.relative_to(example_dir)
            size = f.stat().st_size
            icon = {
                '.txt': 'TXT', '.jsonl': '', '.nt': '',
                '.docx': 'DOC', '.csv': '', '.xlsx': 'XLS',
                '.py': 'PY', '.zip': '', '.md': 'MD',
            }.get(f.suffix, 'FILE')
            file_tree.add(f"{icon} [cyan]{rel}[/cyan] [dim]({size:,} bytes)[/dim]")
    console.print(file_tree)
    console.print()


if __name__ == "__main__":
    main()
