# KGClaw — Knowledge Graph Construction Harness

**[English](README.md) | [中文](README.zh.md)**

<p align="center">
  <em>基于 AI Agent Harness 的本体驱动知识图谱构建系统</em>
</p>

---

## 概述

KGClaw 是一个**类 Claude Code / OpenCode 的 Agent Harness 系统**，专门用于**根据用户给定的本体（Ontology），从非结构化文本中自动构建知识图谱**。

提供三种使用模式：

- **命令行模式** (`kgclaw run`)：给定本体 + 文档，自动运行 8 阶段流水线，输出 N-Triples / JSON / JSONL
- **交互式 REPL** (`kgclaw interactive`) — **推荐**：Claude Code 风格终端，支持流式对话、斜杠命令、Agent 主动提案、Ctrl+O 实时切换详细日志
- **Python API** (`from kgclaw import Harness`)：编程式集成

### 核心特性

- **自然语言自定义本体**：用自然语言直接定义实体类型和关系类型，无需学习 OWL、RDF 等形式化本体语言。LLM 自动将自然语言解析为结构化 Schema，包含描述、属性、domain/range 约束和父子层级关系。
- **零本体也能自动发现**：不提供本体也没关系，直接把文档扔进去——KGClaw 会让 LLM 阅读文档内容，自动归纳出合理的实体类型和关系类型，然后基于自动发现的本体构建知识图谱。
- **本体引导的全流程图谱构建**：全部 8 个阶段始终由你的本体驱动和约束。只抽取本体中定义的实体类型和关系类型；通过 Schema Canonicalization 将 LLM 的开放关系名自动映射到本体标准关系名；三元组构造阶段利用 domain/range 约束确保类型一致性。
- **本体可随时修改、迭代优化**：随时修改本体并用 `/rebuild` 一键重建。使用 `/refine` 命令，用自然语言描述问题（如「实体类型太少，需要增加作者和编辑者」），LLM 自动分析上次构建结果，生成具体的本体变更方案、策略调整建议和 Prompt 优化建议，确认后一键应用。
- **结构化数据也能融入图谱**：CSV 和 XLSX 文件不会被简单展平为文本。KGClaw 保留原始行列结构，通过 LLM 自动分析列名并映射到本体（如「姓名」列 → Person 实体、「所属部门」列 →「任职于」关系），从表格中直接抽取实体和关系。
- **共现图谱自动补充隐含关系**：即使 LLM 没有显式抽取出两个实体之间的关系，KGClaw 也会构建段落级实体共现加权网络。当本体的 domain/range 约束匹配时，共现对会自动升级为对应的本体关系类型，有效发现文档中分散的隐含关系。
- **Gleaning 二次查漏**（受 LightRAG 启发）：首轮实体抽取完成后，将已抽取的实体列表反馈给 LLM，针对性追问「还有哪些实体被遗漏或格式错误？」，做第二轮补充抽取，显著提升实体召回率。
- **质量审核与自动修正**：独立的质检阶段会对照本体审核每条实体和关系的合理性，标记应拒绝的错误条目，提出类型修正建议，并对不匹配本体的关系做 Schema Canonicalization。最终聚合结果时自动过滤被拒绝项、应用修正建议。
- **会话恢复 + 智能增量重建**：重新启动 KGClaw 时自动检测上次构建状态，可一键恢复。通过文档清单（MD5 哈希）对比文件变更——如果文件和本体都没变，直接复用缓存结果；如果有新增/修改/删除的文件，明确告知变更内容及重建原因。
- **构建历史可回滚**：每次 `/run` 自动通过 Git 提交构建结果。使用 `/history` 浏览历史版本，`/rollback <hash>` 回滚到任一历史版本。本体的每次修改也会被单独追踪。

---

## 快速开始

### 1. 安装

```bash
cd KGClaw
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. 首次配置

```bash
kgclaw setup
```

交互式向导引导选择 LLM 服务商 → 输入 API Key → 选择默认模型，保存到 `~/.kgclaw/config.yaml`。

### 3. 直接运行

```bash
# 使用内置模板 + 示例数据
kgclaw run -T 1 -d examples/人物图谱/人物关系图谱原始数据.txt

# 或启动交互式 REPL
kgclaw interactive
```

---

## 命令行模式

```bash
# 自然语言本体 + 文档
kgclaw run -t "实体: 人物, 地点  关系: 居住于, 出生地" -d docs.txt

# 内置模板（1=人物关系, 2=企业, 3=法律法规）
kgclaw run -T 1 -d examples/人物图谱/人物关系图谱原始数据.txt

# 从文件加载本体、处理整个目录
kgclaw run -O ontology.yaml -D my_docs/ -f json

# 快速策略 + 禁共现
kgclaw run --strategy fast --no-co-occurrence -d simple.txt

# Debug 模式（所有细节写入 .kgclaw/logs/）
kgclaw run --debug -T 1 -d examples/人物图谱/人物关系图谱原始数据.txt
```

| 参数 | 说明 |
|------|------|
| `--ontology` / `-t` | 本体定义文本 |
| `--ontology-file` / `-O` | 本体定义文件路径 |
| `--template` / `-T` | 内置模板：1=人物关系, 2=企业, 3=法律法规 |
| `--docs` / `-d` | 文档路径（可多次指定） |
| `--docs-dir` / `-D` | 文档目录 |
| `--output` / `-o` | 输出路径（默认 `.kgclaw/output.nt`） |
| `--format` / `-f` | `nt` / `json` / `jsonl` |
| `--model` / `-m` | 覆盖默认模型 |
| `--verbose` / `-v` | 详细输出 |
| `--debug` | 调试日志写入文件 |
| `--strategy` / `-s` | `auto` / `fast` / `standard` / `code` |
| `--co-occurrence` / `--no-co-occurrence` | 是否构建共现图谱（默认开启） |
| `--work-dir` | 工作目录（默认 `.kgclaw`） |

---

## 交互式 REPL

交互模式提供 Claude Code 风格的终端体验。除了斜杠命令外，你还可以用**自然语言**与 AI 助手交互——例如，直接输入 `"加载这个目录下的全部doc文件"`，Agent 会自动找到并加载目录下所有 `.docx` 文件。

```
> /load 人物关系图谱原始数据.txt
  [OK] 已加载 人物关系图谱原始数据.txt (3463 行, 362,621 bytes)

> /ontology Entity Types: 人物\nRelation Types: 生父, 儿子, 老师
  [OK] 本体已设置

> /strategy standard
  [OK] 工作流策略: standard

> /run
  (8 阶段流水线执行，实时加权进度条 + Token 统计)

> /chat 这里的文本主要描述了人物之间的父子、夫妻、朋友等关系
  ... (流式 Markdown 回复，Agent 主动 propose_action)
```

### 斜杠命令

| 命令 | 功能 |
|------|------|
| `/load <path>` | 加载文档文件或目录 |
| `/docs` | 查看已加载文档列表（含元数据） |
| `/unload <path\|filename>` | 移除已加载的某个文档 |
| `/clear-docs` | 清除所有已加载的文档 |
| `/ontology <text>` | 设置本体定义（支持自然语言） |
| `/template <1\|2\|3>` | 使用内置本体模板 |
| `/run` | 运行完整 KG 构建流水线 |
| `/extract-entities` | 仅运行实体抽取 |
| `/strategy <auto\|fast\|standard\|code>` | 设置工作流策略 |
| `/cooccur` | 切换共现图谱构建 |
| `/output <path>` | 设置输出文件路径 |
| `/format <nt\|json\|jsonl>` | 设置输出格式 |
| `/chat <msg>` | 与 AI 助手自由对话（流式 Markdown 渲染） |
| `/verbose` | 切换详细消息流（显示 LLM 调用细节） |
| `/debug` | 切换调试日志（全部细节写入文件） |
| `/status` | 显示当前会话状态 |
| `/history` | 显示对话历史 |
| `/examples` | 查看内置示例数据 |
| `/config` | 查看当前 LLM 配置 |
| `/clear` | 清除对话历史 |
| `/reset` | 完全重置会话 |
| `/help` | 显示所有命令 |
| `/quit` | 退出 |

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+O` | 实时切换 verbose 模式 |
| `Ctrl+C` | 中断当前操作 |
| `Ctrl+D` | 退出 REPL |

---

## 架构

### 自适应流水线

```
用户输入 (本体 + 文档/目录)
        │
        ▼
┌──────────────────────────────────────────────┐
│ Phase 0:   本体自动发现                        │  ← 无本体时 LLM 归纳
├──────────────────────────────────────────────┤
│ Phase 1:   本体分析                           │  ← LLM 结构化 Schema
├──────────────────────────────────────────────┤
│ Phase 1.5: Agent 代码抽取（可选）               │  ← Agent 生成定制化 prompt/code
├──────────────────────────────────────────────┤
│ Phase 2:   实体抽取 (并行 ThreadPool)          │  ← LLM 并行分块 + Gleaning 补漏
├──────────────────────────────────────────────┤
│ Phase 2.5: 结构化数据抽取                      │  ← CSV/XLSX → 列名→本体映射
├──────────────────────────────────────────────┤
│ Phase 3:   关系抽取                           │  ← 文档分组 + 实体预过滤并行
├──────────────────────────────────────────────┤
│ Phase 3.5: 共现图谱                           │  ← 段落级共现 + 频率加权 + 兜底
├──────────────────────────────────────────────┤
│ Phase 4:   质量审核                           │  ← LLM 审核 + Schema Canonicalization
├──────────────────────────────────────────────┤
│ Phase 5:   三元组构造（程序化）                 │  ← 4 级模糊匹配 + domain/range 约束
└──────────────────────────────────────────────┘
        │
        ▼
   输出: N-Triples / JSON / JSONL
```

### 策略选择

| 策略 | 适用场景 | 说明 |
|------|----------|------|
| `fast` | 少量短文档 | 单轮合并实体+关系抽取 |
| `standard` | 长文本/混合格式 | 完整 8 阶段流水线 |
| `code` | 大量表格数据 | Agent 生成 Python 代码沙盒执行 |
| `auto` | 不确定时 | 根据数据特征自动选择 |

### 项目结构

```
kgclaw/
├── agent.py                # Agent 系统 (LLM + Tool Use + Stream + SubAgent + 熔断)
├── cli.py                  # CLI 入口 (Click + Rich)
├── interactive_app.py      # 交互式 REPL (prompt_toolkit + Live/Markdown)
├── models.py               # Pydantic 数据模型
├── config.py               # 用户配置管理 (~/.kgclaw/config.yaml)
├── memory.py               # 会话记忆 (对话压缩 + 工作流持久化 + .nt 导出)
├── logger.py               # 结构化日志 (RotatingFileHandler + Debug)
├── loaders.py              # 13 种格式文件加载器 + 目录递归
├── sandbox.py              # 沙盒执行 (run_python + AST 审计 + 格式分析)
├── i18n.py                 # 国际化 (gettext-style, zh/en)
├── refinement.py           # KG 优化引擎 (基于用户反馈的本体优化)
├── git_manager.py          # Git 版本管理 (构建历史追踪)
├── harness/                # 编排引擎
│   ├── engine.py           #   主引擎 + 文档加载 + 导出
│   ├── phases.py           #   8 阶段实现 + Gleaning + Canonicalization
│   ├── strategies.py       #   auto/fast/code 三种策略
│   └── helpers.py          #   分块/去重/模糊匹配/Agent 工厂
├── tools/                  # 工具系统 (13 个工具)
│   ├── file_tools.py       #   read_file, write_file, list_files
│   ├── text_tools.py       #   search_in_text, extract_text_segments, parse_json
│   ├── validation_tools.py #   validate_against_ontology, deduplicate_entities
│   ├── agent_tools.py      #   propose_action, run_python, analyze_file_format
│   └── extraction_tools.py #   extract_with_llm_prompt, extract_with_code
├── skills/                 # 技能系统
│   ├── __init__.py         #   Skill 基类 + SkillRegistry
│   └── builtins.py         #   5 个内置 Skill
├── prompts/                # Prompt 模板
│   └── system_prompts.py   #   System/Task Prompt + Few-shot 生成器
└── ui/                     # UI 共享层
    ├── progress.py         #   加权进度回调工厂
    └── display.py          #   结果展示工具
```

详细文档见 [ARCHITECTURE.md](docs/ARCHITECTURE.md) — 系统架构与模块参考；[TECHNICAL_OVERVIEW.md](docs/TECHNICAL_OVERVIEW.md) — 综合技术白皮书。

---

## Python API

```python
from kgclaw import Harness, HarnessConfig, LLMConfig

config = HarnessConfig(
    llm=LLMConfig(model="gpt-4o", api_key="sk-..."),
    enable_gleaning=True,
    chunk_size=2000,
)
harness = Harness(config)
harness.load_documents(["docs.txt"])
harness.set_ontology("Entity Types: 人物\nRelation Types: 生父, 儿子")

# 注册进度回调
from kgclaw.ui import make_progress_callback
cb, stop = make_progress_callback()
harness.on_event(cb)

result = harness.run()
harness.export_nt("output.nt")
harness.export_json("output.json")
stop()
```

---

## 配置

### 配置文件 `~/.kgclaw/config.yaml`

```yaml
llm:
  provider: openai
  model: gpt-4o
  api_key: sk-xxx
  api_base: https://api.openai.com/v1
  temperature: 0.3
  max_tokens: 16384
preferences:
  output_format: nt
  chunk_size: 2000
  verbose: false
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | API 密钥 |
| `KGCLAW_MODEL` | 模型名称 |
| `KGCLAW_API_BASE` | API 端点 URL |
| `KGCLAW_VERBOSE` | 详细输出 |

### 支持的 LLM

任何 OpenAI 兼容 API：OpenAI / DeepSeek / 通义千问 / Ollama (本地) / vLLM / 自定义

---

## 开发

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v                # 128 项测试，无需 API Key
```

### 依赖

```
openai >= 1.0.0         # LLM API
pydantic >= 2.0.0       # 数据模型
rich >= 13.0.0          # 终端 UI
click >= 8.0.0          # CLI
prompt_toolkit >= 3.0.0 # REPL
pyyaml >= 6.0           # 配置解析
httpx >= 0.24.0         # HTTP 客户端
pypdf >= 4.0            # PDF 支持
beautifulsoup4 >= 4.12  # HTML 支持
openpyxl >= 3.1         # Excel 支持
lxml >= 5.0             # XML 解析
```

纯 Python，无外部数据库依赖。

---

## 致谢

本项目在设计和实现过程中，从以下优秀项目中获得了灵感和启发：

- **Claude Code** (Anthropic, 2024-2026) — Agent Harness 架构、Dynamic Workflows 编排模式
- **OpenCode** (2024-2026) — Agent Harness 架构、Tool 注册中心、Permission 系统
- **LightRAG** (HKU, 2024-2025) — Gleaning 二次补漏、实体描述字段、命名规范化规则
- **edc** (2024) — 开放抽取→标准化、Schema Canonicalization
- **Apple ODKE+** (2025) — 生产级本体引导知识抽取流水线
- **Microsoft GraphRAG** (2024) — 非结构化文本→实体/关系→社区检测

---

## License

MIT
