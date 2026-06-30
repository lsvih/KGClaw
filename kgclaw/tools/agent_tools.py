"""Agent interaction and sandbox tools for KGClaw agents."""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

from typing import Any

from . import Tool


@Tool.register(
    name="propose_action",
    description="""当用户表现出执行系统操作的意图时，调用此工具提议执行相应操作。

**关键规则——你必须主动调用此工具而不是仅用文字回复：**
- 用户说"帮我构建"、"开始抽取"、"运行"、"构建图谱" → 调用 propose_action("run")
- 用户说"加载这些文件"、"加载目录"、"加载所有数据" → 调用 propose_action("load", path=".")
- 用户说"退出"、"结束" → 调用 propose_action("quit")
- 你分析完数据、检查完文件结构后用户想要执行操作 → 主动 propose_action

**不要在分析完文件后只说"我准备好了"——直接 propose！**

可用操作:
- "run": 运行完整的知识图谱构建流水线
- "extract_entities": 仅运行实体抽取
- "load": 加载文档文件或目录 (参数: path，传 "." 加载当前目录所有文件)
- "ontology": 设置本体定义 (参数: definition)
- "quit": 退出程序""",
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run", "extract_entities", "load", "ontology", "quit"],
                "description": "要执行的操作",
            },
            "path": {
                "type": "string",
                "description": "文件路径 (仅 load 操作需要)",
            },
            "definition": {
                "type": "string",
                "description": "本体定义 (仅 ontology 操作需要)",
            },
            "reason": {
                "type": "string",
                "description": "为什么建议执行此操作",
            },
        },
        "required": ["action"],
    },
)
def tool_propose_action(action: str, path: str = "", definition: str = "", reason: str = "") -> dict[str, Any]:
    """Propose a system operation. This tool is intercepted at the REPL layer."""
    return {
        "action": action,
        "path": path,
        "definition": definition,
        "reason": reason,
        "status": "proposed",
    }


@Tool.register(
    name="run_python",
    description="""在隔离沙盒中执行 Python 代码。适合处理自定义数据格式、批量转换、正则提取等任务。

代码运行在临时目录中，有 30 秒超时限制。可以使用 json, csv, re, collections, itertools, math, pathlib, io, textwrap, datetime 等标准库。
代码中 print() 的输出会被捕获并返回。最后一次表达式的结果也会被捕获。
无法访问网络、子进程或文件系统写操作。

用法示例:
- 处理二进制文件头: 读取文件的前 N 字节进行分析
- 批量正则提取: import re; results = re.findall(pattern, text)
- CSV 转换: import csv, io; reader = csv.DictReader(io.StringIO(data))
- 自定义格式解析: 手写解析逻辑处理非标准格式""",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 Python 代码",
            },
            "input_data": {
                "type": "string",
                "description": "通过 stdin 传递给代码的数据（可选）",
            },
        },
        "required": ["code"],
    },
)
def tool_run_python(code: str, input_data: str = "") -> dict[str, Any]:
    """Execute Python code in sandbox."""
    from ..sandbox import run_python_code, check_code_safety

    safe, reason = check_code_safety(code)
    if not safe:
        return {"success": False, "stdout": "", "stderr": f"Safety check failed: {reason}", "returncode": -1}

    return run_python_code(code, input_data=input_data)


@Tool.register(
    name="analyze_file_format",
    description="分析文件的格式、编码和结构，帮助决定如何加载和处理该文件。返回魔数、编码检测、文本样本和建议的加载方式。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要分析的文件路径"},
        },
        "required": ["path"],
    },
)
def tool_analyze_file_format(path: str) -> dict[str, Any]:
    """Analyze a file's format to determine how to load it."""
    from ..sandbox import analyze_file_format
    return analyze_file_format(path)
