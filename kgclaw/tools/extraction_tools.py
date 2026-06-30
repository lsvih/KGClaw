"""LLM-based and code-based extraction tools for KGClaw agents."""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

from . import Tool

# Module-level cached OpenAI client to avoid creating a new connection on every
# tool invocation. Protected by a lock for thread safety.
_cached_client: Optional[Any] = None
_cached_client_key: tuple[str, str, str] = ("", "", "")
_client_lock = threading.Lock()


def _get_cached_llm_client(api_key: str, api_base: str, model: str) -> Any:
    """Return a cached OpenAI client, creating a new one if config changed."""
    global _cached_client, _cached_client_key
    key = (api_key, api_base, model)
    with _client_lock:
        if _cached_client is None or _cached_client_key != key:
            from openai import OpenAI
            _cached_client = OpenAI(api_key=api_key, base_url=api_base, max_retries=3)
            _cached_client_key = key
        return _cached_client


@Tool.register(
    name="extract_with_llm_prompt",
    description="""用 KGClaw 的 LLM 配置执行自定义抽取。你需要编写一个针对当前数据优化的抽取 prompt，KGClaw 会用这个 prompt 调用 LLM 进行实际抽取。

**工作流程**:
1. 分析数据格式（先用 read_file 查看样本）
2. 编写一个针对性的抽取 prompt（不是代码，是 LLM prompt！）
3. 调用此工具，KGClaw 会用你的 prompt + 完整数据调用 LLM
4. 返回 LLM 抽取的结构化结果（entities + relations）

**你的 prompt 应该**:
- 明确列出要抽取的实体类型和关系类型（来自本体）
- 给出 2-3 个具体的数据示例
- 指定输出 JSON 格式: {"entities": [{"name":..., "type":..., "confidence":0.9}], "relations": [{"subject":..., "predicate":..., "object":..., "confidence":0.9}]}
- 针对数据特点给出抽取提示（如"作者名以逗号分隔"、"标题在句号之前"）

**对比**: 这比 extract_with_code 更强大，因为 LLM 直接做语义理解抽取，而非脆弱的正则匹配。""",
    parameters={
        "type": "object",
        "properties": {
            "extraction_prompt": {
                "type": "string",
                "description": "你编写的自定义抽取 prompt，将发送给 LLM 执行抽取",
            },
            "data_text": {
                "type": "string",
                "description": "要处理的数据文本（通过 read_file 读取的内容）",
            },
            "ontology_json": {
                "type": "string",
                "description": "本体定义的 JSON（可选，如已在 prompt 中包含可不传）",
            },
        },
        "required": ["extraction_prompt", "data_text"],
    },
)
def tool_extract_with_llm_prompt(extraction_prompt: str, data_text: str = "", ontology_json: str = "{}") -> dict[str, Any]:
    """Use KGClaw's LLM config to execute a custom extraction prompt."""
    from ..config import UserConfig

    llm_cfg = UserConfig.get_llm_config()
    api_key = llm_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    api_base = llm_cfg.get("api_base") or os.environ.get("KGCLAW_API_BASE", "https://api.openai.com/v1")
    model = llm_cfg.get("model") or os.environ.get("KGCLAW_MODEL", "gpt-4o")

    # Use cached client to avoid re-creating connections on every tool call
    client = _get_cached_llm_client(api_key, api_base, model)

    full_prompt = f"""{extraction_prompt}

## 数据（需要从中抽取）
{data_text[:80000]}

## 输出要求
只返回 JSON: {{"entities": [...], "relations": [...]}}"""

    def _parse_json_from_response(raw: str) -> tuple:
        """Try multiple strategies to parse JSON from LLM response."""
        import json as _json
        import re as _re3
        try:
            return (True, _json.loads(raw))
        except _json.JSONDecodeError:
            pass
        m = _re3.search(r'\{[\s\S]*"entities"[\s\S]*\}', raw)
        if m:
            try:
                return (True, _json.loads(m.group(0)))
            except _json.JSONDecodeError:
                pass
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return (True, _json.loads(raw[brace_start:brace_end + 1]))
            except _json.JSONDecodeError:
                pass
        return (False, None)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.3,
            max_tokens=16384,
        )
        raw = resp.choices[0].message.content or ""
        total_prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        total_completion_tokens = resp.usage.completion_tokens if resp.usage else 0

        ok, parsed = _parse_json_from_response(raw)
        if not ok:
            repair_prompt = f"""Your previous response was not valid JSON. Please fix it and return ONLY a valid JSON object with "entities" and "relations" arrays.

Your previous (malformed) response:
```
{raw[:6000]}
```

Return ONLY the corrected JSON — no markdown, no explanation."""
            try:
                repair_resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": repair_prompt}],
                    temperature=0.1,
                    max_tokens=16384,
                )
                raw2 = repair_resp.choices[0].message.content or ""
                if repair_resp.usage:
                    total_prompt_tokens += repair_resp.usage.prompt_tokens
                    total_completion_tokens += repair_resp.usage.completion_tokens
                ok, parsed = _parse_json_from_response(raw2)
            except Exception:
                pass

        if not ok:
            return {"success": False, "error": f"Failed to parse LLM response as JSON after repair: {raw[:500]}"}

        return {
            "success": True,
            "entities": parsed.get("entities", []),
            "relations": parsed.get("relations", []),
            "entity_count": len(parsed.get("entities", [])),
            "relation_count": len(parsed.get("relations", [])),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@Tool.register(
    name="extract_with_code",
    description="""用自定义 Python 代码从数据中提取实体和关系。适合处理 LLM 不擅长的结构化/半结构化数据。

**使用场景**: 论文列表、参考书目、日志文件、CSV、JSON 等有规律格式的数据。

**工作流程**:
1. 分析数据格式（先用 read_file 查看内容）
2. 编写 Python 提取代码，输出 JSON: {"entities": [{"name":..., "type":..., "confidence":...}], "relations": [{"subject":..., "predicate":..., "object":..., "confidence":...}]}
3. 调用此工具，传入代码和本体定义
4. 工具返回提取的结构化结果

**代码中可以使用的变量**:
- `ONTOLOGY_JSON`: 本体定义的 JSON 对象 (entity_types + relation_types)
- `DATA_TEXT`: 传入的数据文本

**示例** (提取论文列表):
```python
import re, json
entities = []
relations = []
for line in DATA_TEXT.split('\\n'):
    match = re.match(r'(\\S+)\\s*-\\s*(.+?)\\.\\s*(.+)', line)
    if match:
        venue, authors_str, title = match.groups()
        entities.append({"name": title, "type": "Paper", "confidence": 0.9})
        for author in authors_str.split(', '):
            entities.append({"name": author, "type": "Person", "confidence": 0.9})
            relations.append({"subject": author, "predicate": "author_of", "object": title, "confidence": 0.9})
print(json.dumps({"entities": entities, "relations": relations}, ensure_ascii=False))
```""",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "提取数据的 Python 代码。使用 DATA_TEXT 和 ONTOLOGY_JSON 变量。print() JSON 结果到 stdout。",
            },
            "data_text": {
                "type": "string",
                "description": "要处理的数据文本（通过 read_file 读取的内容）",
            },
            "ontology_json": {
                "type": "string",
                "description": "本体定义的 JSON（包含 entity_types 和 relation_types）",
            },
        },
        "required": ["code", "data_text"],
    },
)
def tool_extract_with_code(code: str, data_text: str = "", ontology_json: str = "{}") -> dict[str, Any]:
    """Run agent-written extraction code in sandbox."""
    from ..sandbox import run_python_code, check_code_safety

    safe, reason = check_code_safety(code)
    if not safe:
        return {"success": False, "error": f"Safety check: {reason}"}

    wrapped_code = f"""
import json

ONTOLOGY_JSON = json.loads({json.dumps(ontology_json)!r})
DATA_TEXT = {json.dumps(data_text)!r}

{code}
"""

    result = run_python_code(wrapped_code, timeout=60)
    if result["success"] and result["stdout"].strip():
        try:
            parsed = json.loads(result["stdout"].strip())
            return {
                "success": True,
                "entities": parsed.get("entities", []),
                "relations": parsed.get("relations", []),
                "entity_count": len(parsed.get("entities", [])),
                "relation_count": len(parsed.get("relations", [])),
            }
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": f"Code output is not valid JSON: {result['stdout'][:500]}",
            }
    return {
        "success": False,
        "error": result.get("stderr", "No output"),
        "stdout": result.get("stdout", "")[:500],
    }
