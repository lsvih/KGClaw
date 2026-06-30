"""Text processing tools for KGClaw agents."""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import json
import re
from typing import Any

from . import Tool


@Tool.register(
    name="extract_text_segments",
    description="Extract matching text segments from a larger text using regex patterns.",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Source text to search in"},
            "pattern": {"type": "string", "description": "Regular expression pattern to match"},
            "max_matches": {"type": "integer", "description": "Max matches to return", "default": 50},
        },
        "required": ["text", "pattern"],
    },
)
def tool_extract_text_segments(text: str, pattern: str, max_matches: int = 50) -> list[str]:
    matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
    return matches[:max_matches]


@Tool.register(
    name="search_in_text",
    description="Search for keywords or patterns in text and return surrounding context.",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to search in"},
            "query": {"type": "string", "description": "Keyword or phrase to search for"},
            "context_chars": {"type": "integer", "description": "Characters of context around match", "default": 100},
            "max_results": {"type": "integer", "description": "Max results", "default": 10},
        },
        "required": ["text", "query"],
    },
)
def tool_search_in_text(text: str, query: str, context_chars: int = 100, max_results: int = 10) -> list[dict[str, Any]]:
    results = []
    idx = 0
    while len(results) < max_results:
        idx = text.find(query, idx)
        if idx == -1:
            break
        start = max(0, idx - context_chars)
        end = min(len(text), idx + len(query) + context_chars)
        context = text[start:end]
        results.append({"position": idx, "context": context})
        idx += len(query)
    return results


@Tool.register(
    name="parse_json",
    description="Parse a JSON string into a Python object. Handles JSON embedded in markdown code blocks.",
    parameters={
        "type": "object",
        "properties": {
            "json_string": {"type": "string", "description": "JSON string to parse"},
        },
        "required": ["json_string"],
    },
)
def tool_parse_json(json_string: str) -> Any:
    cleaned = json_string.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)
