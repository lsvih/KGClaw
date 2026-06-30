"""
Tool definitions for KGClaw agents.

Tools are the atomic capabilities that agents can use.
Each tool has a name, description, parameter schema, and an execute function.

Tool implementations are organized into submodules:
- file_tools:       read_file, write_file, list_files
- text_tools:       search_in_text, extract_text_segments, parse_json
- validation_tools: validate_against_ontology, deduplicate_entities
- agent_tools:      propose_action, run_python, analyze_file_format
- extraction_tools: extract_with_llm_prompt, extract_with_code
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from ..models import ToolDefinition, ToolResult

# ─── Tool Registry ───────────────────────────────────────────────────────────

_tool_registry: dict[str, "Tool"] = {}


class Tool:
    """A callable tool that can be used by agents."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        execute_fn: Callable[..., Any],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self._execute_fn = execute_fn

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def execute(self, **kwargs) -> ToolResult:
        try:
            result = self._execute_fn(**kwargs)
            return ToolResult(success=True, data=result)
        except TypeError:
            # LLM may use different parameter names — try positional fallback
            sig = inspect.signature(self._execute_fn)
            param_names = list(sig.parameters.keys())
            if len(kwargs) == 1 and len(param_names) == 1:
                val = list(kwargs.values())[0]
                try:
                    result = self._execute_fn(val)
                    return ToolResult(success=True, data=result)
                except Exception:
                    pass
            return ToolResult(success=False, error=str(TypeError))
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    @classmethod
    def register(
        cls,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ):
        """Decorator to register a tool function."""

        def decorator(fn: Callable) -> "Tool":
            tool = cls(name, description, parameters, fn)
            _tool_registry[name] = tool
            return tool

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional["Tool"]:
        return _tool_registry.get(name)

    @classmethod
    def list_all(cls) -> list["Tool"]:
        return list(_tool_registry.values())

    @classmethod
    def get_definitions(cls, names: Optional[list[str]] = None) -> list[ToolDefinition]:
        if names is None:
            tools = cls.list_all()
        else:
            tools = [cls.get(n) for n in names]
        return [t.to_definition() for t in tools if t is not None]


# ─── Tool Discovery (imports trigger registration via @Tool.register) ────────

def discover_tools(names: Optional[list[str]] = None) -> list[ToolDefinition]:
    return Tool.get_definitions(names)


def get_tool(name: str) -> Optional[Tool]:
    return Tool.get(name)


def execute_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
    tool = Tool.get(name)
    if tool is None:
        return ToolResult(success=False, error=f"Unknown tool: {name}")
    return tool.execute(**arguments)


# ─── Import tool implementations to trigger registration ────────────────────

from . import file_tools       # noqa: E402, F401 — registers read_file, write_file, list_files
from . import text_tools       # noqa: E402, F401 — registers search_in_text, extract_text_segments, parse_json
from . import validation_tools # noqa: E402, F401 — registers validate_against_ontology, deduplicate_entities
from . import agent_tools      # noqa: E402, F401 — registers propose_action, run_python, analyze_file_format
from . import extraction_tools # noqa: E402, F401 — registers extract_with_llm_prompt, extract_with_code
