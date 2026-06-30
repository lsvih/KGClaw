"""
Skill system for KGClaw.

Skills are self-contained capabilities that encapsulate domain-specific
logic, prompts, and tool configurations. The Skill registry supports:
- Progressive disclosure (skills loaded on demand)
- Custom user-defined skills via filesystem discovery
- Composition (skills can call other skills)

Built-in skills live in builtins.py and are auto-registered on import.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import importlib
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..models import LLMConfig


@dataclass
class SkillMeta:
    """Metadata for a skill."""
    name: str
    description: str
    version: str = "0.1.0"
    author: str = ""
    tags: list[str] = field(default_factory=list)
    requires_ontology: bool = True
    produces: list[str] = field(default_factory=list)


class Skill(ABC):
    """
    Abstract base class for all skills.

    A Skill encapsulates a complete KGC capability:
    - A system prompt for the LLM agent
    - A set of tools available to the agent
    - Processing logic (pre/post processing)
    - Input/output schemas
    """

    meta: SkillMeta

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for the LLM agent using this skill."""

    @abstractmethod
    def get_tool_names(self) -> list[str]:
        """Return the list of tool names needed by this skill."""

    def pre_process(self, context: dict[str, Any]) -> dict[str, Any]:
        """Pre-process the input context before the LLM call."""
        return context

    def post_process(self, raw_output: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Post-process the LLM output."""
        return raw_output

    def get_input_schema(self) -> dict[str, Any]:
        """Define the expected input schema for this skill."""
        return {}

    def get_output_schema(self) -> dict[str, Any]:
        """Define the output schema for this skill."""
        return {}


# ─── Skill Registry ──────────────────────────────────────────────────────────

class SkillRegistry:
    """Global registry of available skills."""

    _instance: Optional["SkillRegistry"] = None
    _skills: dict[str, type[Skill]] = {}
    _instances: dict[str, Skill] = {}

    def __new__(cls) -> "SkillRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, meta: SkillMeta):
        """Decorator to register a skill class."""

        def decorator(skill_cls: type[Skill]):
            skill_cls.meta = meta
            cls._skills[meta.name] = skill_cls
            return skill_cls

        return decorator

    @classmethod
    def get(cls, name: str, llm_config: Optional[LLMConfig] = None) -> Optional[Skill]:
        """Get a skill instance by name (cached)."""
        cache_key = f"{name}:{id(llm_config)}"
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        skill_cls = cls._skills.get(name)
        if skill_cls is None:
            return None

        instance = skill_cls(llm_config=llm_config)
        cls._instances[cache_key] = instance
        return instance

    @classmethod
    def list_all(cls) -> list[SkillMeta]:
        """List all registered skill metadata."""
        return [skill_cls.meta for skill_cls in cls._skills.values()]

    @classmethod
    def discover_from_directory(cls, directory: str):
        """Discover and load custom skills from a directory."""
        skill_dir = Path(directory)
        if not skill_dir.exists():
            return

        for py_file in skill_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                module_name = py_file.stem
                spec = importlib.util.spec_from_file_location(
                    f"kgclaw_custom_skills.{module_name}", py_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[f"kgclaw_custom_skills.{module_name}"] = module
                    spec.loader.exec_module(module)
            except Exception as e:
                import logging
                logging.warning(f"Failed to load custom skill from {py_file}: {e}")


# ─── Skill Helpers ───────────────────────────────────────────────────────────

def get_all_skill_names() -> list[str]:
    """Get names of all registered skills."""
    return [m.name for m in SkillRegistry.list_all()]


def get_skill(name: str, llm_config: Optional[LLMConfig] = None) -> Optional[Skill]:
    """Get a skill instance by name."""
    return SkillRegistry.get(name, llm_config)


def get_default_pipeline_skills() -> list[str]:
    """Return the default skill pipeline for end-to-end KG construction."""
    return [
        "ontology_analyzer",
        "entity_extractor",
        "relation_extractor",
        "quality_checker",
        "triple_constructor",
    ]


# ─── Import built-in skills to trigger registration ─────────────────────────

from . import builtins  # noqa: E402, F401 — registers all 5 built-in skills
