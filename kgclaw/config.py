"""
Configuration management for KGClaw.

Handles user configuration stored at ~/.kgclaw/config.yaml:
- LLM provider settings (api_key, api_base, model)
- User preferences
- Supports environment variable overrides
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import copy

import os
from pathlib import Path
from typing import Any, Optional

import yaml


class UserConfig:
    """User-facing configuration loaded from ~/.kgclaw/config.yaml."""

    DEFAULT_CONFIG = {
        "llm": {
            "provider": "openai",
            "model": "deepseek-v4-flash",
            "api_key": "",
            "api_base": "https://api.deepseek.com/v1",
            "temperature": 0.3,
            "max_tokens": 16384,
        },
        "preferences": {
            "output_format": "nt",
            "chunk_size": 2000,
            "verbose": False,
            "lang": "en",
        },
    }

    @classmethod
    def config_dir(cls) -> Path:
        return Path.home() / ".kgclaw"

    @classmethod
    def config_path(cls) -> Path:
        return cls.config_dir() / "config.yaml"

    @classmethod
    def exists(cls) -> bool:
        return cls.config_path().exists()

    @classmethod
    def load(cls) -> dict[str, Any]:
        """Load config from file, falling back to defaults."""
        path = cls.config_path()
        if path.exists():
            try:
                with open(path) as f:
                    loaded = yaml.safe_load(f) or {}
                # Deep merge with defaults
                return cls._deep_merge(cls.DEFAULT_CONFIG.copy(), loaded)
            except Exception as e:
                import logging
                logging.getLogger("kgclaw").warning(
                    f"Failed to load config from {path}: {e}. Using defaults."
                )
        return cls.DEFAULT_CONFIG.copy()

    @classmethod
    def save(cls, config: dict[str, Any]):
        """Save config to file."""
        cls.config_dir().mkdir(parents=True, exist_ok=True)
        with open(cls.config_path(), "w") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    @classmethod
    def get_llm_config(cls) -> dict[str, Any]:
        """Get LLM config, with env var overrides."""
        config = cls.load()
        llm = config.get("llm", cls.DEFAULT_CONFIG["llm"])

        # Env var overrides
        if os.environ.get("OPENAI_API_KEY"):
            llm["api_key"] = os.environ["OPENAI_API_KEY"]
        if os.environ.get("KGCLAW_MODEL"):
            llm["model"] = os.environ["KGCLAW_MODEL"]
        if os.environ.get("KGCLAW_API_BASE"):
            llm["api_base"] = os.environ["KGCLAW_API_BASE"]

        return llm

    @classmethod
    def get_lang(cls) -> str:
        """Get the configured UI language. Falls back to env var or auto-detect."""
        # 1. Explicit env var
        env_lang = os.environ.get("KGCLAW_LANG", "")
        if env_lang and env_lang[:2] in ("zh", "en"):
            return env_lang[:2]
        # 2. Config file
        config = cls.load()
        lang = config.get("preferences", {}).get("lang", "en")
        if lang in ("zh", "en"):
            return lang
        return "en"

    @classmethod
    def is_configured(cls) -> bool:
        """Check if user has completed setup (has API key configured)."""
        llm = cls.get_llm_config()
        return bool((llm.get("api_key", "") or "").strip())

    @classmethod
    def _deep_merge(cls, base: dict, override: dict) -> dict:
        """Recursively merge override into base (deep copy to avoid mutation)."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
