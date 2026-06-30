"""File I/O tools for KGClaw agents."""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import Tool


@Tool.register(
    name="read_file",
    description="Read the contents of a file. Supports text files and common formats.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read"},
            "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
            "max_lines": {"type": "integer", "description": "Maximum number of lines to read"},
        },
        "required": ["path"],
    },
)
def tool_read_file(path: str, encoding: str = "utf-8", max_lines: Optional[int] = None) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    from ..loaders import get_loader
    ext = p.suffix.lower()
    loader = get_loader(ext)
    if loader and ext not in ('.txt', '.md', '.markdown', '.text'):
        try:
            doc = loader(str(p))
            content = doc.text
        except Exception:
            try:
                content = p.read_text(encoding=encoding, errors='replace')
            except Exception:
                content = p.read_text(encoding='utf-8', errors='replace')
    else:
        try:
            content = p.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            content = p.read_text(encoding='utf-8', errors='replace')

    if max_lines:
        lines_list = content.split("\n")
        content = "\n".join(lines_list[:max_lines])
        if len(lines_list) > max_lines:
            content += f"\n... ({len(lines_list) - max_lines} more lines)"
    return content


@Tool.register(
    name="write_file",
    description="Write content to a file. Creates parent directories if needed.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write the file to"},
            "content": {"type": "string", "description": "Content to write"},
            "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
        },
        "required": ["path", "content"],
    },
)
def tool_write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding=encoding)
    return f"File written: {path} ({len(content)} characters)"


@Tool.register(
    name="list_files",
    description="List files in a directory matching an optional glob pattern.",
    parameters={
        "type": "object",
        "properties": {
            "directory": {"type": "string", "description": "Directory to list"},
            "pattern": {"type": "string", "description": "Glob pattern", "default": "*"},
            "recursive": {"type": "boolean", "description": "Search recursively", "default": False},
        },
        "required": ["directory"],
    },
)
def tool_list_files(directory: str, pattern: str = "*", recursive: bool = False) -> list[str]:
    p = Path(directory)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if recursive:
        matches = list(p.rglob(pattern))
    else:
        matches = list(p.glob(pattern))
    return [str(m) for m in matches]
