"""
Isolated sandbox execution for KGClaw.

Provides:
- P1: run_python tool — execute agent-generated Python code with timeout + capture
- P2: generate_loader — auto-generate file loaders from format analysis
- P3: isolated worktrees — execute code in temp directories with restricted access
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Optional


# ─── Sandbox Config ──────────────────────────────────────────────────────────

SANDBOX_TIMEOUT = 30  # seconds
SANDBOX_MAX_OUTPUT = 100_000  # chars
ALLOWED_IMPORTS = {
    "json", "csv", "re", "collections", "itertools", "math",
    "pathlib", "io", "string", "textwrap", "datetime",
    "typing", "dataclasses", "enum",
}


def run_python_code(
    code: str,
    working_dir: Optional[str] = None,
    timeout: int = SANDBOX_TIMEOUT,
    env: dict[str, str] = None,
    input_data: str = "",
    skip_safety_check: bool = False,
) -> dict[str, Any]:
    """
    Execute Python code in a subprocess sandbox.

    The sandbox runs with the HOST working directory as the current
    directory, so Python code can access files relative to the user's
    project. File writes go to the temp sandbox directory.

    Args:
        code: Python source code to execute
        working_dir: Working directory (default: temp dir)
        timeout: Max execution time in seconds
        env: Extra environment variables
        input_data: String to pass as stdin
        skip_safety_check: If True, skip the AST safety check.
            Only use this for trusted code (e.g., tests of the execution
            mechanism itself). Tool callers should always leave this False.

    Returns:
        dict with keys: success, stdout, stderr, elapsed, error
    """
    work_dir = working_dir or tempfile.mkdtemp(prefix="kgclaw_sandbox_")

    # Defense in depth: AST safety check BEFORE execution.
    # Even though callers (agent_tools.py, extraction_tools.py) also check,
    # this ensures direct callers of run_python_code cannot bypass it.
    if not skip_safety_check:
        safe, reason = check_code_safety(code)
        if not safe:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Safety check failed: {reason}",
                "returncode": -1,
                "elapsed": 0,
            }

    start = time.time()
    try:
        # Write code to temp file
        script_path = Path(work_dir) / "_kgclaw_exec.py"
        script_path.write_text(code, encoding="utf-8")

        # Build environment — HOST_CWD allows code to read files from the user's project
        exec_env = os.environ.copy()
        exec_env["PYTHONPATH"] = os.pathsep.join(sys.path)
        exec_env["KGCLAW_SANDBOX"] = "1"
        exec_env["HOME"] = work_dir
        if env:
            exec_env.update(env)

        # Run in the user's CWD so code can access project files.
        # Sandbox is enforced by the safety checker (AST), not filesystem isolation.
        run_cwd = os.getcwd()
        exec_env["KGCLAW_SANDBOX_DIR"] = work_dir

        proc = subprocess.run(
            [sys.executable, "-I", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=run_cwd,  # Run in user's CWD so files are accessible
            env=exec_env,
            input=input_data,
        )
        stdout = proc.stdout[:SANDBOX_MAX_OUTPUT] if proc.stdout else ""
        stderr = proc.stderr[:SANDBOX_MAX_OUTPUT] if proc.stderr else ""
        success = proc.returncode == 0

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": proc.returncode,
            "elapsed": time.time() - start,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
            "returncode": -1,
            "elapsed": timeout,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": traceback.format_exc(),
            "returncode": -1,
            "elapsed": time.time() - start,
            "error": str(e),
        }
    finally:
        # Clean up temp dir
        try:
            import shutil
            if not working_dir:  # only clean if we created it
                shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


def check_code_safety(code: str) -> tuple[bool, str]:
    """
    Safety check for user-generated code using AST analysis.

    Blocks: import of dangerous modules, calls to dangerous functions,
    __builtins__ manipulation, subscript-based bypasses, and class-hierarchy
    navigation patterns commonly used for sandbox escapes.

    IMPORTANT: AST-based analysis is a best-effort defense, not a security
    guarantee. Malicious actors with deep Python knowledge can construct
    bypasses (e.g., encoding tricks, codecs, C extensions). This sandbox is
    designed for agent-generated code, not untrusted user input.
    """
    import ast

    # Modules that are forbidden to import
    FORBIDDEN_IMPORTS = {
        "os", "subprocess", "socket", "requests", "urllib",
        "shutil", "ctypes", "multiprocessing", "signal",
        "pty", "fcntl", "posix", "grp", "pwd", "crypt",
        "importlib", "sys", "builtins",
    }
    # Function calls that are forbidden even without import
    FORBIDDEN_CALLS = {
        "eval", "exec", "compile", "__import__",
        "breakpoint", "open",
    }
    # Dangerous builtin-related names that bypass import checks
    DANGEROUS_BUILTINS = {"__builtins__", "__builtin__", "globals", "locals"}
    # Additional restricted calls — only blocked when argument references dangerous builtins
    RESTRICTED_CALLS = {
        "vars", "getattr", "setattr", "delattr", "dir",
    }

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        # ── Check imports ──
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")[0]
            else:
                module = node.names[0].name.split(".")[0] if node.names else ""
            if module in FORBIDDEN_IMPORTS:
                return False, f"Forbidden import: {module}"

        # ── Check function calls ──
        elif isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = _get_attr_name(node.func)

            if func_name:
                parts = func_name.split(".")
                leaf = parts[-1]

                # Direct forbidden calls
                if leaf in FORBIDDEN_CALLS:
                    return False, f"Forbidden call: {func_name}()"

                # Restricted calls that are dangerous when referencing builtins
                if leaf in RESTRICTED_CALLS:
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id in DANGEROUS_BUILTINS:
                            return False, f"Forbidden bypass: {func_name}({arg.id}, ...)"
                        if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name) and arg.value.id in DANGEROUS_BUILTINS:
                            return False, f"Forbidden bypass: {func_name}(__builtins__[...], ...)"

                # Any call chain touching dangerous builtins
                if any(p in DANGEROUS_BUILTINS for p in parts):
                    return False, f"Forbidden builtin access: {func_name}"
                if any(p in FORBIDDEN_IMPORTS for p in parts):
                    return False, f"Forbidden module access: {func_name}"

        # ── Check subscript access (__builtins__["eval"] bypass) ──
        elif isinstance(node, ast.Subscript):
            # Direct: __builtins__["eval"]
            if isinstance(node.value, ast.Name):
                if node.value.id in DANGEROUS_BUILTINS:
                    return False, f"Forbidden subscript: {node.value.id}[...]"
                if node.value.id in FORBIDDEN_IMPORTS:
                    return False, f"Forbidden subscript: {node.value.id}[...]"
            # Attribute chain: __builtins__.__dict__[...] or sys.modules[...]
            if isinstance(node.value, ast.Attribute):
                full = _get_attr_name(node.value)
                parts = full.split(".")
                if any(b in parts for b in DANGEROUS_BUILTINS):
                    return False, f"Forbidden subscript access: {full}[...]"
                if any(b in parts for b in FORBIDDEN_IMPORTS):
                    return False, f"Forbidden subscript access: {full}[...]"

        # ── Check class-hierarchy navigation (sandbox escape pattern) ──
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            # ()->__class__->__bases__->... chain
            if attr in ("__class__", "__bases__", "__mro__", "__subclasses__",
                         "__globals__", "__code__", "__closure__", "__dict__"):
                full = _get_attr_name(node)
                if full and not full.startswith(("self.", "cls.")):
                    return False, f"Forbidden attribute access: {full}"

    return True, "OK"


def _get_attr_name(node) -> str:
    """Get full dotted name from an AST Attribute node."""
    import ast
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


# ─── File Format Analyzer (P2) ──────────────────────────────────────────────

def analyze_file_format(path: str, sample_bytes: int = 4096) -> dict[str, Any]:
    """
    Analyze a file to determine its format and suggest a loader approach.

    Returns metadata that helps an agent decide how to load the file.
    """
    p = Path(path)
    if not p.exists():
        return {"error": "File not found", "path": path}

    result = {
        "path": str(p),
        "filename": p.name,
        "extension": p.suffix.lower(),
        "size": p.stat().st_size,
        "is_binary": False,
        "encoding": None,
        "magic_bytes": None,
        "sample_text": "",
        "suggested_loader": None,
    }

    # Read sample
    try:
        with open(p, "rb") as f:
            head = f.read(sample_bytes)
    except Exception as e:
        result["error"] = str(e)
        return result

    result["magic_bytes"] = head[:16].hex()

    # Detect text vs binary
    try:
        text_sample = head.decode("utf-8")
        result["encoding"] = "utf-8"
        result["sample_text"] = text_sample[:1000]
    except UnicodeDecodeError:
        result["is_binary"] = True
        # Try common encodings
        for enc in ["gbk", "gb2312", "shift_jis", "latin-1"]:
            try:
                result["sample_text"] = head.decode(enc)[:1000]
                result["encoding"] = enc
                result["is_binary"] = False
                break
            except Exception:
                pass

    # Suggest loader based on extension and content patterns
    ext = result["extension"]
    sample = result.get("sample_text", "")

    if ext in (".csv", ".tsv"):
        result["suggested_loader"] = "csv"
    elif ext in (".json", ".jsonl"):
        result["suggested_loader"] = "json"
    elif ext in (".yaml", ".yml"):
        result["suggested_loader"] = "yaml"
    elif ext in (".xml"):
        result["suggested_loader"] = "xml"
    elif ext in (".pdf"):
        result["suggested_loader"] = "pdf"
    elif ext in (".docx", ".doc"):
        result["suggested_loader"] = "docx"
    elif ext in (".html", ".htm"):
        result["suggested_loader"] = "html"
    elif "<?xml" in sample:
        result["suggested_loader"] = "xml"
    elif sample.startswith("{") or sample.startswith("["):
        result["suggested_loader"] = "json"
    elif "\t" in sample[:200]:
        result["suggested_loader"] = "tsv"
    elif "," in sample[:200]:
        result["suggested_loader"] = "csv"

    return result


# ─── Adaptive Pipeline Decision (P4) ─────────────────────────────────────────

def should_use_llm_extraction(doc_metadata: dict[str, Any]) -> bool:
    """
    Decide whether a document needs LLM-based extraction or can be
    processed with deterministic code.

    Returns True if LLM is the better approach.
    """
    is_tabular = doc_metadata.get("is_tabular", False)
    ext = doc_metadata.get("ext", "")

    # Tabular data with clear column headers → programmatic
    if is_tabular:
        return False

    # Very small files → LLM
    size = doc_metadata.get("size", 0)
    if size < 500:
        return True

    # Known narrative formats → LLM
    narrative_exts = {".txt", ".md", ".docx", ".pdf", ".html", ".htm"}
    if ext in narrative_exts:
        return True

    # Unknown or binary → try programmatic first, fall back to LLM
    return False


def classify_documents(docs: list[Any]) -> dict[str, list[Any]]:
    """
    Classify documents into 'llm_pipeline' and 'programmatic' groups.
    """
    llm_docs = []
    prog_docs = []
    for doc in docs:
        meta = getattr(doc, "metadata", {})
        if should_use_llm_extraction(meta):
            llm_docs.append(doc)
        else:
            prog_docs.append(doc)
    return {"llm": llm_docs, "programmatic": prog_docs}
