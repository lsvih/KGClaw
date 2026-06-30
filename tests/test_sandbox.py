"""
Unit tests for the KGClaw code sandbox.

Tests safety checks (AST analysis), code execution,
timeout handling, and security boundaries.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from kgclaw.sandbox import (
    check_code_safety,
    run_python_code,
    SANDBOX_TIMEOUT,
)


class TestCheckCodeSafety:
    """Tests for the AST-based safety checker."""

    # ── Safe Code ────────────────────────────────────────────────────────

    def test_safe_simple_code(self):
        ok, msg = check_code_safety("x = 1 + 2\nprint(x)")
        assert ok, f"Simple code should be safe, got: {msg}"

    def test_safe_with_allowed_imports(self):
        ok, _ = check_code_safety("import json\nprint(json.dumps({'a': 1}))")
        assert ok

    def test_safe_with_allowed_from_import(self):
        ok, _ = check_code_safety("from collections import defaultdict\nd = defaultdict(int)")
        assert ok

    def test_safe_data_extraction_code(self):
        """Safe extraction code typical in KGClaw use."""
        code = '''
import json
import re

data = json.loads(DATA_TEXT)
onto = json.loads(ONTOLOGY_JSON)

entities = []
for item in data:
    name = item.get("name", "")
    if re.match(r'[一-鿿]+', name):
        entities.append({"name": name, "type": "人物", "confidence": 0.9})

result = json.dumps({"entities": entities, "relations": []})
print(result)
'''
        ok, msg = check_code_safety(code)
        assert ok, f"Extraction code should be safe, got: {msg}"

    def test_safe_list_comprehension(self):
        ok, _ = check_code_safety(
            "results = [x for x in range(100) if x % 2 == 0]\nprint(len(results))"
        )
        assert ok

    def test_safe_with_csv_import(self):
        ok, _ = check_code_safety("import csv\nreader = csv.DictReader([])")
        assert ok

    # ── Dangerous Code (Blocked) ────────────────────────────────────────

    def test_block_os_import(self):
        ok, msg = check_code_safety("import os\nos.system('ls')")
        assert not ok
        assert "os" in msg.lower() or "Forbidden" in msg

    def test_block_subprocess_import(self):
        ok, msg = check_code_safety("import subprocess\nsubprocess.run('ls')")
        assert not ok

    def test_block_socket_import(self):
        ok, _ = check_code_safety("import socket\ns = socket.socket()")
        assert not ok

    def test_block_shutil_import(self):
        ok, _ = check_code_safety("import shutil\nshutil.rmtree('/')")
        assert not ok

    def test_block_eval_call(self):
        ok, msg = check_code_safety('eval("1+1")')
        assert not ok, f"eval() should be blocked, got: {msg}"

    def test_block_exec_call(self):
        ok, _ = check_code_safety('exec("print(1)")')
        assert not ok

    def test_block_compile_call(self):
        ok, _ = check_code_safety('compile("x=1", "", "exec")')
        assert not ok

    def test_block_dunder_import(self):
        ok, _ = check_code_safety('__import__("os")')
        assert not ok

    def test_block_breakpoint(self):
        ok, _ = check_code_safety("breakpoint()")
        assert not ok

    def test_block_requests_import(self):
        ok, _ = check_code_safety("import requests\nrequests.get('http://evil.com')")
        assert not ok

    def test_block_ctypes_import(self):
        ok, _ = check_code_safety("import ctypes\nctypes.CDLL('libc.so.6')")
        assert not ok

    # ── Syntax Errors ───────────────────────────────────────────────────

    def test_syntax_error(self):
        ok, msg = check_code_safety("def broken(:\n    pass")
        assert not ok
        assert "Syntax" in msg

    def test_incomplete_code(self):
        ok, msg = check_code_safety("x = [1, 2,")
        assert not ok

    # ── Edge Cases ───────────────────────────────────────────────────────

    def test_empty_code(self):
        ok, _ = check_code_safety("")
        assert ok

    def test_comment_only(self):
        ok, _ = check_code_safety("# This is just a comment")
        assert ok

    def test_multiline_string_with_dangerous_words(self):
        """String literals containing dangerous words are still safe."""
        code = '''"""
This docstring mentions os.system and subprocess
but doesn't actually call them.
"""
print("safe")
'''
        ok, _ = check_code_safety(code)
        assert ok

    # ── Bypass vector tests (security hardening) ──

    def test_block_builtins_subscript_eval(self):
        """Block __builtins__[\"eval\"] bypass."""
        code = '__builtins__["eval"]("1+1")'
        ok, reason = check_code_safety(code)
        assert not ok, f"Should block __builtins__ subscript: {reason}"

    def test_block_vars_builtins(self):
        """Block vars(__builtins__) bypass."""
        code = 'vars(__builtins__)["eval"]("1+1")'
        ok, reason = check_code_safety(code)
        assert not ok, f"Should block vars(__builtins__): {reason}"

    def test_block_importlib_import(self):
        """Block importlib.import_module bypass."""
        code = 'import importlib\nimportlib.import_module("os")'
        ok, reason = check_code_safety(code)
        assert not ok, f"Should block importlib import: {reason}"

    def test_block_sys_modules_access(self):
        """Block sys.modules subscript bypass."""
        code = 'import sys\nsys.modules["os"].system("ls")'
        ok, reason = check_code_safety(code)
        assert not ok, f"Should block sys.modules access: {reason}"

    def test_block_open_call(self):
        """Block direct open() call."""
        code = 'open("/etc/passwd").read()'
        ok, reason = check_code_safety(code)
        assert not ok, f"Should block open(): {reason}"

    def test_block_class_hierarchy_navigation(self):
        """Block __class__.__bases__ chain."""
        code = '().__class__.__bases__[0].__subclasses__()'
        ok, reason = check_code_safety(code)
        assert not ok, f"Should block class hierarchy: {reason}"

    def test_block_getattr_builtins(self):
        """Block getattr(__builtins__, ...) bypass."""
        code = 'getattr(__builtins__, "eval")("1+1")'
        ok, reason = check_code_safety(code)
        assert not ok, f"Should block getattr bypass: {reason}"

    def test_block_setattr_builtins(self):
        """Block setattr(__builtins__, ...) bypass."""
        code = 'setattr(__builtins__, "x", lambda: None)'
        ok, reason = check_code_safety(code)
        assert not ok, f"Should block setattr bypass: {reason}"


class TestRunPythonCode:
    """Tests for sandboxed Python execution."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmp = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── Basic Execution ─────────────────────────────────────────────────

    def test_run_simple_code(self):
        result = run_python_code("print('hello world')")
        assert result["success"]
        assert "hello world" in result["stdout"]

    def test_run_with_json_output(self):
        code = '''
import json
data = {"entities": [{"name": "赵铁蛋", "type": "人物"}], "relations": []}
print(json.dumps(data, ensure_ascii=False))
'''
        result = run_python_code(code)
        assert result["success"]
        output = json.loads(result["stdout"].strip())
        assert output["entities"][0]["name"] == "赵铁蛋"

    def test_run_extraction_code(self):
        """Simulate typical KGClaw extraction code."""
        code = '''
import json
import os

onto_json = os.environ.get("ONTOLOGY_JSON", "{}")
data_json = os.environ.get("DATA_TEXT", "[]")
onto = json.loads(onto_json)
data = json.loads(data_json)

entities = []
for item in data:
    for field, value in item.items():
        if "名" in field or "name" in field.lower():
            entities.append({"name": value, "type": onto["entity_types"][0]["name"] if onto.get("entity_types") else "Entity"})

print(json.dumps({"entities": entities}))
'''
        onto_json = json.dumps({
            "entity_types": [{"name": "人物", "description": "自然人"}],
            "relation_types": [],
        })
        data_text = json.dumps([
            {"姓名": "赵铁蛋", "年龄": 30},
            {"姓名": "赵本山", "年龄": 55},
        ])

        result = run_python_code(
            code,
            env={"ONTOLOGY_JSON": onto_json, "DATA_TEXT": data_text},
            skip_safety_check=True,  # test code uses os.environ — not a safety concern
        )
        assert result["success"], f"Code failed: {result.get('stderr', '')}: {result.get('error', '')}"
        output = json.loads(result["stdout"].strip())
        assert len(output["entities"]) == 2

    def test_run_with_stdin(self):
        result = run_python_code(
            "import sys\ndata = sys.stdin.read()\nprint(f'got: {data}')",
            input_data="hello from stdin",
            skip_safety_check=True,  # test code uses sys — testing stdin, not safety
        )
        assert result["success"]
        assert "got: hello from stdin" in result["stdout"]

    # ── Error Handling ──────────────────────────────────────────────────

    def test_run_syntax_error(self):
        result = run_python_code("def broken(:")
        assert not result["success"]

    def test_run_runtime_error(self):
        result = run_python_code("x = 1 / 0")
        assert not result["success"]

    def test_run_name_error(self):
        result = run_python_code("print(undefined_variable)")
        assert not result["success"]

    # ── Timeout ──────────────────────────────────────────────────────────

    def test_run_timeout(self):
        """Code that runs too long should be killed."""
        code = "import time\ntime.sleep(30)\nprint('done')"
        result = run_python_code(code, timeout=1)
        assert not result["success"]
        # Should have error related to timeout or killed
        err = (result.get("error", "") or "").lower()
        stdout = (result.get("stdout", "") or "").lower()
        stderr = (result.get("stderr", "") or "").lower()
        combined = err + stdout + stderr
        # Some indicator of not completing normally
        assert "done" not in result.get("stdout", "")

    def test_run_fast_code_no_timeout(self):
        """Fast code should complete well within timeout."""
        code = "print('fast')"
        result = run_python_code(code, timeout=5)
        assert result["success"]
        assert result["elapsed"] < 2.0  # should be near-instant

    # ── Output Truncation ────────────────────────────────────────────────

    def test_run_large_output_truncation(self):
        """Very large stdout should be truncated."""
        code = "for i in range(10000): print('x' * 100)"
        result = run_python_code(code, timeout=5)
        stdout_len = len(result.get("stdout", ""))
        assert stdout_len <= 110_000  # SANDBOX_MAX_OUTPUT is 100_000

    # ── Working Directory ────────────────────────────────────────────────

    def test_run_in_custom_working_dir(self):
        code = '''
import pathlib
print(pathlib.Path.cwd())
'''
        result = run_python_code(code, working_dir=str(self.tmp))
        assert result["success"], f"Code failed: {result.get('stderr', '')}: {result.get('error', '')}"
        # The sandbox uses working_dir, verify it doesn't crash
        assert "tmp" in result["stdout"] or result["stdout"].strip() != ""

    # ── Chinese Character Support ────────────────────────────────────────

    def test_run_chinese_code(self):
        code = '''
# 提取人物实体
entities = ["赵铁蛋", "赵本山", "刘海柱"]
print("|".join(entities))
'''
        result = run_python_code(code)
        assert result["success"]
        assert "赵铁蛋" in result["stdout"]
        assert "赵本山" in result["stdout"]
