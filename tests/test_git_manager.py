"""
Unit tests for GitManager — git-based version management for KGClaw.

Tests init, commit, history, rollback, and edge cases.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from kgclaw.git_manager import GitManager


class TestGitManager:
    """Tests for GitManager version control."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmp = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_gm(self) -> GitManager:
        return GitManager(self.tmp)

    # ── Init ──────────────────────────────────────────────────────────────

    def test_init_creates_git_repo(self):
        gm = self._make_gm()
        assert not gm.is_initialized
        assert gm.init()
        assert gm.is_initialized
        assert (self.tmp / ".git").is_dir()

    def test_init_idempotent(self):
        gm = self._make_gm()
        assert gm.init()
        assert gm.init()  # second init should be fine

    def test_init_writes_gitignore(self):
        gm = self._make_gm()
        gm.init()
        gi = self.tmp / ".gitignore"
        assert gi.exists()
        content = gi.read_text()
        assert "logs/" in content
        assert "messages_*.json" in content
        assert "*.log" in content

    def test_init_not_called_is_initialized_false(self):
        gm = self._make_gm()
        assert not gm.is_initialized

    # ── Commits ──────────────────────────────────────────────────────────

    def test_commit_build_creates_commit(self):
        gm = self._make_gm()
        gm.init()
        # Create files that would normally exist
        (self.tmp / "workflow_state.json").write_text('{"id": "test"}')
        (self.tmp / "ontology.json").write_text('{"name": "test"}')
        (self.tmp / "ontology.md").write_text("# Test")
        (self.tmp / "output.nt").write_text("<a> <b> <c> .")
        (self.tmp / "output.json").write_text('{"entities": []}')

        h = gm.commit_build("abc123def456", {
            "entities": 10, "relations": 5, "triples": 8
        })
        assert h, "Should return commit hash"
        assert gm.has_commits()
        history = gm.get_history(5)
        assert len(history) == 1
        assert "abc123def456" in history[0]["message"]
        assert "10 entities" in history[0]["message"]
        assert "5 relations" in history[0]["message"]
        assert "8 triples" in history[0]["message"]

    def test_commit_build_with_generated_code(self):
        gm = self._make_gm()
        gm.init()
        (self.tmp / "workflow_state.json").write_text("{}")
        (self.tmp / "ontology.json").write_text("{}")
        (self.tmp / "ontology.md").write_text("# Test")
        gen_dir = self.tmp / "generated_code"
        gen_dir.mkdir()
        (gen_dir / "extract_doc_0.py").write_text("print('hello')")

        h = gm.commit_build("test", {"entities": 1, "relations": 0, "triples": 0})
        assert h

    def test_commit_ontology_update(self):
        gm = self._make_gm()
        gm.init()
        (self.tmp / "ontology.json").write_text('{"name": "onto1"}')
        (self.tmp / "ontology.md").write_text("# Ontology v1")

        h = gm.commit_ontology_update("added Person entity")
        assert h
        assert gm.has_commits()
        history = gm.get_history(5)
        assert "ontology:" in history[0]["message"]
        assert "Person" in history[0]["message"]

    def test_commit_no_files_does_nothing(self):
        gm = self._make_gm()
        gm.init()
        h = gm.commit_build("test", {"entities": 0, "relations": 0, "triples": 0})
        assert h == ""  # nothing to commit
        assert not gm.has_commits()

    # ── History ───────────────────────────────────────────────────────────

    def test_get_history_multiple_commits(self):
        gm = self._make_gm()
        gm.init()
        for i in range(3):
            (self.tmp / "workflow_state.json").write_text(f'{{"run": {i}}}')
            (self.tmp / "ontology.json").write_text("{}")
            (self.tmp / "ontology.md").write_text("# Test")
            gm.commit_build(f"run{i:03d}", {"entities": i, "relations": 0, "triples": 0})
        history = gm.get_history(10)
        assert len(history) == 3
        # Most recent first
        assert "run002" in history[0]["message"]

    def test_get_history_limited(self):
        gm = self._make_gm()
        gm.init()
        for i in range(5):
            (self.tmp / "workflow_state.json").write_text(f'{{"run": {i}}}')
            (self.tmp / "ontology.json").write_text("{}")
            (self.tmp / "ontology.md").write_text("# Test")
            gm.commit_build(f"run{i:012d}", {"entities": 0, "relations": 0, "triples": 0})
        history = gm.get_history(2)
        assert len(history) == 2

    def test_get_history_no_commits(self):
        gm = self._make_gm()
        gm.init()
        assert gm.get_history(10) == []

    def test_get_history_uninitialized(self):
        gm = self._make_gm()
        assert gm.get_history(10) == []

    # ── Rollback ──────────────────────────────────────────────────────────

    def test_rollback_restores_file(self):
        gm = self._make_gm()
        gm.init()

        (self.tmp / "workflow_state.json").write_text('{"version": 1}')
        (self.tmp / "ontology.json").write_text("{}")
        (self.tmp / "ontology.md").write_text("# v1")
        h1 = gm.commit_build("aaa", {"entities": 1, "relations": 0, "triples": 0})

        (self.tmp / "workflow_state.json").write_text('{"version": 2}')
        (self.tmp / "ontology.json").write_text("{}")
        (self.tmp / "ontology.md").write_text("# v2")
        gm.commit_build("bbb", {"entities": 2, "relations": 0, "triples": 0})

        assert gm.rollback(h1)
        content = (self.tmp / "workflow_state.json").read_text()
        assert "version" in content

    def test_rollback_uninitialized(self):
        gm = self._make_gm()
        assert not gm.rollback("abc123")

    def test_rollback_bad_hash(self):
        gm = self._make_gm()
        gm.init()
        (self.tmp / "workflow_state.json").write_text("{}")
        (self.tmp / "ontology.json").write_text("{}")
        (self.tmp / "ontology.md").write_text("# Test")
        gm.commit_build("aaa", {"entities": 0, "relations": 0, "triples": 0})
        assert not gm.rollback("nonexistent_hash_12345")

    # ── get_current_hash ──────────────────────────────────────────────────

    def test_get_current_hash_after_commit(self):
        gm = self._make_gm()
        gm.init()
        (self.tmp / "workflow_state.json").write_text("{}")
        (self.tmp / "ontology.json").write_text("{}")
        (self.tmp / "ontology.md").write_text("# Test")
        h = gm.commit_build("test", {"entities": 0, "relations": 0, "triples": 0})
        assert gm.get_current_hash() == h

    def test_get_current_hash_no_commits(self):
        gm = self._make_gm()
        gm.init()
        assert gm.get_current_hash() == ""

    def test_get_current_hash_uninitialized(self):
        gm = self._make_gm()
        assert gm.get_current_hash() == ""

    # ── has_commits ───────────────────────────────────────────────────────

    def test_has_commits_true(self):
        gm = self._make_gm()
        gm.init()
        (self.tmp / "workflow_state.json").write_text("{}")
        (self.tmp / "ontology.json").write_text("{}")
        (self.tmp / "ontology.md").write_text("# Test")
        gm.commit_build("test", {"entities": 0, "relations": 0, "triples": 0})
        assert gm.has_commits()

    def test_has_commits_false(self):
        gm = self._make_gm()
        gm.init()
        assert not gm.has_commits()

    def test_has_commits_uninitialized(self):
        gm = self._make_gm()
        assert not gm.has_commits()
