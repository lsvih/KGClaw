"""
Git-based version management for KGClaw build sessions.

Provides automatic versioning of work directory contents,
including workflow state, ontology, generated code, and outputs.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


class GitManager:
    """Manages git version control for a KGClaw work directory.

    Each complete build run is committed with metadata,
    enabling history tracking and rollback.
    """

    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._initialized: Optional[bool] = None

    # ── Initialization ─────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        """Check if git repo is initialized."""
        if self._initialized is None:
            git_dir = self.work_dir / ".git"
            self._initialized = git_dir.exists() and git_dir.is_dir()
        return self._initialized

    def init(self) -> bool:
        """Initialize git repo if not already initialized. Returns True on success."""
        if self.is_initialized:
            return True

        try:
            subprocess.run(
                ["git", "init"],
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            # Configure local user identity for CI environments where
            # global git config may not be set.
            subprocess.run(
                ["git", "config", "user.name", "KGClaw"],
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "kgclaw@localhost"],
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            self._write_gitignore()
            self._initialized = True
            return True
        except subprocess.CalledProcessError:
            return False

    def _write_gitignore(self):
        """Write .gitignore to exclude transient files from version control."""
        gitignore_content = """\
# KGClaw work directory .gitignore
# Transient/temporary files excluded from version control

logs/
messages_*.json
*.log
__pycache__/
*.pyc
"""
        (self.work_dir / ".gitignore").write_text(gitignore_content)

    # ── Commits ────────────────────────────────────────────────────────────

    def commit_ontology_update(self, description: str = "") -> str:
        """Commit ontology change separately. Returns commit hash or empty string."""
        if not self.is_initialized:
            return ""

        files = ["ontology.json", "ontology.md"]
        return self._stage_and_commit(files, f"ontology: {description}" if description else "ontology: update")

    def commit_build(self, workflow_id: str, summary: dict) -> str:
        """Commit current state with build metadata.

        Args:
            workflow_id: The workflow ID (12-char hex).
            summary: Dict with keys like entities, relations, triples counts.

        Returns:
            Commit hash, or empty string on failure.
        """
        if not self.is_initialized:
            return ""

        entities = summary.get("entities", 0)
        relations = summary.get("relations", 0)
        triples = summary.get("triples", 0)

        message = f"build: {workflow_id} — {entities} entities, {relations} relations, {triples} triples"

        files = [
            "workflow_state.json",
            "ontology.json",
            "ontology.md",
            "output.nt",
            "output.json",
            "output.jsonl",
        ]
        # Include generated_code if it exists
        gen_dir = self.work_dir / "generated_code"
        if gen_dir.exists() and any(gen_dir.iterdir()):
            files.append("generated_code/")

        return self._stage_and_commit(files, message)

    def _stage_and_commit(self, files: list[str], message: str) -> str:
        """Stage specific files and commit. Returns commit hash or empty string."""
        try:
            # Only stage files that exist
            existing = []
            for f in files:
                target = self.work_dir / f
                if target.exists():
                    existing.append(f)
                elif "*" not in f:
                    continue  # skip non-existent non-glob paths
                else:
                    existing.append(f)  # glob patterns pass through

            if not existing:
                return ""

            subprocess.run(
                ["git", "add"] + existing,
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                check=True,
            )

            result = subprocess.run(
                ["git", "commit", "-m", message, "--allow-empty"],
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                check=True,
            )

            # Extract commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            return hash_result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""

    # ── History & Rollback ─────────────────────────────────────────────────

    def get_history(self, n: int = 10) -> list[dict]:
        """Get recent build history from git log.

        Returns list of dicts with keys: hash, date, message.
        """
        if not self.is_initialized:
            return []

        # Use ASCII unit separator (\\x1f) as delimiter to avoid collisions
        # with pipe characters in commit messages.
        _SEP = "\x1f"
        try:
            result = subprocess.run(
                ["git", "log", f"-{n}", f"--format=%h{_SEP}%ad{_SEP}%s", "--date=short"],
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            history = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split(_SEP, 2)
                if len(parts) == 3:
                    history.append({
                        "hash": parts[0],
                        "date": parts[1],
                        "message": parts[2],
                    })
            return history
        except subprocess.CalledProcessError:
            return []

    def rollback(self, commit_hash: str) -> bool:
        """Restore work_dir tracked files to a previous commit. Returns True on success."""
        if not self.is_initialized:
            return False

        try:
            subprocess.run(
                ["git", "checkout", commit_hash, "--", "."],
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def get_current_hash(self) -> str:
        """Get current HEAD commit hash (short)."""
        if not self.is_initialized:
            return ""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""

    def has_commits(self) -> bool:
        """Check if repo has any commits."""
        if not self.is_initialized:
            return False
        try:
            result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            return int(result.stdout.strip()) > 0
        except (subprocess.CalledProcessError, ValueError):
            return False
