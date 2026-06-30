"""
Unit tests for extended Memory features.

Tests ontology export, generated code persistence,
document manifest, and file change detection.
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from kgclaw.memory import Memory
from kgclaw.models import (
    Document,
    EntityType,
    Ontology,
    RelationType,
)


class TestMemoryExtended:
    """Tests for Memory features added post-v0.1.0."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mem = Memory(work_dir=str(self.tmp))
        yield
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_ontology(self, name="test") -> Ontology:
        return Ontology(
            name=name,
            description="A test ontology for unit tests",
            entity_types=[
                EntityType(name="人物", description="自然人"),
                EntityType(name="地点", description="地理位置"),
            ],
            relation_types=[
                RelationType(name="生父", description="生物学父亲",
                             domain="人物", range="人物"),
                RelationType(name="居住于", description="居住在某地",
                             domain="人物", range="地点"),
            ],
            raw_definition="Entity Types: 人物, 地点\nRelation Types: 生父, 居住于",
        )

    # ── Ontology Export ──────────────────────────────────────────────────

    def test_export_ontology_creates_json_and_md(self):
        onto = self._make_ontology()
        self.mem.init_workflow(ontology=onto)
        json_path, md_path = self.mem.export_ontology()

        assert json_path.exists()
        assert md_path.exists()
        assert json_path.suffix == ".json"
        assert md_path.suffix == ".md"

    def test_export_ontology_json_content(self):
        onto = self._make_ontology()
        self.mem.init_workflow(ontology=onto)
        json_path, _ = self.mem.export_ontology()

        data = json.loads(json_path.read_text())
        assert data["name"] == "test"
        assert len(data["entity_types"]) == 2
        assert data["entity_types"][0]["name"] == "人物"
        assert len(data["relation_types"]) == 2
        assert data["raw_definition"] is not None

    def test_export_ontology_md_content(self):
        onto = self._make_ontology()
        self.mem.init_workflow(ontology=onto)
        _, md_path = self.mem.export_ontology()

        content = md_path.read_text()
        assert "# Ontology: test" in content
        assert "人物" in content
        assert "地点" in content
        assert "生父" in content
        assert "居住于" in content
        assert "`人物` → `地点`" in content
        assert "Raw Definition" in content

    def test_export_ontology_no_workflow_writes_empty(self):
        json_path, md_path = self.mem.export_ontology()
        assert json_path.exists()
        assert md_path.exists()
        assert json.loads(json_path.read_text()) == {}
        assert "No ontology" in md_path.read_text()

    def test_export_ontology_custom_path(self):
        onto = self._make_ontology()
        self.mem.init_workflow(ontology=onto)
        custom = self.tmp / "custom_onto.json"
        json_path, md_path = self.mem.export_ontology(output_path=str(custom))
        assert json_path == custom
        assert json_path.exists()
        assert md_path == custom.with_suffix(".md")

    def test_export_ontology_no_entity_types(self):
        onto = Ontology(name="empty", raw_definition="")
        self.mem.init_workflow(ontology=onto)
        json_path, md_path = self.mem.export_ontology()
        content = md_path.read_text()
        assert "empty" in content

    # ── Generated Code Persistence ───────────────────────────────────────

    def test_save_generated_code_creates_dir_and_file(self):
        gen_dir = self.tmp / "generated_code"
        assert not gen_dir.exists()

        path = self.mem.save_generated_code(
            "extract_doc_0_20260624T120000.py",
            'print("hello world")',
        )
        assert gen_dir.is_dir()
        assert path.exists()
        assert path.parent == gen_dir
        assert path.read_text() == 'print("hello world")'

    def test_save_generated_code_multiple_files(self):
        paths = []
        for i in range(5):
            p = self.mem.save_generated_code(
                f"extract_doc_{i}.py",
                f"# extraction code {i}\nprint({i})",
            )
            paths.append(p)

        gen_dir = self.tmp / "generated_code"
        files = sorted(gen_dir.iterdir())
        assert len(files) == 5
        for i, f in enumerate(files):
            assert f"extract_doc_{i}" in f.name

    def test_save_generated_code_empty_code(self):
        path = self.mem.save_generated_code("empty.py", "")
        assert path.exists()
        assert path.read_text() == ""

    def test_save_generated_code_unicode(self):
        path = self.mem.save_generated_code(
            "chinese_extract.py",
            '# 提取"人物"实体\nprint("你好")',
        )
        assert "人物" in path.read_text()

    # ── Document Manifest ────────────────────────────────────────────────

    def test_save_and_load_manifest(self):
        docs = [
            Document(
                text="content1",
                source="/path/to/file1.txt",
                metadata={
                    "content_hash": "abc123",
                    "file_mtime": 1234567890.0,
                    "file_size": 100,
                },
            ),
            Document(
                text="content2",
                source="/path/to/file2.jsonl",
                metadata={
                    "content_hash": "def456",
                    "file_mtime": 1234567891.0,
                    "file_size": 200,
                },
            ),
        ]
        self.mem.save_document_manifest(docs)

        manifest = self.mem.load_document_manifest()
        assert len(manifest) == 2
        assert "/path/to/file1.txt" in manifest
        assert manifest["/path/to/file1.txt"]["content_hash"] == "abc123"
        assert manifest["/path/to/file1.txt"]["size"] == 100
        assert manifest["/path/to/file2.jsonl"]["content_hash"] == "def456"

    def test_save_manifest_skips_no_source(self):
        docs = [
            Document(text="no source doc", source=""),
            Document(text="has source doc", source="/path/to/file.txt",
                     metadata={"content_hash": "abc"}),
        ]
        self.mem.save_document_manifest(docs)
        manifest = self.mem.load_document_manifest()
        assert len(manifest) == 1
        assert "/path/to/file.txt" in manifest

    def test_load_manifest_no_file_returns_empty(self):
        assert self.mem.load_document_manifest() == {}

    def test_load_manifest_corrupt_json(self):
        (self.tmp / "document_manifest.json").write_text("{not valid json")
        assert self.mem.load_document_manifest() == {}

    def test_manifest_includes_last_processed_at(self):
        docs = [Document(text="x", source="/f.txt", metadata={"content_hash": "abc"})]
        self.mem.save_document_manifest(docs)
        manifest = self.mem.load_document_manifest()
        assert "last_processed_at" in manifest["/f.txt"]

    # ── File Change Detection ────────────────────────────────────────────

    def _make_file_with_md5(self, path: Path, content: str) -> str:
        path.write_text(content)
        return hashlib.md5(path.read_bytes()).hexdigest()

    def test_detect_all_unchanged(self):
        f1 = self.tmp / "doc1.txt"
        f2 = self.tmp / "doc2.txt"
        h1 = self._make_file_with_md5(f1, "hello world")
        h2 = self._make_file_with_md5(f2, "foo bar baz")

        docs = [
            Document(text="hello world", source=str(f1),
                     metadata={"content_hash": h1, "file_mtime": f1.stat().st_mtime, "file_size": f1.stat().st_size}),
            Document(text="foo bar baz", source=str(f2),
                     metadata={"content_hash": h2, "file_mtime": f2.stat().st_mtime, "file_size": f2.stat().st_size}),
        ]
        self.mem.save_document_manifest(docs)

        changes = self.mem.detect_file_changes([str(f1), str(f2)])
        assert len(changes["unchanged"]) == 2
        assert changes["added"] == []
        assert changes["modified"] == []
        assert changes["deleted"] == []

    def test_detect_added_files(self):
        f1 = self.tmp / "existing.txt"
        f2 = self.tmp / "new_file.txt"
        h1 = self._make_file_with_md5(f1, "existing content")
        self._make_file_with_md5(f2, "new content")

        docs = [Document(text="existing", source=str(f1),
                         metadata={"content_hash": h1, "file_mtime": f1.stat().st_mtime, "file_size": f1.stat().st_size})]
        self.mem.save_document_manifest(docs)

        changes = self.mem.detect_file_changes([str(f1), str(f2)])
        assert str(f1) in changes["unchanged"]
        assert str(f2) in changes["added"]
        assert changes["modified"] == []
        assert changes["deleted"] == []

    def test_detect_modified_file(self):
        f1 = self.tmp / "doc.txt"
        self._make_file_with_md5(f1, "original content")

        # Store manifest with old hash
        docs = [Document(text="old", source=str(f1),
                         metadata={"content_hash": "00000000000000000000000000000000",
                                   "file_mtime": f1.stat().st_mtime, "file_size": f1.stat().st_size})]
        self.mem.save_document_manifest(docs)

        changes = self.mem.detect_file_changes([str(f1)])
        assert str(f1) in changes["modified"]
        assert changes["unchanged"] == []

    def test_detect_deleted_file(self):
        f1 = self.tmp / "will_be_deleted.txt"
        h1 = self._make_file_with_md5(f1, "temporary content")

        docs = [Document(text="tmp", source=str(f1),
                         metadata={"content_hash": h1, "file_mtime": f1.stat().st_mtime, "file_size": f1.stat().st_size})]
        self.mem.save_document_manifest(docs)

        # Delete the file
        f1.unlink()

        changes = self.mem.detect_file_changes([])  # no current files
        assert str(f1) in changes["deleted"]

    def test_detect_mixed_changes(self):
        # unchanged
        f_unchanged = self.tmp / "unchanged.txt"
        h_unchanged = self._make_file_with_md5(f_unchanged, "same")

        # will be modified
        f_modified = self.tmp / "modified.txt"
        self._make_file_with_md5(f_modified, "old content")

        # will be deleted (not created yet, only in manifest)
        f_deleted_path = str(self.tmp / "deleted.txt")

        # will be added
        f_added = self.tmp / "added.txt"
        self._make_file_with_md5(f_added, "new stuff")

        docs = [
            Document(text="same", source=str(f_unchanged),
                     metadata={"content_hash": h_unchanged, "file_mtime": f_unchanged.stat().st_mtime, "file_size": f_unchanged.stat().st_size}),
            Document(text="old", source=str(f_modified),
                     metadata={"content_hash": "00000000000000000000000000000000",
                               "file_mtime": f_modified.stat().st_mtime, "file_size": f_modified.stat().st_size}),
            Document(text="gone", source=f_deleted_path,
                     metadata={"content_hash": "11111111111111111111111111111111",
                               "file_mtime": 0, "file_size": 0}),
        ]
        self.mem.save_document_manifest(docs)

        changes = self.mem.detect_file_changes([str(f_unchanged), str(f_modified), str(f_added)])
        assert str(f_unchanged) in changes["unchanged"]
        assert str(f_modified) in changes["modified"]
        assert str(f_added) in changes["added"]
        assert f_deleted_path in changes["deleted"]

    def test_detect_no_manifest_all_added(self):
        f1 = self.tmp / "doc1.txt"
        f2 = self.tmp / "doc2.txt"
        self._make_file_with_md5(f1, "content1")
        self._make_file_with_md5(f2, "content2")

        changes = self.mem.detect_file_changes([str(f1), str(f2)])
        assert len(changes["added"]) == 2
        assert changes["unchanged"] == []

    def test_detect_unreadable_file_skipped(self):
        f1 = self.tmp / "readable.txt"
        self._make_file_with_md5(f1, "readable")

        changes = self.mem.detect_file_changes([str(f1), "/nonexistent/path/file.txt"])
        assert str(f1) in changes["added"]
        # nonexistent paths are silently skipped (is_file() check)

    def test_detect_change_preserves_hash_order(self):
        """Verify that repeated detection yields consistent results."""
        f1 = self.tmp / "stable.txt"
        h1 = self._make_file_with_md5(f1, "stable content")

        docs = [Document(text="stable", source=str(f1),
                         metadata={"content_hash": h1, "file_mtime": f1.stat().st_mtime, "file_size": f1.stat().st_size})]
        self.mem.save_document_manifest(docs)

        for _ in range(3):
            changes = self.mem.detect_file_changes([str(f1)])
            assert str(f1) in changes["unchanged"]
