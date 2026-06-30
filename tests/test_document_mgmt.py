"""
Tests for document management: Memory unload/clear/list, Harness unload/clear/list,
and interactive REPL /docs /unload /clear-docs commands.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from kgclaw.harness.engine import Harness
from kgclaw.interactive_app import Session
from kgclaw.memory import Memory
from kgclaw.models import (
    Document,
    HarnessConfig,
    LLMConfig,
    Ontology,
    WorkflowState,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def memory(tmp_dir):
    return Memory(work_dir=str(tmp_dir))


@pytest.fixture
def memory_with_docs(memory, tmp_dir):
    """Memory with a workflow containing 3 documents."""
    f1 = tmp_dir / "doc1.txt"
    f2 = tmp_dir / "doc2.csv"
    f3 = tmp_dir / "doc3.md"
    f1.write_text("hello world")
    f2.write_text("a,b,c\n1,2,3")
    f3.write_text("# Title\nSome content here")
    memory.init_workflow()
    memory.workflow.documents = [
        Document(text="hello world", source=str(f1), metadata={"filename": "doc1.txt", "ext": ".txt", "file_size": 11}),
        Document(text="a,b,c\n1,2,3", source=str(f2), metadata={"filename": "doc2.csv", "ext": ".csv", "file_size": 13, "is_tabular": True, "raw_rows": [{"a": "1", "b": "2", "c": "3"}]}),
        Document(text="# Title\nSome content here", source=str(f3), metadata={"filename": "doc3.md", "ext": ".md", "file_size": 26}),
    ]
    memory.save_workflow()
    return memory


@pytest.fixture
def harness(tmp_dir):
    config = HarnessConfig(
        llm=LLMConfig(api_key="sk-test", model="test-model"),
        work_dir=str(tmp_dir),
    )
    return Harness(config)


# ─── Memory Document Management ───────────────────────────────────────────────

class TestMemoryDocumentManagement:

    def test_remove_document_existing(self, memory_with_docs):
        m = memory_with_docs
        docs = m.get_loaded_documents()
        source = docs[0]["source"]
        assert m.remove_document(source) is True
        assert m.has_document(source) is False
        assert len(m.get_loaded_documents()) == 2

    def test_remove_document_not_found(self, memory_with_docs):
        assert memory_with_docs.remove_document("/nonexistent/path.txt") is False

    def test_remove_document_no_workflow(self, memory):
        assert memory.remove_document("/any/path.txt") is False

    def test_remove_documents_multiple(self, memory_with_docs):
        m = memory_with_docs
        docs = m.get_loaded_documents()
        sources = [docs[0]["source"], docs[1]["source"]]
        assert m.remove_documents(sources) == 2
        assert len(m.get_loaded_documents()) == 1

    def test_remove_documents_empty_list(self, memory_with_docs):
        assert memory_with_docs.remove_documents([]) == 0

    def test_remove_documents_some_not_found(self, memory_with_docs):
        m = memory_with_docs
        docs = m.get_loaded_documents()
        sources = [docs[0]["source"], "/nonexistent/file.txt"]
        assert m.remove_documents(sources) == 1

    def test_remove_documents_no_workflow(self, memory):
        assert memory.remove_documents(["/a.txt"]) == 0

    def test_clear_documents_with_docs(self, memory_with_docs):
        assert memory_with_docs.clear_documents() == 3
        assert len(memory_with_docs.get_loaded_documents()) == 0

    def test_clear_documents_empty(self, memory):
        memory.init_workflow()
        assert memory.clear_documents() == 0

    def test_clear_documents_no_workflow(self, memory):
        assert memory.clear_documents() == 0

    def test_get_loaded_documents_empty(self, memory):
        memory.init_workflow()
        assert memory.get_loaded_documents() == []

    def test_get_loaded_documents_no_workflow(self, memory):
        assert memory.get_loaded_documents() == []

    def test_get_loaded_documents_with_docs(self, memory_with_docs):
        docs = memory_with_docs.get_loaded_documents()
        assert len(docs) == 3
        # Verify structure of each entry
        for d in docs:
            assert "source" in d
            assert "filename" in d
            assert "chars" in d
            assert "ext" in d
            assert "size" in d
            assert "is_tabular" in d
            assert isinstance(d["chars"], int)
            assert d["chars"] > 0

    def test_get_loaded_documents_metadata(self, memory_with_docs):
        docs = memory_with_docs.get_loaded_documents()
        filenames = [d["filename"] for d in docs]
        assert "doc1.txt" in filenames
        assert "doc2.csv" in filenames
        assert "doc3.md" in filenames
        # doc2.csv is tabular
        csv_doc = [d for d in docs if d["filename"] == "doc2.csv"][0]
        assert csv_doc["is_tabular"] is True
        assert csv_doc["chars"] == 11

    def test_has_document_true(self, memory_with_docs):
        docs = memory_with_docs.get_loaded_documents()
        assert memory_with_docs.has_document(docs[0]["source"]) is True

    def test_has_document_false(self, memory_with_docs):
        assert memory_with_docs.has_document("/no/such/file.txt") is False

    def test_has_document_no_workflow(self, memory):
        assert memory.has_document("/any.txt") is False

    def test_remove_document_persists(self, memory_with_docs, tmp_dir):
        """Verify that removal is persisted to workflow_state.json."""
        m = memory_with_docs
        docs = m.get_loaded_documents()
        m.remove_document(docs[0]["source"])
        # Reload from disk
        m2 = Memory(work_dir=str(tmp_dir))
        m2.load_workflow()
        assert len(m2.get_loaded_documents()) == 2

    def test_clear_documents_persists(self, memory_with_docs, tmp_dir):
        """Verify that clear is persisted."""
        memory_with_docs.clear_documents()
        m2 = Memory(work_dir=str(tmp_dir))
        m2.load_workflow()
        assert len(m2.get_loaded_documents()) == 0


# ─── Harness Document Management ──────────────────────────────────────────────

class TestHarnessDocumentManagement:

    def test_unload_document_existing(self, harness, tmp_dir):
        f = tmp_dir / "test.txt"
        f.write_text("hello harness world")
        harness.load_documents([str(f)])
        assert len(harness.list_documents()) == 1
        assert harness.unload_document(str(f)) is True
        assert len(harness.list_documents()) == 0

    def test_unload_document_not_found(self, harness, tmp_dir):
        f = tmp_dir / "test.txt"
        f.write_text("hello")
        harness.load_documents([str(f)])
        assert harness.unload_document("/no/such/file.txt") is False

    def test_unload_document_event(self, harness, tmp_dir):
        f = tmp_dir / "test.txt"
        f.write_text("event test")
        harness.load_documents([str(f)])
        events = []
        harness.on_event(lambda et, d: events.append((et, d)))
        harness.unload_document(str(f))
        unload_events = [e for e in events if e[0] == "documents_unloaded"]
        assert len(unload_events) == 1
        assert unload_events[0][1]["count"] == 1
        assert unload_events[0][1]["remaining"] == 0

    def test_unload_documents_multiple(self, harness, tmp_dir):
        f1 = tmp_dir / "a.txt"
        f2 = tmp_dir / "b.txt"
        f3 = tmp_dir / "c.txt"
        f1.write_text("a")
        f2.write_text("b")
        f3.write_text("c")
        harness.load_documents([str(f1), str(f2), str(f3)])
        assert harness.unload_documents([str(f1), str(f2)]) == 2
        assert len(harness.list_documents()) == 1

    def test_unload_documents_some_not_found(self, harness, tmp_dir):
        f1 = tmp_dir / "a.txt"
        f1.write_text("a")
        harness.load_documents([str(f1)])
        assert harness.unload_documents([str(f1), "/ghost.txt"]) == 1

    def test_clear_documents(self, harness, tmp_dir):
        f1 = tmp_dir / "x.txt"
        f2 = tmp_dir / "y.txt"
        f1.write_text("x")
        f2.write_text("y")
        harness.load_documents([str(f1), str(f2)])
        assert harness.clear_documents() == 2
        assert len(harness.list_documents()) == 0

    def test_clear_documents_event(self, harness, tmp_dir):
        f = tmp_dir / "z.txt"
        f.write_text("z")
        harness.load_documents([str(f)])
        events = []
        harness.on_event(lambda et, d: events.append((et, d)))
        harness.clear_documents()
        clear_events = [e for e in events if e[0] == "documents_cleared"]
        assert len(clear_events) == 1
        assert clear_events[0][1]["count"] == 1

    def test_clear_documents_no_docs(self, harness):
        assert harness.clear_documents() == 0

    def test_list_documents_empty(self, harness):
        assert harness.list_documents() == []

    def test_list_documents_populated(self, harness, tmp_dir):
        f = tmp_dir / "sample.txt"
        f.write_text("sample content here")
        harness.load_documents([str(f)])
        docs = harness.list_documents()
        assert len(docs) == 1
        assert docs[0]["source"] == str(f)
        assert docs[0]["chars"] == 19
        assert docs[0]["ext"] == ".txt"

    def test_unload_preserves_ontology(self, harness, tmp_dir):
        f = tmp_dir / "data.txt"
        f.write_text("some data")
        harness.load_documents([str(f)])
        harness.set_ontology("Entity: Person\nRelation: knows")
        harness.unload_document(str(f))
        wf = harness.memory.workflow
        assert wf is not None
        assert wf.ontology is not None
        assert "Person" in wf.ontology.raw_definition

    def test_unload_refreshes_structured_rows(self, harness, tmp_dir):
        """Unloading a tabular document should clear its structured rows."""
        import csv
        f_csv = tmp_dir / "data.csv"
        f_txt = tmp_dir / "data.txt"
        f_csv.write_text("name,age\nAlice,30\nBob,25")
        f_txt.write_text("plain text")
        harness.load_documents([str(f_csv), str(f_txt)])
        assert len(harness._structured_rows) == 2  # Alice + Bob
        harness.unload_document(str(f_csv))
        assert len(harness._structured_rows) == 0

    def test_clear_documents_resets_structured_rows(self, harness, tmp_dir):
        import csv
        f_csv = tmp_dir / "data.csv"
        f_csv.write_text("x,y\n1,2\n3,4")
        harness.load_documents([str(f_csv)])
        assert len(harness._structured_rows) == 2
        harness.clear_documents()
        assert len(harness._structured_rows) == 0

    def test_load_texts_resets_structured_rows(self, harness, tmp_dir):
        """Bug fix: load_texts should clear structured_rows."""
        # First load a tabular file to populate structured_rows
        f_csv = tmp_dir / "data.csv"
        f_csv.write_text("a,b\n1,2")
        harness.load_documents([str(f_csv)])
        assert len(harness._structured_rows) == 1
        # Then load from texts — should reset structured_rows
        harness.load_texts(["plain text document"])
        assert len(harness._structured_rows) == 0


# ─── Interactive REPL Commands ────────────────────────────────────────────────

class TestInteractiveCommands:
    """Integration tests for _cmd_docs, _cmd_unload, _cmd_clear_docs via Session."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.tmp = tmp_dir
        self.session = Session(
            api_key="sk-test",
            api_base="https://test.api/v1",
            model="test-model",
            work_dir=str(tmp_dir),
        )

    def test_cmd_docs_empty(self):
        """_cmd_docs on empty session should not raise."""
        from kgclaw.interactive_app import _cmd_docs
        _cmd_docs(self.session)  # Should print empty message, not crash

    def test_cmd_docs_populated(self):
        """_cmd_docs should display loaded documents."""
        f = self.tmp / "hello.txt"
        f.write_text("hello world")
        self.session.load_docs([str(f)])
        from kgclaw.interactive_app import _cmd_docs
        _cmd_docs(self.session)  # Should display the table

    def test_cmd_unload_no_arg(self):
        """_cmd_unload with empty arg should show usage."""
        from kgclaw.interactive_app import _cmd_unload
        _cmd_unload(self.session, "")  # Should show usage, not crash

    def test_cmd_unload_by_full_path(self):
        """Unload by exact source path."""
        f = self.tmp / "remove_me.txt"
        f.write_text("to be removed")
        self.session.load_docs([str(f)])
        assert len(self.session.harness.list_documents()) == 1
        from kgclaw.interactive_app import _cmd_unload
        _cmd_unload(self.session, str(f))
        assert len(self.session.harness.list_documents()) == 0
        assert self.session.documents_loaded is False

    def test_cmd_unload_by_basename(self):
        """Unload by filename only (basename match)."""
        f = self.tmp / "unique_name.txt"
        f.write_text("content")
        self.session.load_docs([str(f)])
        from kgclaw.interactive_app import _cmd_unload
        _cmd_unload(self.session, "unique_name.txt")
        assert len(self.session.harness.list_documents()) == 0

    def test_cmd_unload_not_found(self):
        """Unload non-existent file should show error."""
        f = self.tmp / "exists.txt"
        f.write_text("here")
        self.session.load_docs([str(f)])
        from kgclaw.interactive_app import _cmd_unload
        _cmd_unload(self.session, "ghost_file.txt")  # Should show error, not crash

    def test_cmd_unload_empty_docs(self):
        """Unload when no documents are loaded."""
        from kgclaw.interactive_app import _cmd_unload
        _cmd_unload(self.session, "anything.txt")  # Should not crash

    def test_cmd_clear_docs_empty(self):
        """Clear when no documents loaded."""
        from kgclaw.interactive_app import _cmd_clear_docs
        _cmd_clear_docs(self.session)  # Should not crash
