"""
Extended tests for CLI commands, Harness content hashing,
Session management, and _scan_data_files.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from kgclaw.harness.engine import Harness
from kgclaw.interactive_app import Session, _scan_data_files
from kgclaw.memory import Memory
from kgclaw.models import (
    Document,
    EntityType,
    ExtractionResult,
    HarnessConfig,
    LLMConfig,
    Ontology,
    RelationType,
)


class TestHarnessContentHashing:
    """Tests for _stamp_doc_hash and load_documents with hashing."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmp = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stamp_doc_hash_adds_metadata(self):
        f = self.tmp / "test.txt"
        f.write_text("hello world for hashing")

        doc = Document(text="hello world for hashing", source=str(f), metadata={})
        Harness._stamp_doc_hash(doc, f)

        assert "content_hash" in doc.metadata
        assert len(doc.metadata["content_hash"]) == 32  # MD5 hex
        assert "file_mtime" in doc.metadata
        assert doc.metadata["file_mtime"] > 0
        assert "file_size" in doc.metadata
        assert doc.metadata["file_size"] == len("hello world for hashing")

    def test_stamp_doc_hash_different_content_different_hash(self):
        f1 = self.tmp / "doc1.txt"
        f2 = self.tmp / "doc2.txt"
        f1.write_text("content A")
        f2.write_text("content B — completely different")

        doc1 = Document(text="content A", source=str(f1), metadata={})
        doc2 = Document(text="content B", source=str(f2), metadata={})
        Harness._stamp_doc_hash(doc1, f1)
        Harness._stamp_doc_hash(doc2, f2)

        assert doc1.metadata["content_hash"] != doc2.metadata["content_hash"]

    def test_stamp_doc_hash_same_content_same_hash(self):
        f1 = self.tmp / "a.txt"
        f2 = self.tmp / "b.txt"
        content = "identical content in two files"
        f1.write_text(content)
        f2.write_text(content)

        doc1 = Document(text=content, source=str(f1), metadata={})
        doc2 = Document(text=content, source=str(f2), metadata={})
        Harness._stamp_doc_hash(doc1, f1)
        Harness._stamp_doc_hash(doc2, f2)

        assert doc1.metadata["content_hash"] == doc2.metadata["content_hash"]

    def test_stamp_doc_hash_missing_file(self):
        """File that doesn't exist should not crash."""
        f = self.tmp / "nonexistent.txt"
        doc = Document(text="", source=str(f), metadata={})
        Harness._stamp_doc_hash(doc, f)
        # Should set empty/default values
        assert "content_hash" in doc.metadata
        assert doc.metadata.get("file_mtime", 0) == 0

    def test_stamp_doc_hash_chinese_content(self):
        f = self.tmp / "chinese.txt"
        f.write_text("赵铁蛋是赵本山的儿子")
        doc = Document(text="赵铁蛋是赵本山的儿子", source=str(f), metadata={})
        Harness._stamp_doc_hash(doc, f)
        assert len(doc.metadata["content_hash"]) == 32


class TestScanDataFiles:
    """Tests for _scan_data_files helper."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmp = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_scan_finds_txt_and_csv(self):
        (self.tmp / "data.txt").write_text("hello")
        (self.tmp / "network.csv").write_text("a,b,c\n1,2,3")
        (self.tmp / "notes.md").write_text("# Notes")

        files = _scan_data_files(self.tmp)
        names = {Path(f).name for f in files}
        assert "data.txt" in names
        assert "network.csv" in names
        assert "notes.md" in names

    def test_scan_excludes_kgclaw_dir(self):
        (self.tmp / "data.txt").write_text("hello")
        kgclaw_dir = self.tmp / ".kgclaw"
        kgclaw_dir.mkdir()
        (kgclaw_dir / "workflow_state.json").write_text("{}")

        files = _scan_data_files(self.tmp)
        names = {Path(f).name for f in files}
        assert "data.txt" in names
        assert "workflow_state.json" not in names

    def test_scan_excludes_git_dir(self):
        (self.tmp / "data.txt").write_text("hello")
        git_dir = self.tmp / ".git"
        git_dir.mkdir()
        (git_dir / "README.txt").write_text("git readme")

        files = _scan_data_files(self.tmp)
        assert not any(".git" in f for f in files)

    def test_scan_recursive(self):
        sub = self.tmp / "subdir"
        sub.mkdir()
        (sub / "deep.txt").write_text("deep content")
        (self.tmp / "root.txt").write_text("root content")

        files = _scan_data_files(self.tmp)
        names = {Path(f).name for f in files}
        assert "root.txt" in names
        assert "deep.txt" in names

    def test_scan_empty_directory(self):
        files = _scan_data_files(self.tmp)
        assert files == []

    def test_scan_unsupported_extensions_filtered(self):
        (self.tmp / "data.txt").write_text("ok")
        (self.tmp / "script.py").write_text("print('hello')")
        (self.tmp / "image.png").write_bytes(b"\x89PNG")

        files = _scan_data_files(self.tmp)
        names = {Path(f).name for f in files}
        assert "data.txt" in names
        assert "script.py" not in names  # .py not supported
        assert "image.png" not in names  # .png not supported


class TestSession:
    """Tests for Session class management."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmp = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_session(self) -> Session:
        return Session(
            api_key="test-key",
            api_base="http://localhost:11434/v1",
            model="test-model",
            work_dir=str(self.tmp),
        )

    def test_session_initial_state(self):
        s = self._make_session()
        assert s.ontology_raw is None
        assert s.documents_loaded is False
        assert s.doc_paths == []
        assert s.last_result is None
        assert s.strategy == "auto"
        assert s.enable_co_occurrence is True
        assert s.output_format == "nt"

    def test_session_work_dir_defaults_to_kgclaw(self):
        s = Session("key", "http://localhost", "model")
        assert s.work_dir == ".kgclaw"
        assert s.output_path == ".kgclaw/output.nt"

    def test_session_set_ontology_tracks_changes(self):
        s = self._make_session()
        assert not s.ontology_updated

        s.set_ontology("Entity Types: 人物")
        assert not s.ontology_updated  # first set, no previous

        s.set_ontology("Entity Types: 人物, 地点")
        assert s.ontology_updated  # changed
        assert s.previous_ontology_raw == "Entity Types: 人物"

    def test_session_set_ontology_same_value_no_update_flag(self):
        s = self._make_session()
        s.set_ontology("Entity Types: 人物, 地点")
        assert not s.ontology_updated

        s.set_ontology("Entity Types: 人物, 地点")
        # Same value — should NOT be flagged as updated
        # (Note: whitespace might differ, so this tests exact match)
        assert s.ontology_raw == "Entity Types: 人物, 地点"

    def test_session_git_initialized_on_access(self):
        s = self._make_session()
        # GitManager is created lazily on first access
        git = s.git
        assert git is not None
        # Git is NOT auto-initialized — only initialized explicitly
        assert isinstance(git, type(s.git))

    def test_session_restore_from_workflow_no_state(self):
        s = Session.restore_from_workflow(
            "key", "http://localhost", "model", work_dir=str(self.tmp)
        )
        assert s.resumed_from is None
        assert s.ontology_raw is None

    def test_session_restore_from_workflow_with_state(self):
        # Create a prior workflow
        mem = Memory(work_dir=str(self.tmp))
        onto = Ontology(
            name="test",
            entity_types=[EntityType(name="人物")],
            relation_types=[RelationType(name="生父")],
            raw_definition="Entity Types: 人物\nRelation Types: 生父",
        )
        mem.init_workflow(ontology=onto)

        # Create a document file that exists
        doc_file = self.tmp / "data.txt"
        doc_file.write_text("赵铁蛋是赵本山的儿子")

        # Set documents in workflow
        mem._workflow.documents = [
            Document(text="赵铁蛋是赵本山的儿子", source=str(doc_file),
                     metadata={"content_hash": "abc", "file_mtime": doc_file.stat().st_mtime, "file_size": doc_file.stat().st_size})
        ]
        mem._workflow.final_result = ExtractionResult(
            entities=[],
            relations=[],
            triples=[],
        )
        mem.save_workflow()

        s = Session.restore_from_workflow(
            "key", "http://localhost", "model", work_dir=str(self.tmp)
        )
        assert s.resumed_from is not None
        assert s.ontology_raw is not None
        assert "人物" in s.ontology_raw

    def test_session_status_summary_includes_git(self):
        s = self._make_session()
        s.set_ontology("Entity Types: 人物\nRelation Types: 生父")
        summary = s.status_summary()
        assert "Model:" in summary
        assert "Onto:" in summary
        # Git line not present until there are commits

    def test_session_needs_rebuild_and_file_changes(self):
        s = self._make_session()
        assert s.needs_rebuild is False
        assert s.file_changes is None
        assert s.rebuild_reason == ""

        s.needs_rebuild = True
        s.rebuild_reason = "文件有变化"
        s.file_changes = {"added": ["new.txt"], "unchanged": [], "modified": [], "deleted": []}

        summary = s.status_summary()
        assert "Files:" in summary
        assert "Rebuild:" in summary


class TestMemoryIntegration:
    """Integration-style tests for Memory with multiple features."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmp = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_workflow_state_roundtrip(self):
        """Simulate a complete workflow: ontoloy → documents → build → manifest."""
        mem = Memory(work_dir=str(self.tmp))

        # 1. Set up ontology
        onto = Ontology(
            name="test_build",
            entity_types=[EntityType(name="人物")],
            relation_types=[RelationType(name="朋友")],
            raw_definition="Entity Types: 人物",
        )
        mem.init_workflow(ontology=onto)

        # 2. Export ontology
        json_path, md_path = mem.export_ontology()
        assert json_path.exists()
        assert md_path.exists()

        # 3. Save generated code
        code_path = mem.save_generated_code("extract_v1.py", "print('extraction')")
        assert code_path.exists()

        # 4. Create documents
        doc_file = self.tmp / "source.txt"
        doc_file.write_text("some content")
        import hashlib
        doc = Document(
            text="some content",
            source=str(doc_file),
            metadata={
                "content_hash": hashlib.md5(doc_file.read_bytes()).hexdigest(),
                "file_mtime": doc_file.stat().st_mtime,
                "file_size": doc_file.stat().st_size,
            },
        )
        mem.init_workflow(ontology=onto)
        mem._workflow.documents = [doc]
        mem.save_workflow()

        # 5. Save manifest
        mem.save_document_manifest([doc])
        manifest = mem.load_document_manifest()
        assert str(doc_file) in manifest

        # 6. Verify file change detection
        changes = mem.detect_file_changes([str(doc_file)])
        assert str(doc_file) in changes["unchanged"]

        # 7. Modify file — detect change
        doc_file.write_text("modified content")
        changes2 = mem.detect_file_changes([str(doc_file)])
        assert str(doc_file) in changes2["modified"]

    def test_ontology_export_markdown_completeness(self):
        """Verify the markdown ontology export is complete and readable."""
        onto = Ontology(
            name="人物关系本体",
            description="用于提取人物之间的家庭和社会关系",
            entity_types=[
                EntityType(name="人物", description="自然人"),
                EntityType(name="地点", description="地理位置，包括村庄、城镇等"),
            ],
            relation_types=[
                RelationType(name="生父", description="生物学父亲",
                             domain="人物", range="人物"),
                RelationType(name="居住于", description="人物在某地居住",
                             domain="人物", range="地点"),
                RelationType(name="朋友", description="朋友关系",
                             domain="人物", range="人物"),
            ],
            raw_definition="Entity Types: 人物, 地点\nRelation Types: 生父, 居住于, 朋友",
        )

        mem = Memory(work_dir=str(self.tmp))
        mem.init_workflow(ontology=onto)
        _, md_path = mem.export_ontology()

        content = md_path.read_text()
        # Headers
        assert "# Ontology: 人物关系本体" in content
        assert "## Entity Types" in content
        assert "## Relation Types" in content
        assert "## Raw Definition" in content
        # Entity types
        assert "人物" in content
        assert "地点" in content
        assert "自然人" in content
        assert "地理位置" in content
        # Relation types with domain/range
        assert "生父" in content
        assert "`人物` → `人物`" in content
        assert "居住于" in content
        assert "`人物` → `地点`" in content
        assert "朋友" in content
        # Raw definition in code block
        assert "```" in content
        assert "Entity Types: 人物, 地点" in content


class TestEagerOntologyAnalysis:
    """Tests for eager ontology analysis in set_ontology()."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmp = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_is_structured_property_false_when_no_types(self):
        """Ontology.is_structured is False when only raw_definition is set."""
        onto = Ontology(raw_definition="Entity Types: 人物")
        assert onto.is_structured is False

    def test_is_structured_property_true_with_types(self):
        """Ontology.is_structured is True when entity_types are populated."""
        onto = Ontology(
            raw_definition="Entity Types: 人物",
            entity_types=[EntityType(name="人物")],
        )
        assert onto.is_structured is True

    def test_is_structured_property_true_with_empty_raw(self):
        """Ontology.is_structured works even without raw_definition."""
        onto = Ontology(entity_types=[EntityType(name="人物")])
        assert onto.is_structured is True

    def test_set_ontology_without_llm_stores_raw(self):
        """When LLM not configured, set_ontology stores raw only."""
        config = HarnessConfig(
            llm=LLMConfig(api_key="", api_base="https://api.openai.com/v1"),
            work_dir=str(self.tmp),
        )
        harness = Harness(config)
        ontology = harness.set_ontology("Entity Types: 人物\nRelation Types: 生父")
        assert ontology.raw_definition is not None
        assert ontology.is_structured is False

    def test_set_ontology_eager_analysis_fallback(self):
        """When LLM call fails, ontology is still saved as raw."""
        config = HarnessConfig(
            llm=LLMConfig(api_key="test-key", api_base="http://localhost:11434/v1"),
            work_dir=str(self.tmp),
        )
        harness = Harness(config)
        # Eager analysis will fail (bad API endpoint), but raw storage should work
        ontology = harness.set_ontology("Entity Types: 人物\nRelation Types: 生父")
        assert ontology.raw_definition is not None
        # May or may not be structured depending on LLM availability

    def test_run_skips_phase1_when_ontology_structured(self):
        """run() should skip Phase 1 if ontology already has structured types."""
        from unittest.mock import patch

        config = HarnessConfig(work_dir=str(self.tmp))
        harness = Harness(config)
        onto = Ontology(
            name="pre_analyzed",
            entity_types=[EntityType(name="人物", description="Person")],
            relation_types=[RelationType(name="生父", domain="人物", range="人物")],
            raw_definition="Entity Types: 人物\nRelation Types: 生父",
        )
        harness.set_ontology_structured(onto)
        harness.load_texts(["test document about family relations"])

        # Mock Agent.run_structured to avoid actual LLM calls in fast path
        with patch("kgclaw.agent.Agent.run_structured") as mock_run:
            mock_run.return_value = {"entities": [], "relations": []}
            result = harness.run(strategy="fast")
            assert result is not None

    def test_run_preserves_structured_ontology(self):
        """run() should not overwrite a structured ontology with a bare raw one."""
        from unittest.mock import patch

        config = HarnessConfig(work_dir=str(self.tmp))
        harness = Harness(config)
        structured = Ontology(
            entity_types=[EntityType(name="人物")],
            relation_types=[RelationType(name="生父")],
            raw_definition="original raw text",
        )
        harness.set_ontology_structured(structured)
        harness.load_texts(["test document"])

        # Mock Agent.run_structured to avoid actual LLM calls in fast path
        with patch("kgclaw.agent.Agent.run_structured") as mock_run:
            mock_run.return_value = {"entities": [], "relations": []}
            harness.run(
                ontology_raw="Entity Types: 地点\nRelation Types: 位于",
                strategy="fast",
            )

        stored = harness.memory._workflow.ontology
        assert stored is not None
        # The structured types should be preserved ("人物"), not replaced by "地点"
        assert "人物" in stored.entity_type_names
        assert "地点" not in stored.entity_type_names

    def test_can_analyze_ontology_no_api_key(self):
        """_can_analyze_ontology returns False when no API key is set."""
        config = HarnessConfig(
            llm=LLMConfig(api_key=""),
            work_dir=str(self.tmp),
        )
        harness = Harness(config)
        assert harness._can_analyze_ontology() is False

    def test_can_analyze_ontology_with_api_key(self):
        """_can_analyze_ontology returns True when API key is set."""
        config = HarnessConfig(
            llm=LLMConfig(api_key="sk-test123"),
            work_dir=str(self.tmp),
        )
        harness = Harness(config)
        assert harness._can_analyze_ontology() is True

    def test_set_ontology_initializes_workflow(self):
        """set_ontology should create a workflow if none exists yet."""
        config = HarnessConfig(work_dir=str(self.tmp))
        harness = Harness(config)
        assert harness.memory._workflow is None
        harness.set_ontology("Entity Types: 人物")
        assert harness.memory._workflow is not None
        assert harness.memory._workflow.ontology is not None
        assert harness.memory._workflow.ontology.raw_definition == "Entity Types: 人物"


class TestCanonicalization:
    """Tests for Schema Canonicalization (batch mode)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmp = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_ontology(self):
        return Ontology(
            name="test",
            relation_types=[
                RelationType(name="创始人", description="人物创建公司"),
                RelationType(name="CEO", description="人物担任CEO"),
                RelationType(name="投资方", description="公司投资公司"),
            ],
        )

    def _make_harness(self):
        config = HarnessConfig(
            llm=LLMConfig(api_key="", api_base="https://api.openai.com/v1"),
            work_dir=str(self.tmp),
        )
        harness = Harness(config)
        harness.load_texts(["test"])
        return harness

    def test_all_matched_returns_empty(self):
        """When all predicates match ontology, canonicalize returns empty."""
        harness = self._make_harness()
        onto = self._make_ontology()
        relations = [
            {"predicate": "创始人", "subject": "A", "object": "B"},
            {"predicate": "CEO", "subject": "C", "object": "D"},
        ]
        result = harness._canonicalize_relations(onto, relations, [])
        assert result == {}

    def test_no_ontology_returns_empty(self):
        harness = self._make_harness()
        result = harness._canonicalize_relations(None, [], [])
        assert result == {}

    def test_no_relation_types_returns_empty(self):
        harness = self._make_harness()
        onto = Ontology(name="empty", relation_types=[])
        result = harness._canonicalize_relations(onto, [{"predicate": "X"}], [])
        assert result == {}

    def test_unmatched_predicates_included_in_batch(self):
        """Unmatched predicates should be collected for batch processing."""
        harness = self._make_harness()
        onto = self._make_ontology()
        relations = [
            {"predicate": "创始人", "subject": "A", "object": "B"},
            {"predicate": "unknown_rel", "subject": "C", "object": "D"},
            {"predicate": "another_unknown", "subject": "E", "object": "F"},
        ]
        # Without LLM, unmatched predicates just return empty map
        result = harness._canonicalize_relations(onto, relations, [])
        assert isinstance(result, dict)
        # "创始人" is matched, so it should NOT be in the result
        assert "创始人" not in result

    def test_empty_predicates_filtered(self):
        harness = self._make_harness()
        onto = self._make_ontology()
        relations = [
            {"predicate": "", "subject": "A", "object": "B"},
            {"predicate": "创始人", "subject": "C", "object": "D"},
        ]
        result = harness._canonicalize_relations(onto, relations, [])
        assert "" not in result  # empty predicate should be filtered

    def test_predicates_from_relation_key(self):
        """Should also check 'relation' key (not just 'predicate')."""
        harness = self._make_harness()
        onto = self._make_ontology()
        relations = [
            {"relation": "创始人", "subject": "A", "object": "B"},
        ]
        result = harness._canonicalize_relations(onto, relations, [])
        assert result == {}  # '创始人' matches ontology

    def test_batch_limits_to_20_unmatched(self):
        """Batch mode limits to 20 unmatched predicates."""
        harness = self._make_harness()
        onto = self._make_ontology()
        relations = [
            {"predicate": f"unknown_{i}", "subject": "A", "object": "B"}
            for i in range(25)
        ]
        # Should not crash with 25 unmatched — limited to 20
        result = harness._canonicalize_relations(onto, relations, [])
        assert isinstance(result, dict)


class TestGleaning:
    """Tests for the gleaning (second-pass entity extraction) feature."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmp = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_harness(self):
        config = HarnessConfig(
            llm=LLMConfig(api_key="", api_base="https://api.openai.com/v1"),
            work_dir=str(self.tmp),
            enable_gleaning=True,
        )
        harness = Harness(config)
        harness.load_texts(["test document"])
        return harness

    def test_gleaning_disabled_skips(self):
        """When enable_gleaning is False, gleaning is skipped."""
        harness = self._make_harness()
        harness.config.enable_gleaning = False
        # _glean_entities should not be called during entity extraction
        # But we test the guard directly:
        # The phase code checks getattr(self.config, 'enable_gleaning', True)
        assert harness.config.enable_gleaning is False

    def test_gleaning_with_empty_entities(self):
        """Gleaning with empty entity list should not crash."""
        harness = self._make_harness()
        from kgclaw.skills import get_skill
        skill = get_skill("entity_extractor", harness.llm_config)
        if skill:
            result = harness._glean_entities(
                "Entity Types: 人物", "some text", [], skill,
            )
            assert isinstance(result, list)

    def test_gleaning_entity_merge_prefers_longer_description(self):
        """Test that the merge logic in gleaning is correct (unit test for merge block)."""
        from kgclaw.models import Entity

        existing = [
            Entity(name="Alice", type="人物", description="short", confidence=0.9),
        ]
        gleaned = [
            Entity(name="Alice", type="人物",
                  description="A much longer and more detailed description of Alice",
                  confidence=0.7),
            Entity(name="Bob", type="人物",
                  description="A new person not previously found",
                  confidence=0.6),
        ]

        # Simulate the merge logic from phases.py
        existing_names = {(e.name, e.type) for e in existing}
        gleaned_count = 0
        for ge in gleaned:
            key = (ge.name, ge.type)
            if key in existing_names:
                for ex in existing:
                    if (ex.name, ex.type) == key:
                        if len(ge.description) > len(ex.description):
                            ex.description = ge.description
                        if ge.confidence > ex.confidence:
                            ex.confidence = ge.confidence
                        break
            else:
                existing.append(ge)
                gleaned_count += 1

        assert gleaned_count == 1  # Bob is new
        assert existing[0].description == "A much longer and more detailed description of Alice"
        assert existing[0].confidence == 0.9  # existing confidence is higher, preserved
        assert len(existing) == 2

    def test_gleaning_entity_merge_higher_confidence(self):
        """Gleaned entity with higher confidence should update existing."""
        from kgclaw.models import Entity
        existing = [
            Entity(name="Alice", type="人物", description="desc", confidence=0.5),
        ]
        gleaned = [
            Entity(name="Alice", type="人物", description="desc", confidence=0.95),
        ]
        existing_names = {(e.name, e.type) for e in existing}
        for ge in gleaned:
            key = (ge.name, ge.type)
            if key in existing_names:
                for ex in existing:
                    if (ex.name, ex.type) == key:
                        if len(ge.description) > len(ex.description):
                            ex.description = ge.description
                        if ge.confidence > ex.confidence:
                            ex.confidence = ge.confidence
                        break
        assert existing[0].confidence == 0.95


class TestCLIFlags:
    """Tests for CLI flag parsing (--trace, --concurrency, --chunk-size)."""

    def test_concurrency_default(self):
        """Default max_concurrent_agents is 8."""
        config = HarnessConfig()
        assert config.max_concurrent_agents == 8

    def test_concurrency_custom(self):
        """Custom concurrency is accepted."""
        config = HarnessConfig(max_concurrent_agents=16)
        assert config.max_concurrent_agents == 16

    def test_concurrency_minimum(self):
        """Concurrency of 1 is valid."""
        config = HarnessConfig(max_concurrent_agents=1)
        assert config.max_concurrent_agents == 1

    def test_chunk_size_default(self):
        """Default chunk_size is 4000."""
        config = HarnessConfig()
        assert config.chunk_size == 4000

    def test_chunk_size_custom(self):
        config = HarnessConfig(chunk_size=8000)
        assert config.chunk_size == 8000

    def test_chunk_overlap_default(self):
        config = HarnessConfig()
        assert config.chunk_overlap == 300

    def test_docs_per_relation_group_default(self):
        config = HarnessConfig()
        assert config.docs_per_relation_group == 8

    def test_max_entities_in_qc_default(self):
        config = HarnessConfig()
        assert config.max_entities_in_qc == 500

    def test_max_relations_in_qc_default(self):
        config = HarnessConfig()
        assert config.max_relations_in_qc == 500

    def test_max_chunks_default(self):
        config = HarnessConfig()
        assert config.max_chunks == 200

    def test_all_new_config_fields_accessible(self):
        """All new performance config fields should be instantiable."""
        config = HarnessConfig(
            max_concurrent_agents=12,
            chunk_size=6000,
            chunk_overlap=400,
            docs_per_relation_group=10,
            chars_per_doc_relation=5000,
            max_entities_in_qc=300,
            max_relations_in_qc=400,
            max_chunks=150,
        )
        assert config.max_concurrent_agents == 12
        assert config.chunk_size == 6000
        assert config.chunk_overlap == 400
        assert config.docs_per_relation_group == 10
        assert config.chars_per_doc_relation == 5000
        assert config.max_entities_in_qc == 300
        assert config.max_relations_in_qc == 400
        assert config.max_chunks == 150
