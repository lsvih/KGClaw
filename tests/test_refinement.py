"""
Tests for kgclaw.refinement — RefinementEngine and RefinementPlan.

Tests are designed to run without any LLM calls (using empty API keys
or mocked Agent responses).
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kgclaw.memory import Memory
from kgclaw.models import (
    Entity,
    EntityType,
    ExtractionResult,
    LLMConfig,
    Ontology,
    OntologyChange,
    RefinementPlan,
    Relation,
    RelationType,
    Triple,
)
from kgclaw.refinement import (
    RefinementEngine,
    create_refinement_engine,
    SYSTEM_PROMPT_REFINEMENT,
    TASK_REFINE,
)


# ─── RefinementPlan model tests ──────────────────────────────────────────────

class TestRefinementPlan:
    def test_empty_plan_has_no_changes(self):
        plan = RefinementPlan()
        assert not plan.has_changes

    def test_ontology_changes_flag(self):
        plan = RefinementPlan(ontology_changes=[
            OntologyChange(action="add", target="entity_type", name="作者")
        ])
        assert plan.has_changes

    def test_updated_raw_flags(self):
        plan = RefinementPlan(updated_ontology_raw="Entity Types: 人物, 地点")
        assert plan.has_changes

    def test_strategy_change_flags(self):
        plan = RefinementPlan(suggested_strategy="fast")
        assert plan.has_changes

    def test_extraction_tips_flag(self):
        plan = RefinementPlan(extraction_tips="Focus on cross-sentence relations.")
        assert plan.has_changes

    def test_prompt_additions_flag(self):
        plan = RefinementPlan(prompt_additions=["Pay attention to implicit relations."])
        assert plan.has_changes

    def test_gleaning_toggle_flag(self):
        plan = RefinementPlan(enable_gleaning=True)
        assert plan.has_changes

    def test_co_occurrence_toggle_flag(self):
        plan = RefinementPlan(enable_co_occurrence=False)
        assert plan.has_changes

    def test_chunk_size_change_flag(self):
        plan = RefinementPlan(suggested_chunk_size=8000)
        assert plan.has_changes

    def test_combined_changes(self):
        plan = RefinementPlan(
            ontology_changes=[
                OntologyChange(action="add", target="entity_type", name="作者"),
            ],
            suggested_strategy="standard",
            enable_gleaning=True,
        )
        assert plan.has_changes


class TestOntologyChange:
    def test_add_entity_type(self):
        oc = OntologyChange(
            action="add", target="entity_type", name="作者",
            description="文章或书籍的作者", reason="用户反馈需要区分作者和编辑者"
        )
        assert oc.action == "add"
        assert oc.target == "entity_type"
        assert oc.name == "作者"
        assert oc.reason

    def test_remove_relation_type(self):
        oc = OntologyChange(
            action="remove", target="relation_type", name="过时关系",
        )
        assert oc.action == "remove"

    def test_modify_entity_with_parent(self):
        oc = OntologyChange(
            action="modify", target="entity_type", name="人物",
            description="更新后的描述", parent="Agent"
        )
        assert oc.parent == "Agent"


# ─── RefinementEngine tests ──────────────────────────────────────────────────

@pytest.fixture
def sample_result():
    entities = [
        Entity(name="Alice", type="人物", description="A researcher", confidence=0.95),
        Entity(name="Bob", type="人物", description="An engineer", confidence=0.90),
        Entity(name="Acme Corp", type="公司", confidence=0.85),
        Entity(name="Unknown Thing", type="Entity", confidence=0.45),
    ]
    relations = [
        Relation(subject="Alice", predicate="创始人", object="Acme Corp", confidence=0.90),
        Relation(subject="Bob", predicate="CEO", object="Acme Corp", confidence=0.80),
        Relation(subject="Alice", predicate="同事", object="Bob", confidence=0.50),
    ]
    triples = [
        Triple(subject=entities[0], predicate="创始人", object=entities[2], confidence=0.90),
    ]
    return ExtractionResult(entities=entities, relations=relations, triples=triples)


@pytest.fixture
def sample_ontology():
    return Ontology(
        name="企业知识图谱",
        entity_types=[
            EntityType(name="人物", description="自然人"),
            EntityType(name="公司", description="企业法人"),
        ],
        relation_types=[
            RelationType(name="创始人", description="人物创建公司", domain="人物", range="公司"),
            RelationType(name="CEO", description="人物担任CEO", domain="人物", range="公司"),
        ],
        raw_definition="Entity Types: 人物, 公司\nRelation Types: 创始人, CEO",
    )


@pytest.fixture
def sample_docs():
    class MockDoc:
        def __init__(self, text, ext="txt"):
            self.text = text
            self.metadata = {"ext": ext}
    return [MockDoc("Alice founded Acme Corp.", "txt")]


@pytest.fixture
def engine():
    mem = Memory(work_dir=tempfile.mkdtemp())
    llm_cfg = LLMConfig(api_key="", model="gpt-4o")  # no API key — no real calls
    return RefinementEngine(llm_cfg, mem)


class TestRefinementEngineAnalyze:
    """Tests for RefinementEngine.analyze()."""

    def test_empty_feedback_returns_empty(self, engine, sample_result, sample_ontology, sample_docs):
        plan = engine.analyze(sample_result, sample_ontology, sample_docs, "")
        assert isinstance(plan, RefinementPlan)
        assert not plan.has_changes

    def test_whitespace_feedback_returns_empty(self, engine, sample_result, sample_ontology, sample_docs):
        plan = engine.analyze(sample_result, sample_ontology, sample_docs, "   \n  ")
        assert not plan.has_changes

    def test_with_feedback_returns_plan(self, engine, sample_result, sample_ontology, sample_docs):
        """With empty API key, LLM call fails → returns empty plan (no crash)."""
        plan = engine.analyze(sample_result, sample_ontology, sample_docs,
                             "需要增加作者实体类型")
        assert isinstance(plan, RefinementPlan)
        # Should not crash even though API key is empty

    def test_prompt_includes_context(self, engine, sample_result, sample_ontology, sample_docs):
        """Verify that the prompt is built with correct context (unit test)."""
        # Build the prompt manually to verify it includes expected data
        prompt = TASK_REFINE.format(
            user_feedback="test feedback",
            strategy="standard",
            entity_count=4,
            relation_count=3,
            triple_count=1,
            quality_score="N/A",
            ontology_guide=sample_ontology.to_extraction_guide(),
            entity_type_distribution="  - 人物: 2 个",
            relation_type_distribution="  - 创始人: 1 个",
            entity_samples="[]",
            relation_samples="[]",
            doc_count=1,
            total_chars="23",
            doc_formats="txt",
        )
        assert "test feedback" in prompt
        assert "企业知识图谱" in prompt
        assert "4" in prompt  # entity count
        assert "standard" in prompt


class TestRefinementEngineApply:
    """Tests for RefinementEngine.apply()."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmp = Path(tempfile.mkdtemp())
        yield
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_session(self):
        mem = Memory(work_dir=str(self.tmp))
        mem.init_workflow(ontology=Ontology(
            name="test",
            entity_types=[EntityType(name="人物", description="Person")],
            relation_types=[RelationType(name="居住于", domain="人物", range="地点")],
            raw_definition="Entity Types: 人物\nRelation Types: 居住于",
        ))

        # Build a mock harness that delegates memory and config access to real objects
        harness = MagicMock()
        harness.memory = mem
        harness.memory._workflow = mem._workflow
        harness.config.enable_gleaning = False
        harness.config.chunk_size = 4000

        # Build a mock session where set_ontology() actually updates the workflow
        session = MagicMock()
        session.ontology_raw = "Entity Types: 人物\nRelation Types: 居住于"
        session.strategy = "standard"
        session.enable_co_occurrence = True
        session.refinement_tips = {}
        session.harness = harness

        # Make set_ontology track calls and update the real workflow
        _call_log = []
        def _real_set_ontology(raw):
            _call_log.append(raw)
            session.ontology_raw = raw
            from kgclaw.models import Ontology as Ont
            mem._workflow.ontology = Ont(raw_definition=raw)
            mem.save_workflow()
        session.set_ontology = MagicMock(wraps=_real_set_ontology)

        return session

    def _make_engine(self, session):
        return RefinementEngine(
            LLMConfig(api_key="test-key"),
            session.harness.memory,
        )

    def test_apply_add_entity_type(self):
        session = self._make_session()
        plan = RefinementPlan(ontology_changes=[
            OntologyChange(action="add", target="entity_type", name="组织",
                          description="各类组织机构"),
        ])
        engine = self._make_engine(session)
        changes = engine.apply(plan, session)
        assert changes["ontology_updated"]
        # set_ontology should be called with raw text containing the new type
        session.set_ontology.assert_called_once()
        raw = session.set_ontology.call_args[0][0]
        assert "组织" in raw

    def test_apply_remove_entity_type(self):
        session = self._make_session()
        plan = RefinementPlan(ontology_changes=[
            OntologyChange(action="remove", target="entity_type", name="人物"),
        ])
        engine = self._make_engine(session)
        engine.apply(plan, session)
        session.set_ontology.assert_called_once()
        raw = session.set_ontology.call_args[0][0]
        # "人物" should have been removed from the raw definition
        assert "- **人物**" not in raw

    def test_apply_add_relation_type(self):
        session = self._make_session()
        plan = RefinementPlan(ontology_changes=[
            OntologyChange(action="add", target="relation_type", name="工作于",
                          description="人物在某组织工作", domain="人物", range="组织"),
        ])
        engine = self._make_engine(session)
        engine.apply(plan, session)
        session.set_ontology.assert_called_once()
        raw = session.set_ontology.call_args[0][0]
        assert "工作于" in raw

    def test_apply_updated_raw_ontology(self):
        session = self._make_session()
        new_raw = "Entity Types: 人物, 组织\nRelation Types: 居住于, 工作于"
        plan = RefinementPlan(updated_ontology_raw=new_raw)
        engine = self._make_engine(session)
        changes = engine.apply(plan, session)
        session.set_ontology.assert_called_once_with(new_raw)
        assert changes["ontology_updated"]

    def test_apply_strategy_change(self):
        session = self._make_session()
        plan = RefinementPlan(suggested_strategy="fast")
        engine = self._make_engine(session)
        changes = engine.apply(plan, session)
        assert changes["strategy_changed"]
        assert session.strategy == "fast"

    def test_apply_invalid_strategy_ignored(self):
        session = self._make_session()
        plan = RefinementPlan(suggested_strategy="invalid_strategy")
        engine = self._make_engine(session)
        changes = engine.apply(plan, session)
        assert not changes["strategy_changed"]

    def test_apply_gleaning_toggle(self):
        session = self._make_session()
        plan = RefinementPlan(enable_gleaning=True)
        engine = self._make_engine(session)
        engine.apply(plan, session)
        assert session.harness.config.enable_gleaning is True

    def test_apply_chunk_size(self):
        session = self._make_session()
        plan = RefinementPlan(suggested_chunk_size=8000)
        engine = self._make_engine(session)
        engine.apply(plan, session)
        assert session.harness.config.chunk_size == 8000

    def test_apply_stores_tips(self):
        session = self._make_session()
        plan = RefinementPlan(
            extraction_tips="Focus on implicit relations.",
            prompt_additions=["Check temporal relations."],
        )
        engine = self._make_engine(session)
        changes = engine.apply(plan, session)
        assert changes["tips_added"]
        assert session.refinement_tips["tips"] == "Focus on implicit relations."

    def test_apply_empty_plan_no_changes(self):
        session = self._make_session()
        plan = RefinementPlan()
        engine = self._make_engine(session)
        changes = engine.apply(plan, session)
        assert not changes["ontology_updated"]
        assert not changes["strategy_changed"]


class TestCreateRefinementEngine:
    def test_factory_creates_engine(self):
        llm_cfg = LLMConfig(api_key="test-key")
        mem = Memory(work_dir=tempfile.mkdtemp())
        engine = create_refinement_engine(llm_cfg, mem)
        assert isinstance(engine, RefinementEngine)


class TestRefinementPrompts:
    def test_system_prompt_content(self):
        assert "知识图谱构建优化专家" in SYSTEM_PROMPT_REFINEMENT
        assert "本体优化" in SYSTEM_PROMPT_REFINEMENT

    def test_task_prompt_placeholders(self):
        for ph in ("{user_feedback}", "{entity_count}", "{strategy}", "{ontology_guide}"):
            assert ph in TASK_REFINE
