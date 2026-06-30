"""
Integration tests for KGClaw.

Tests the structural flow of the harness without requiring LLM API calls.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kgclaw import (
    Document,
    Entity,
    EntityType,
    ExtractionResult,
    Harness,
    HarnessConfig,
    LLMConfig,
    Ontology,
    Relation,
    RelationType,
    Triple,
)
from kgclaw.memory import Memory
from kgclaw.models import Message, Role, PhaseResult, PhaseStatus, WorkflowState
from kgclaw.tools import discover_tools, execute_tool, get_tool, Tool
from kgclaw.skills import (
    SkillRegistry,
    get_skill,
    get_all_skill_names,
    get_default_pipeline_skills,
)
from kgclaw.agent import Agent, AgentConfig
from kgclaw.prompts.system_prompts import (
    build_entity_extraction_prompt,
    build_relation_extraction_prompt,
    build_ontology_analysis_prompt,
    build_quality_check_prompt,
)


def test_models():
    """Test core data models."""
    # Ontology construction
    etype = EntityType(name="人物", description="Person")
    rtype = RelationType(name="生父", domain="人物", range="人物", inverse="儿子")
    ontology = Ontology(
        name="test_onto",
        entity_types=[etype],
        relation_types=[rtype],
        raw_definition="人物: 人物类型",
    )

    assert ontology.entity_type_names == ["人物"]
    assert ontology.relation_type_names == ["生父"]
    guide = ontology.to_extraction_guide()
    assert "人物" in guide
    assert "生父" in guide

    # Entity creation
    entity = Entity(name="赵铁蛋", type="人物", confidence=0.9)
    assert entity.name == "赵铁蛋"

    # Relation creation
    relation = Relation(
        subject="赵铁蛋", predicate="生父", object="赵本山", confidence=0.85
    )
    assert relation.predicate == "生父"

    # Triple creation and NT format
    triple = Triple(
        subject=entity,
        predicate="生父",
        object=Entity(name="赵本山", type="人物"),
        confidence=0.85,
    )
    nt_line = triple.to_nt_line()
    # URL-encoded Chinese characters are the expected behavior for valid URIs
    assert "%E4%BA%BA%E7%89%A9" in nt_line  # 人物 URL-encoded
    assert "%E8%B5%B5%E9%93%81%E8%9B%8B" in nt_line  # 赵铁蛋 URL-encoded
    assert "%E7%94%9F%E7%88%B6" in nt_line  # 生父 URL-encoded

    # Document creation
    doc = Document(text="赵铁蛋是赵本山的儿子", source="test.txt")
    assert doc.text == "赵铁蛋是赵本山的儿子"

    # ExtractionResult
    result = ExtractionResult(
        entities=[entity],
        relations=[relation],
        triples=[triple],
    )
    assert len(result.entities) == 1

    # WorkflowState
    state = WorkflowState(ontology=ontology)
    state.documents = [doc]
    assert len(state.documents) == 1

    # PhaseResult
    phase = PhaseResult(phase_name="test_phase", status=PhaseStatus.COMPLETED)
    assert phase.status == PhaseStatus.COMPLETED

    # Message
    msg = Message(role=Role.USER, content="hello")
    assert msg.role == Role.USER

    print("✅ Models tests passed")


def test_tools():
    """Test tool registration and execution."""
    # Discover all tools
    tools = discover_tools()
    tool_names = {t.name for t in tools}
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "parse_json" in tool_names
    assert "validate_against_ontology" in tool_names
    assert "deduplicate_entities" in tool_names

    # Execute tools
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write tool
        result = execute_tool("write_file", {
            "path": f"{tmpdir}/test.txt",
            "content": "Hello World",
        })
        assert result.success

        # Read tool
        result = execute_tool("read_file", {"path": f"{tmpdir}/test.txt"})
        assert result.success
        assert "Hello World" in result.data

        # Parse JSON tool
        result = execute_tool("parse_json", {"json_string": '{"a": 1}'})
        assert result.success
        assert result.data["a"] == 1

        # Validate against ontology — invalid entity type
        ontology_json = json.dumps({
            "entity_types": [{"name": "人物"}],
            "relation_types": [{"name": "生父"}],
        })
        entities_json = json.dumps([{"name": "test", "type": "InvalidType"}])
        relations_json = json.dumps([])
        result = execute_tool("validate_against_ontology", {
            "entities_json": entities_json,
            "relations_json": relations_json,
            "ontology_json": ontology_json,
        })
        assert result.success
        assert not result.data["valid"]
        assert len(result.data["issues"]) == 1

        # Validate against ontology — valid
        entities_json = json.dumps([{"name": "test", "type": "人物"}])
        result = execute_tool("validate_against_ontology", {
            "entities_json": entities_json,
            "relations_json": relations_json,
            "ontology_json": ontology_json,
        })
        assert result.success
        assert result.data["valid"]

        # Deduplicate
        entities_json = json.dumps([
            {"name": "A", "type": "人物", "confidence": 0.9},
            {"name": "A", "type": "人物", "confidence": 0.5},
            {"name": "B", "type": "人物"},
        ])
        result = execute_tool("deduplicate_entities", {"entities_json": entities_json})
        assert result.success
        assert len(result.data) == 2

        # Search in text
        result = execute_tool("search_in_text", {
            "text": "Hello World. World is big.",
            "query": "World",
        })
        assert result.success
        assert len(result.data) == 2

    print("✅ Tools tests passed")


def test_skills():
    """Test skill registry and skill instances."""
    # List all skills
    names = get_all_skill_names()
    assert "ontology_analyzer" in names
    assert "entity_extractor" in names
    assert "relation_extractor" in names
    assert "quality_checker" in names
    assert "triple_constructor" in names

    # Default pipeline
    pipeline = get_default_pipeline_skills()
    assert len(pipeline) == 5

    # Get skill instances
    for name in names:
        skill = get_skill(name)
        assert skill is not None
        assert skill.get_system_prompt() != ""
        # 提取类 skill 不需要工具（文本在 prompt 中），但必须有输出 schema

        # Check output schema
        schema = skill.get_output_schema()
        assert "type" in schema

    print("✅ Skills tests passed")


def test_memory():
    """Test memory system."""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = Memory(work_dir=tmpdir)

        # Init workflow
        ontology = Ontology(
            name="test",
            entity_types=[EntityType(name="人物")],
            relation_types=[RelationType(name="生父")],
        )
        state = memory.init_workflow(ontology=ontology)
        assert state.ontology.name == "test"

        # Add messages
        memory.add_message("agent1", Message(role=Role.SYSTEM, content="You are a bot"))
        memory.add_message("agent1", Message(role=Role.USER, content="Hello"))
        msgs = memory.get_messages("agent1")
        assert len(msgs) == 2

        # Add phase result
        phase = PhaseResult(phase_name="extraction", status=PhaseStatus.COMPLETED)
        memory.add_phase_result(phase)
        state = memory.load_workflow()
        if state:
            assert len(state.phases) == 1

        # Context store
        memory.set_context("key1", "value1")
        assert memory.get_context("key1") == "value1"

        # Compact messages
        for i in range(60):
            memory.add_message("agent2", Message(role=Role.USER, content=f"msg {i}"))
        compacted = memory.compact_messages("agent2", max_messages=20)
        assert len(compacted) <= 20

        # Progress summary
        summary = memory.get_progress_summary()
        assert "Workflow" in summary

    print("✅ Memory tests passed")


def test_harness_structure():
    """Test harness structural flow (without API calls)."""
    config = HarnessConfig(
        llm=LLMConfig(
            model="gpt-4o",
            api_key="dummy-key",
        ),
        work_dir=".kgclaw_test_harness",
        chunk_size=2000,
    )

    harness = Harness(config)

    # Test document loading from file
    test_file = Path(__file__).parent.parent / "examples" / "人物图谱" / "人物关系图谱原始数据.txt"
    if test_file.exists():
        docs = harness.load_documents([str(test_file)])
        assert len(docs) == 1
        assert "赵铁蛋" in docs[0].text

    # Test text loading
    docs = harness.load_texts(["测试文本一", "测试文本二"])
    assert len(docs) == 2
    assert docs[0].text == "测试文本一"

    # Test ontology setting
    ontology = harness.set_ontology("Entity Types: 人物\nRelation Types: 生父")
    assert ontology.raw_definition is not None

    # Test chunking — need enough text to exceed MAX_CHUNK_SIZE (16000 chars)
    long_text = "这是一段较长的测试文本。它包含许多句子。\n" * 1000
    chunks = harness._chunk_text(long_text)
    assert len(chunks) > 1

    # Test agent creation through skill
    from kgclaw.skills import get_skill
    skill = get_skill("entity_extractor")
    agent = harness._create_skill_agent("test_agent", skill)
    assert agent.agent_id == "test_agent"

    # Test final result building
    phase = PhaseResult(
        phase_name="test",
        status=PhaseStatus.COMPLETED,
        output=ExtractionResult(
            entities=[Entity(name="A", type="人物")],
            triples=[
                Triple(
                    subject=Entity(name="A", type="人物"),
                    predicate="test",
                    object=Entity(name="B", type="人物"),
                )
            ],
        ),
    )
    harness.memory.add_phase_result(phase)
    result = harness._build_final_result()
    assert len(result.entities) == 1
    assert len(result.triples) == 1

    print("✅ Harness structural tests passed")


def test_prompts():
    """Test prompt generation."""
    guide = "## Entity Types\n- 人物\n## Relation Types\n- 生父"

    prompt = build_ontology_analysis_prompt("Entity Types: 人物")
    assert "人物" in prompt

    prompt = build_entity_extraction_prompt(
        ontology_guide=guide,
        texts="赵铁蛋是赵本山的儿子",
    )
    assert "人物" in prompt
    assert "赵铁蛋" in prompt

    prompt = build_relation_extraction_prompt(
        ontology_guide=guide,
        entities_summary='[{"name": "赵铁蛋", "type": "人物"}]',
        texts="赵铁蛋是赵本山的儿子",
    )
    assert "人物" in prompt

    prompt = build_quality_check_prompt(
        ontology_guide=guide,
        extraction_summary="{}",
        original_texts="测试",
    )
    assert "quality" in prompt.lower() or "质量" in prompt

    print("✅ Prompts tests passed")


def test_config():
    """Test configuration models."""
    config = HarnessConfig(
        llm=LLMConfig(model="gpt-4o", api_key="test"),
        max_concurrent_agents=4,
        chunk_size=1500,
        output_format="jsonl",
    )
    assert config.llm.model == "gpt-4o"
    assert config.max_concurrent_agents == 4
    assert config.chunk_size == 1500
    assert config.output_format == "jsonl"

    # Validation
    try:
        config = HarnessConfig(output_format="invalid")
        assert False, "Should have raised"
    except Exception:
        pass

    print("✅ Config tests passed")


if __name__ == "__main__":
    try:
        test_models()
        test_tools()
        test_skills()
        test_memory()
        test_harness_structure()
        test_prompts()
        test_config()
        print("\n🎉 All integration tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
