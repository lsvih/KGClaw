"""Tests for multi-paradigm ontology builder (kgclaw/ontology_builder.py)."""
import pytest
from kgclaw.ontology_builder import OntologyBuilder, _extract_nouns_spacy
from kgclaw.models import Document, LLMConfig, Ontology


class TestOntologyBuilder:
    """Test OntologyBuilder initialization and mode routing."""

    def test_init(self):
        builder = OntologyBuilder(LLMConfig())
        assert builder is not None
        assert builder.llm_config is not None

    def test_create_agent(self):
        builder = OntologyBuilder(LLMConfig())
        agent = builder._create_agent("test", "Be helpful.")
        assert agent is not None

    def test_auto_select_mode_short_docs(self):
        builder = OntologyBuilder(LLMConfig())
        docs = [Document(text="Short text.")]
        mode = builder._auto_select_mode(docs)
        assert mode in ("text-to-ontology", "relation-to-ontology",
                        "ht-relation-to-ontology", "affinity-clustering")

    def test_auto_select_mode_empty(self):
        builder = OntologyBuilder(LLMConfig())
        assert builder._auto_select_mode([]) == "text-to-ontology"

    def test_build_empty_docs(self):
        builder = OntologyBuilder(LLMConfig())
        result = builder.build([], mode="text-to-ontology")
        assert result is None

    def test_normalize_type_name_simple(self):
        builder = OntologyBuilder(LLMConfig())
        assert builder._normalize_type_name("  Hello World  ") == "Hello World"
        assert builder._normalize_type_name("hello") == "Hello"
        assert builder._normalize_type_name("") == "Unknown"

    def test_normalize_type_name_with_numbers(self):
        builder = OntologyBuilder(LLMConfig())
        assert builder._normalize_type_name("123. Type Name") == "Type Name"

    def test_result_to_ontology(self):
        builder = OntologyBuilder(LLMConfig())
        result = {
            "ontology_name": "test",
            "entity_types": [
                {"name": "Person", "description": "A person", "parent": None},
                {"name": "Employee", "description": "An employee", "parent": "Person"},
            ],
            "relation_types": [
                {"name": "works_at", "description": "Works at", "domain": "Person", "range": "Organization"},
            ],
        }
        onto = builder._result_to_ontology(result, "TO")
        assert onto is not None
        assert len(onto.entity_types) == 2
        assert onto.entity_types[1].parent == "Person"
        assert len(onto.relation_types) == 1

    def test_result_to_ontology_dedup(self):
        builder = OntologyBuilder(LLMConfig())
        result = {
            "ontology_name": "test",
            "entity_types": [
                {"name": "Person", "description": "A person"},
                {"name": "person", "description": "Duplicate"},  # same after normalize
            ],
            "relation_types": [],
        }
        onto = builder._result_to_ontology(result, "TO")
        assert len(onto.entity_types) == 1  # deduplicated


class TestNounExtraction:
    """Test spaCy noun extraction with fallback."""

    def test_extract_nouns_fallback(self):
        nouns = _extract_nouns_spacy(["Hello World. Machine Learning is great."])
        # Fallback extracts capitalized phrases
        assert len(nouns) > 0

    def test_extract_nouns_empty(self):
        nouns = _extract_nouns_spacy([""])
        assert nouns == []

    def test_extract_nouns_dedup(self):
        text = "Machine Learning. Machine Learning is important."
        nouns = _extract_nouns_spacy([text])
        # Should be deduplicated
        assert len(nouns) == len(set(n.lower() for n in nouns))


class TestOntologyBuilderModes:
    """Test that all modes are recognized."""

    def test_all_modes_recognized(self):
        builder = OntologyBuilder(LLMConfig())
        docs = [Document(text="Test content.")]
        for mode in ("text-to-ontology", "relation-to-ontology",
                     "ht-relation-to-ontology", "affinity-clustering",
                     "dense-ontology"):
            # Modes should not raise errors (they may return None without real LLM)
            try:
                result = builder.build(docs, mode=mode)
            except Exception as e:
                # Only fail if it's not an LLM-related error
                if "compact_messages" not in str(e) and "api_key" not in str(e).lower():
                    raise
