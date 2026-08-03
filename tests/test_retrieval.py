"""Tests for the deterministic warning-sign tool and chunk merging."""

from src.agents.retrieval import _dedupe, check_warning_signs


def test_matches_literal_warning_signs():
    matched = check_warning_signs("I have severe abdominal pain and bleeding gums.")
    assert "severe abdominal pain" in matched
    assert "bleeding gums" in matched


def test_matches_colloquial_phrasings():
    assert "persistent vomiting" in check_warning_signs("he keeps vomiting")
    assert "persistent vomiting" in check_warning_signs("she can't keep anything down")
    assert "severe abdominal pain" in check_warning_signs("bad stomach pain since morning")
    assert "difficulty breathing" in check_warning_signs("shortness of breath")
    assert "reduced urine output" in check_warning_signs("he hasn't peed all day")
    assert "blood in vomit" in check_warning_signs("he is vomiting blood")
    assert "blood in vomit" in check_warning_signs("she threw up blood")
    assert "blood in stool" in check_warning_signs("there is blood in his stool")


def test_no_false_positive_on_ordinary_question():
    assert check_warning_signs("How do I stop mosquitoes breeding in my garden?") == []


def test_no_duplicate_signs():
    matched = check_warning_signs("nosebleed and a nose bleed and bleeding gums")
    assert len(matched) == len(set(matched))


def test_dedupe_keeps_highest_score_per_chunk():
    chunks = [
        {"source_doc": "a.pdf", "chunk_index": 1, "score": 0.4},
        {"source_doc": "a.pdf", "chunk_index": 1, "score": 0.9},
        {"source_doc": "b.pdf", "chunk_index": 1, "score": 0.5},
    ]
    result = _dedupe(chunks)
    assert len(result) == 2
    assert max(chunk["score"] for chunk in result if chunk["source_doc"] == "a.pdf") == 0.9
