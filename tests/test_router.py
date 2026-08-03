"""
Router tests with the provider stubbed out.

The point of these is the failure behaviour, not the happy path: what the
router does when the model returns something it wasn't asked for.
"""

import pytest

from src.agents import planner, router
from src.providers import ProviderError


def stub(monkeypatch, module, payload):
    def fake_chat_json(**kwargs):
        if isinstance(payload, Exception):
            raise payload
        return payload

    monkeypatch.setattr(module, "chat_json", fake_chat_json)


def test_normal_classification_passes_through(monkeypatch):
    stub(monkeypatch, router, {"category": "prevention_advice", "urgency": "normal", "reason": "x"})
    result = router.route_query("how do I stop mosquitoes breeding?")

    assert result["category"] == "prevention_advice"
    assert result["urgency"] == "normal"


def test_emergency_urgency_forces_emergency_category(monkeypatch):
    # The model can disagree with itself; the two fields must not.
    stub(monkeypatch, router, {"category": "symptom_check", "urgency": "emergency", "reason": ""})
    result = router.route_query("she is bleeding from the gums")

    assert result["category"] == "emergency"
    assert result["urgency"] == "emergency"


def test_emergency_category_forces_emergency_urgency(monkeypatch):
    stub(monkeypatch, router, {"category": "emergency", "urgency": "normal", "reason": ""})
    assert router.route_query("collapsed")["urgency"] == "emergency"


def test_unknown_category_fails_toward_escalation(monkeypatch):
    stub(monkeypatch, router, {"category": "banana", "urgency": "normal", "reason": ""})
    result = router.route_query("what is dengue?")

    assert result["category"] == "emergency"
    assert result["urgency"] == "emergency"


def test_missing_fields_fail_toward_escalation(monkeypatch):
    stub(monkeypatch, router, {})
    assert router.route_query("what is dengue?")["urgency"] == "emergency"


def test_provider_outage_is_not_disguised_as_advice(monkeypatch):
    # An outage must surface as an error, not as an escalation message that
    # looks like a considered clinical judgement.
    stub(monkeypatch, router, ProviderError("groq is down"))
    with pytest.raises(ProviderError):
        router.route_query("what is dengue?")


def test_planner_always_keeps_the_original_query(monkeypatch):
    stub(monkeypatch, planner, {"sub_queries": ["dengue warning signs"]})
    result = planner.decompose_query("should I worry about my fever?", "symptom_check")

    assert "should I worry about my fever?" in result["sub_queries"]
    assert "dengue warning signs" in result["sub_queries"]


def test_planner_survives_a_junk_decomposition(monkeypatch):
    stub(monkeypatch, planner, {"sub_queries": [None, "", 42]})
    result = planner.decompose_query("what is dengue?", "symptom_check")

    assert result["sub_queries"] == ["what is dengue?"]


def test_planner_caps_the_number_of_sub_queries(monkeypatch):
    stub(monkeypatch, planner, {"sub_queries": [f"q{index}" for index in range(10)]})
    result = planner.decompose_query("a big question", "symptom_check")

    assert len(result["sub_queries"]) == planner.MAX_SUB_QUERIES + 1  # + the original
