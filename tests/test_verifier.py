"""
Tests for the deterministic half of the verifier.

These are the rules that must hold for every answer the app produces,
regardless of what any model says — so they are tested without a model.
"""

from src.agents.verifier import (
    enforce_disclaimer,
    has_escalation_advice,
    safety_issues,
)

CHUNKS = [
    {
        "text": "Warning signs include severe abdominal pain and persistent vomiting.",
        "source_doc": "who.pdf",
        "section": "Warning signs",
        "chunk_index": 0,
    }
]


def test_detects_escalation_advice_in_several_phrasings():
    assert has_escalation_advice("Seek medical attention immediately.")
    assert has_escalation_advice("You should see a doctor if this continues.")
    assert has_escalation_advice("Go to the nearest hospital.")
    assert has_escalation_advice("Call your local emergency services.")
    assert not has_escalation_advice("Drink plenty of fluids and rest.")


def test_symptom_answer_without_escalation_advice_is_rejected():
    issues = safety_issues(
        "Rest and drink fluids [S1].", CHUNKS, category="symptom_check"
    )
    assert any("seek medical attention" in issue.lower() for issue in issues)


def test_prevention_answer_does_not_need_escalation_advice():
    issues = safety_issues(
        "Remove standing water from containers [S1].",
        CHUNKS,
        category="prevention_advice",
    )
    assert issues == []


def test_missing_citation_markers_are_rejected():
    issues = safety_issues(
        "Remove standing water from containers.",
        CHUNKS,
        category="prevention_advice",
    )
    assert any("[S#]" in issue for issue in issues)


def test_dose_not_present_in_sources_is_rejected():
    issues = safety_issues(
        "Take 500 mg of paracetamol every six hours [S1]. See a doctor if worse.",
        CHUNKS,
        category="symptom_check",
    )
    assert any("dose" in issue.lower() for issue in issues)


def test_diagnosis_claim_is_rejected():
    issues = safety_issues(
        "Based on this, you have dengue [S1]. Seek medical attention.",
        CHUNKS,
        category="symptom_check",
    )
    assert any("diagnosis" in issue.lower() for issue in issues)


def test_matched_warning_sign_must_be_addressed():
    issues = safety_issues(
        "Rest and drink fluids [S1]. See a doctor if you are worried.",
        CHUNKS,
        category="symptom_check",
        warning_signs_matched=["persistent vomiting"],
    )
    assert any("persistent vomiting" in issue for issue in issues)


def test_empty_answer_is_rejected():
    assert safety_issues("   ", CHUNKS, category="symptom_check")


def test_clean_symptom_answer_passes():
    issues = safety_issues(
        "Severe abdominal pain is a warning sign [S1]. Seek medical attention "
        "if it develops.",
        CHUNKS,
        category="symptom_check",
        warning_signs_matched=["severe abdominal pain"],
    )
    assert issues == []


def test_enforce_disclaimer_appends_when_missing():
    out = enforce_disclaimer("Rest and drink fluids.", "symptom_check", [])
    assert has_escalation_advice(out)
    assert out.startswith("Rest and drink fluids.")


def test_enforce_disclaimer_is_a_noop_when_already_present():
    text = "Rest and drink fluids. Seek medical attention if you worsen."
    assert enforce_disclaimer(text, "symptom_check", []) == text


def test_enforce_disclaimer_skips_non_symptom_categories():
    text = "Remove standing water."
    assert enforce_disclaimer(text, "prevention_advice", []) == text
