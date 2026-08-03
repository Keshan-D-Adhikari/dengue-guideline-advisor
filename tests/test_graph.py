"""
Tests for graph control flow — the branches, without calling a model.

The two edges tested here are the ones carrying the safety guarantees:
emergencies must bypass generation, and the retry loop must terminate.
"""

from src import config
from src.graph.build_graph import (
    MAX_RETRIES,
    build_graph,
    emergency_node,
    route_after_router,
    route_after_verifier,
)


def test_graph_compiles():
    assert build_graph() is not None


def test_emergency_urgency_bypasses_the_planner():
    assert route_after_router({"urgency": "emergency"}) == "emergency"


def test_normal_urgency_reaches_the_planner():
    assert route_after_router({"urgency": "normal"}) == "planner"


def test_emergency_node_returns_the_fixed_message_without_a_model():
    state = {"user_query": "he is vomiting blood and very drowsy", "trace": []}
    result = emergency_node(state)

    assert result["final_answer"] == config.EMERGENCY_MESSAGE
    assert result["chunks"] == []
    assert result["trace"][-1]["detail"]["llm_calls"] == 0
    # The deterministic tool still runs, so the trace records what was seen.
    assert "blood in vomit" in result["warning_signs_matched"]


def test_approved_answer_goes_straight_to_finalize():
    assert route_after_verifier({"approved": True, "retry_count": 0}) == "finalize"


def test_first_rejection_triggers_one_retry():
    assert route_after_verifier({"approved": False, "retry_count": 1}) == "synthesis"


def test_retry_loop_terminates():
    state = {"approved": False, "retry_count": MAX_RETRIES + 1}
    assert route_after_verifier(state) == "finalize"
