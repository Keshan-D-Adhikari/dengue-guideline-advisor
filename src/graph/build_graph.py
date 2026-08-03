"""
LangGraph state graph wiring the agents together.

Flow:
  router -> [emergency? -> fixed escalation message : planner]
  planner -> retrieval -> synthesis -> verifier
  verifier -> (issues found, retries < 1) -> synthesis
  verifier -> (approved, or retries exhausted) -> finalize -> END

Two control-flow decisions carry most of the safety weight:

  * The emergency branch never reaches a generation step. It returns a
    fixed, human-written message from config. An answer that cannot be
    generated cannot be hallucinated, and this is precisely the case where
    a hallucination would do the most harm.

  * The verifier retry is capped at one. An uncapped reflection loop
    between two models that disagree runs until the budget dies; capping it
    and falling through to `finalize` — which shows the answer alongside the
    unresolved objections rather than hiding them — degrades honestly.

Every node appends to state["trace"], which app.py renders in the sidebar.
That trace is the observable evidence that five distinct agents ran.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src import config
from src.agents.planner import decompose_query
from src.agents.retrieval import check_warning_signs, retrieve_chunks
from src.agents.synthesis import synthesize_answer
from src.agents.verifier import enforce_disclaimer, verify_answer
from src.agents.router import route_query
from src.models.state import GraphState

MAX_RETRIES = 1


def _trace(state: GraphState, agent: str, summary: str, **detail) -> list:
    return list(state.get("trace", [])) + [
        {"agent": agent, "summary": summary, "detail": detail}
    ]


# --- Nodes -----------------------------------------------------------------

def router_node(state: GraphState) -> dict:
    result = route_query(state["user_query"])
    return {
        "category": result["category"],
        "urgency": result["urgency"],
        "retry_count": 0,
        "trace": _trace(
            state,
            "router",
            f"{result['category']} / {result['urgency']}",
            reason=result.get("reason", ""),
            model=config.ROUTER_MODEL,
        ),
    }


def emergency_node(state: GraphState) -> dict:
    """Fixed escalation message. No LLM call happens on this path."""
    return {
        "final_answer": config.EMERGENCY_MESSAGE,
        "citations": [],
        "chunks": [],
        "approved": True,
        "issues": [],
        "warning_signs_matched": check_warning_signs(state["user_query"]),
        "trace": _trace(
            state,
            "emergency_escalation",
            "short-circuited before any generation",
            llm_calls=0,
        ),
    }


def planner_node(state: GraphState) -> dict:
    result = decompose_query(state["user_query"], state.get("category", ""))
    return {
        "sub_queries": result["sub_queries"],
        "trace": _trace(
            state,
            "planner",
            f"{len(result['sub_queries'])} sub-quer"
            f"{'y' if len(result['sub_queries']) == 1 else 'ies'}",
            sub_queries=result["sub_queries"],
            model=config.RERANK_MODEL,
        ),
    }


def retrieval_node(state: GraphState) -> dict:
    result = retrieve_chunks(state.get("sub_queries", []), user_query=state["user_query"])
    sources = sorted({chunk["source_doc"] for chunk in result["chunks"]})
    return {
        "chunks": result["chunks"],
        "warning_signs_matched": result["warning_signs_matched"],
        "trace": _trace(
            state,
            "retrieval",
            f"{len(result['chunks'])} chunk(s) from {len(sources)} document(s)",
            sources=sources,
            warning_signs_matched=result["warning_signs_matched"],
            tools=["vector_search", "check_warning_signs"],
        ),
    }


def synthesis_node(state: GraphState) -> dict:
    attempt = state.get("retry_count", 0)
    result = synthesize_answer(
        state["user_query"],
        state.get("chunks", []),
        warning_signs_matched=state.get("warning_signs_matched", []),
        issues=state.get("issues", []) if attempt else None,
    )
    return {
        "draft_answer": result["draft_answer"],
        "citations": result["citations"],
        "trace": _trace(
            state,
            "synthesis",
            f"draft {'revised' if attempt else 'written'}, "
            f"{len(result['citations'])} citation(s)",
            attempt=attempt + 1,
            model=config.SYNTHESIS_MODEL,
        ),
    }


def verifier_node(state: GraphState) -> dict:
    result = verify_answer(
        state.get("draft_answer", ""),
        state.get("citations", []),
        state.get("chunks", []),
        category=state.get("category", ""),
        warning_signs_matched=state.get("warning_signs_matched", []),
    )

    updates = {
        "approved": result["approved"],
        "issues": result["issues"],
        "trace": _trace(
            state,
            "verifier",
            "approved" if result["approved"] else f"rejected ({len(result['issues'])} issue(s))",
            issues=result["issues"],
            model=config.VERIFIER_MODEL,
        ),
    }

    # A small factual correction from the verifier is accepted directly
    # rather than spending a whole synthesis round-trip on it.
    if result["approved"] and result.get("revised_answer"):
        updates["draft_answer"] = result["revised_answer"]

    if not result["approved"]:
        updates["retry_count"] = state.get("retry_count", 0) + 1

    return updates


def finalize_node(state: GraphState) -> dict:
    """Assemble the answer the user actually sees.

    If the verifier never approved, the answer still ships — with the
    objections attached. Showing a caveated answer is more useful than
    showing nothing, and hiding the caveat would be worse than both.
    """
    answer = state.get("draft_answer", "")

    if not state.get("approved") and state.get("issues"):
        objections = "\n".join(f"- {issue}" for issue in state["issues"])
        answer = (
            f"{answer.rstrip()}\n\n"
            f"---\n"
            f"**Unverified.** An independent check flagged the following and they "
            f"were not resolved:\n{objections}\n\n"
            f"Treat this answer with extra caution and confirm it with a health "
            f"professional."
        )

    answer = enforce_disclaimer(
        answer, state.get("category", ""), state.get("warning_signs_matched", [])
    )

    return {
        "final_answer": answer,
        "trace": _trace(
            state,
            "finalize",
            "approved" if state.get("approved") else "delivered with caveats",
            retries_used=state.get("retry_count", 0),
        ),
    }


# --- Edges -----------------------------------------------------------------

def route_after_router(state: GraphState) -> str:
    return "emergency" if state.get("urgency") == "emergency" else "planner"


def route_after_verifier(state: GraphState) -> str:
    if not state.get("approved") and state.get("retry_count", 0) <= MAX_RETRIES:
        return "synthesis"
    return "finalize"


def build_graph():
    """Compile the agent graph."""
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("emergency", emergency_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"emergency": "emergency", "planner": "planner"},
    )
    graph.add_edge("emergency", END)
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "synthesis")
    graph.add_edge("synthesis", "verifier")
    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"synthesis": "synthesis", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


_COMPILED = None


def get_graph():
    """Compile once and reuse — building the graph on every message is waste."""
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED


def run(user_query: str) -> dict:
    """Single entry point for app.py.

    Returns the full final state: final_answer, citations, chunks, category,
    urgency, approved, issues, and the agent trace.
    """
    return get_graph().invoke({"user_query": user_query, "trace": []})
