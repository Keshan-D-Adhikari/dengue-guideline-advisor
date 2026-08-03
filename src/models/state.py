"""
Typed data models shared across the graph.

`GraphState` is the single message bus: every agent reads some of its keys
and writes others, and nothing is passed between agents any other way. The
per-agent contract is documented in each agent module and in the README.

The pydantic models below describe the shape each agent must return. They
are used to validate structured LLM output before it enters the state, so a
malformed model reply fails at the boundary with a clear error instead of
surfacing as a KeyError three nodes later.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

Category = Literal[
    "symptom_check",
    "prevention_advice",
    "reporting_procedure",
    "emergency",
]
Urgency = Literal["normal", "emergency"]


class Chunk(BaseModel):
    """One retrieved passage plus the metadata that makes it citable."""

    text: str
    source_doc: str
    section: str = ""
    chunk_index: int = 0
    score: float = 0.0

    def label(self) -> str:
        """Human-readable citation, e.g. 'who_dengue.pdf § Warning signs'."""
        return f"{self.source_doc} § {self.section}" if self.section else self.source_doc


class RouterOutput(BaseModel):
    category: Category
    urgency: Urgency
    reason: str = ""


class PlannerOutput(BaseModel):
    sub_queries: list[str] = Field(default_factory=list)


class SynthesisOutput(BaseModel):
    draft_answer: str
    citations: list[str] = Field(default_factory=list)


class VerifierOutput(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)
    revised_answer: str | None = None


class TraceEntry(TypedDict):
    """One row of the sidebar agent trace in app.py."""

    agent: str
    summary: str
    detail: dict[str, Any]


class GraphState(TypedDict, total=False):
    """State threaded through the LangGraph run.

    total=False because nodes fill it in progressively — the router node
    only ever sees `user_query`.
    """

    user_query: str
    category: str
    urgency: str
    sub_queries: list[str]
    chunks: list[dict]
    warning_signs_matched: list[str]
    draft_answer: str
    citations: list[str]
    approved: bool
    issues: list[str]
    retry_count: int
    final_answer: str
    trace: list[TraceEntry]
