"""
Planner agent (pattern: Planning / task-decomposition).

Breaks a query into the sub-questions that need separate retrieval,
e.g. a symptom_check query decomposes into: warning-signs check,
fluid-intake guidance, referral-criteria check.

Message in:  {"user_query": str, "category": str}
Message out: {"sub_queries": list[str]}

Why decompose at all: a single embedding of "I have a fever and a rash,
what should I do?" lands between the symptom section and the home-care
section and retrieves a muddy mixture of both. Three targeted sub-queries
each land squarely in one section, and the union covers the question.

Runs on config.RERANK_MODEL (Groq) — the task is short-form rewriting, not
reasoning, so it stays on the cheap tier alongside the router. That choice
is recorded in the README model table.

Emergency queries never reach this agent; the graph short-circuits them
before planning. That is an orchestrator decision, not this agent's job.
"""

from __future__ import annotations

from src import config
from src.models.state import PlannerOutput
from src.providers import chat_json

MAX_SUB_QUERIES = 4

SYSTEM_PROMPT = """You rewrite a user's dengue question into standalone search
queries for a retrieval system over official dengue clinical guidelines.

Reply with JSON only: {"sub_queries": ["...", "..."]}

Rules:
- Between 1 and 4 queries. Use 1 if the question is already single-purpose.
- Each query must stand alone — no pronouns, no "it", no reference to the
  other queries.
- Write them as guideline terminology, not as the user's phrasing.
  "should I be worried about my fever" -> "dengue fever warning signs
  requiring medical review"
- Cover the distinct things the user needs, not paraphrases of one thing.
- Do not answer the question. Only produce search queries."""


def decompose_query(user_query: str, category: str) -> dict:
    """Split a query into independently-retrievable sub-questions."""
    raw = chat_json(
        provider="groq",
        model=config.RERANK_MODEL,
        system=SYSTEM_PROMPT,
        user=f"Category: {category}\nQuestion: {user_query}",
        max_tokens=250,
    )

    try:
        parsed = PlannerOutput.model_validate({"sub_queries": raw.get("sub_queries", [])})
        sub_queries = [
            query.strip()
            for query in parsed.sub_queries
            if isinstance(query, str) and query.strip()
        ][:MAX_SUB_QUERIES]
    except Exception:
        sub_queries = []

    # Always retrieve against the original wording too. The rewrites are an
    # improvement on average, not always — keeping the raw query means a bad
    # decomposition degrades retrieval instead of breaking it.
    if user_query.strip() not in sub_queries:
        sub_queries.append(user_query.strip())

    return {"sub_queries": sub_queries}
