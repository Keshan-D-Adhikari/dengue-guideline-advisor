"""
Planner agent (pattern: Planning / task-decomposition).

Breaks a query into the sub-questions that need separate retrieval,
e.g. a symptom_check query decomposes into: warning-signs check,
fluid-intake guidance, referral-criteria check.

Message in:  {"user_query": str, "category": str}
Message out: {"sub_queries": list[str]}

TODO: implement using a Groq or OpenRouter model (your choice — justify
whichever you pick in the README). Skip planning entirely for categories
that don't need it (e.g. "emergency" should short-circuit before this
agent is ever called — that's an orchestrator/graph decision, not this
agent's job).
"""


def decompose_query(user_query: str, category: str) -> dict:
    raise NotImplementedError
