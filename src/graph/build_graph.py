"""
LangGraph state graph wiring the agents together.

Flow (see README for the full sequence diagram once written):
  router -> [emergency? -> fixed escalation message : planner]
  planner -> retrieval -> synthesis -> verifier
  verifier -> (issues found, retries < 1) -> synthesis
  verifier -> (approved) -> END

TODO: implement.
  1. Define a TypedDict/pydantic State with fields for every message key
     used across src/agents/*.py (user_query, category, urgency,
     sub_queries, chunks, draft_answer, citations, approved, issues,
     retry_count, final_answer).
  2. Add one node per agent function.
  3. Add a conditional edge after the router node: emergency -> a fixed
     response node; anything else -> planner node.
  4. Add a conditional edge after the verifier node: not approved and
     retry_count < 1 -> synthesis node (increment retry_count); otherwise
     -> END.
  5. Compile the graph and expose a single `run(user_query: str) -> dict`
     function for app.py to call.
"""


def build_graph():
    raise NotImplementedError


def run(user_query: str) -> dict:
    raise NotImplementedError
