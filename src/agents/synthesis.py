"""
Synthesis agent.

Combines the user's query with retrieved chunks into a single grounded
answer, citing which chunks it drew from. This is the highest-stakes
generation step — use config.SYNTHESIS_MODEL (the strongest reasoning
model in your stack).

Message in:  {"user_query": str, "chunks": list[dict]}
Message out: {"draft_answer": str, "citations": list[str]}

TODO: implement. The prompt should instruct the model to only state
things traceable to the provided chunks, and to say "guidance not found
in source documents" rather than filling gaps from its own knowledge —
this matters both for correctness and for the reflection step downstream.
"""


def synthesize_answer(user_query: str, chunks: list) -> dict:
    raise NotImplementedError
