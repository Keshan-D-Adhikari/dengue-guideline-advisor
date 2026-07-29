"""
Retrieval agent (pattern: Tool-use).

Given sub-queries, calls the vector store tool (see src/rag/vectorstore.py)
and returns the retrieved chunks with source metadata. Optionally also
exposes a deterministic `check_warning_signs` tool over a small structured
list of official red-flag symptoms, so that specific fact isn't left to
LLM paraphrase.

Message in:  {"sub_queries": list[str]}
Message out: {"chunks": list[{"text": str, "source_doc": str,
                               "section": str, "score": float}]}

TODO: implement. Use src/rag/vectorstore.py for the actual similarity
search. If you add re-ranking, that's where config.RERANK_MODEL gets used
to score/reorder the raw vector-similarity results.
"""


def retrieve_chunks(sub_queries: list) -> dict:
    raise NotImplementedError
