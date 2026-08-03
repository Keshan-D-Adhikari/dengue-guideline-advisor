"""
Retrieval agent (pattern: Tool-use).

Given sub-queries, calls the vector store tool (see src/rag/vectorstore.py)
and returns the retrieved chunks with source metadata. Also exposes a
deterministic `check_warning_signs` tool over the structured red-flag list
in config.WARNING_SIGNS, so that specific fact is never left to an LLM to
paraphrase — a missed warning sign is the failure mode with the worst
consequences in this app, and a string match cannot hallucinate.

Message in:  {"sub_queries": list[str]}
Message out: {"chunks": list[{"text": str, "source_doc": str,
                               "section": str, "score": float}],
              "warning_signs_matched": list[str]}

Two tools, deliberately different in kind:
  - vector search  — fuzzy, for finding relevant guidance
  - warning-sign check — exact, for a fact that must not be approximated
"""

from __future__ import annotations

import re

from src import config
from src.providers import chat_json
from src.rag import vectorstore

MAX_CHUNKS_TO_SYNTHESIS = 8
RERANK_CANDIDATE_LIMIT = 20

# Extra spellings that the config list doesn't cover literally. Maps a
# regex to the canonical warning sign it indicates.
_SYNONYM_PATTERNS: list[tuple[str, str]] = [
    (r"\bthrow(?:ing)? up (?:a lot|repeatedly|constantly)\b", "persistent vomiting"),
    (r"\bkeeps? vomiting\b", "persistent vomiting"),
    (r"\bcan'?t keep (?:anything|food|water|fluids) down\b", "persistent vomiting"),
    (r"\bstomach (?:pain|ache)\b", "severe abdominal pain"),
    (r"\bbelly (?:pain|ache)\b", "severe abdominal pain"),
    (r"\bgums? (?:are )?bleeding\b", "bleeding gums"),
    (r"\bnose ?bleed\b", "nosebleed"),
    (r"\bblood in (?:my |the )?(?:vomit|sick)\b", "blood in vomit"),
    (r"\b(?:vomit(?:ing|ed)?|throw(?:ing|n)? up|threw up|coughing up) blood\b", "blood in vomit"),
    (r"\bblack(?:ish)? stool\b", "blood in stool"),
    (r"\bblood in (?:my |the |his |her )?(?:stool|poo|faeces|feces)\b", "blood in stool"),
    (r"\bshort(?:ness)? of breath\b", "difficulty breathing"),
    (r"\bstruggling to breathe\b", "difficulty breathing"),
    (r"\bnot (?:passing |peeing |urinating)\b", "reduced urine output"),
    (r"\bhasn'?t (?:peed|urinated)\b", "reduced urine output"),
    (r"\bvery (?:sleepy|drowsy)\b", "lethargy"),
    (r"\bcold(?: and)? clammy\b", "cold clammy skin"),
]


def check_warning_signs(text: str) -> list[str]:
    """Deterministic tool: which official red flags appear in `text`.

    Pure and offline on purpose — no model call, no network, fully
    unit-testable, and it returns the same answer every time.
    """
    lowered = text.lower()
    matched: list[str] = []

    for sign in config.WARNING_SIGNS:
        if sign in lowered:
            matched.append(sign)

    for pattern, sign in _SYNONYM_PATTERNS:
        if sign not in matched and re.search(pattern, lowered):
            matched.append(sign)

    return matched


def _dedupe(chunks: list[dict]) -> list[dict]:
    """Collapse chunks retrieved by more than one sub-query, keeping the best score."""
    best: dict[tuple[str, int], dict] = {}
    for chunk in chunks:
        key = (chunk["source_doc"], chunk["chunk_index"])
        if key not in best or chunk["score"] > best[key]["score"]:
            best[key] = chunk
    return list(best.values())


def _rerank(user_query: str, chunks: list[dict]) -> list[dict]:
    """Re-order candidates by LLM-judged relevance (config.RERANK_MODEL).

    Vector similarity is lexical-ish: it ranks a chunk that repeats the
    query's words above one that actually answers it. This second pass is
    best-effort — on any failure the similarity order is kept, because a
    worse ordering is much cheaper than a failed request.
    """
    numbered = "\n\n".join(
        f"[{index}] ({chunk['source_doc']} § {chunk['section']}) {chunk['text'][:500]}"
        for index, chunk in enumerate(chunks)
    )
    try:
        raw = chat_json(
            provider="groq",
            model=config.RERANK_MODEL,
            system=(
                "You score passages for how well they answer a question about "
                "dengue.\n\n"
                'Reply with JSON only: {"ranking": [{"id": 0, "score": 0.0}, ...]}\n'
                "score is 0.0 (irrelevant) to 1.0 (directly answers it). "
                "Include every passage id exactly once. Do not answer the question."
            ),
            user=f"Question: {user_query}\n\nPassages:\n{numbered}",
            max_tokens=600,
        )
        scores = {
            int(entry["id"]): float(entry["score"])
            for entry in raw.get("ranking", [])
            if isinstance(entry, dict) and "id" in entry and "score" in entry
        }
        if not scores:
            return chunks
    except Exception:
        return chunks

    for index, chunk in enumerate(chunks):
        if index in scores:
            chunk["rerank_score"] = round(scores[index], 4)

    return sorted(
        chunks,
        key=lambda chunk: chunk.get("rerank_score", chunk["score"]),
        reverse=True,
    )


def retrieve_chunks(
    sub_queries: list,
    user_query: str = "",
    rerank: bool = True,
) -> dict:
    """Search the guideline corpus for every sub-query and merge the results."""
    collected: list[dict] = []
    for query in sub_queries:
        collected.extend(vectorstore.similarity_search(query, k=config.TOP_K_RETRIEVED))

    chunks = _dedupe(collected)
    chunks.sort(key=lambda chunk: chunk["score"], reverse=True)
    chunks = chunks[:RERANK_CANDIDATE_LIMIT]

    if rerank and chunks and user_query:
        chunks = _rerank(user_query, chunks)

    return {
        "chunks": chunks[:MAX_CHUNKS_TO_SYNTHESIS],
        "warning_signs_matched": check_warning_signs(user_query or " ".join(map(str, sub_queries))),
    }
