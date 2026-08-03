"""
Synthesis agent.

Combines the user's query with retrieved chunks into a single grounded
answer, citing which chunks it drew from. This is the highest-stakes
generation step — it runs on config.SYNTHESIS_MODEL, the strongest
reasoning model in the stack.

Message in:  {"user_query": str, "chunks": list[dict]}
Message out: {"draft_answer": str, "citations": list[str]}

Grounding is enforced three ways, not one:
  1. The prompt forbids anything not traceable to a numbered source.
  2. Sources are numbered [S1], [S2]... so a claim without a marker is
     visibly unsupported, and the verifier can check marker-by-marker.
  3. An empty chunk list never reaches the model — it returns the
     not-found message directly, so a missing corpus cannot be papered
     over with parametric knowledge.
"""

from __future__ import annotations

from src import config
from src.providers import chat_json

NOT_FOUND_MESSAGE = (
    "I could not find guidance on this in the source documents, so I have "
    "nothing to give you here. Please consult a health professional or your "
    "national dengue guidelines directly."
)

SYSTEM_PROMPT = """You answer questions about dengue using ONLY the numbered
source passages provided. You are writing for a member of the public.

Reply with JSON only: {"draft_answer": "...", "citations": ["S1", "S3"]}

Grounding rules — these are absolute:
- Every factual claim must come from a source passage. Mark it inline with
  the source number, e.g. "Persistent vomiting is a warning sign [S2]."
- If the passages do not cover part of the question, say so plainly in the
  answer: "The source documents do not cover X." Do NOT fill the gap from
  your own knowledge, even if you are confident.
- Never give a diagnosis, a drug dose, or a prescription.
- Do not soften or omit a warning sign that appears in the passages.

Style:
- Plain language, short paragraphs or a short list. No jargon unless the
  passage uses it, and then explain it in a few words.
- Be direct. Do not open with a summary of the question.
- If the answer involves symptoms, end with a clear instruction to seek
  medical attention if the person is worried or worsening.

citations: the source numbers you actually used, as strings like "S1"."""


def _format_sources(chunks: list[dict]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        header = f"[S{index}] {chunk.get('source_doc', 'unknown')}"
        section = chunk.get("section")
        if section:
            header += f" § {section}"
        page = chunk.get("page")
        if page:
            header += f" (p.{page})"
        blocks.append(f"{header}\n{chunk.get('text', '')}")
    return "\n\n".join(blocks)


def synthesize_answer(
    user_query: str,
    chunks: list,
    warning_signs_matched: list | None = None,
    issues: list | None = None,
) -> dict:
    """Draft a grounded, cited answer from the retrieved chunks.

    `issues` is populated on a verifier-triggered retry: the second attempt
    sees what the verifier objected to, so the retry is a correction rather
    than a re-roll of the same prompt.
    """
    if not chunks:
        return {"draft_answer": NOT_FOUND_MESSAGE, "citations": []}

    parts = [f"Question: {user_query}", "", "Source passages:", _format_sources(chunks)]

    if warning_signs_matched:
        # Surfaced from the deterministic tool, not from the model's reading.
        parts += [
            "",
            "A deterministic check matched these official warning signs in the "
            "user's own words: " + ", ".join(warning_signs_matched) + ".",
            "Address them explicitly if the passages cover them.",
        ]

    if issues:
        parts += [
            "",
            "A reviewer rejected your previous draft for these reasons. Fix every "
            "one of them:",
            "\n".join(f"- {issue}" for issue in issues),
        ]

    raw = chat_json(
        provider="openrouter",
        model=config.SYNTHESIS_MODEL,
        system=SYSTEM_PROMPT,
        user="\n".join(parts),
        max_tokens=1200,
    )

    draft_answer = str(raw.get("draft_answer", "")).strip()
    if not draft_answer:
        return {"draft_answer": NOT_FOUND_MESSAGE, "citations": []}

    citations = [
        str(citation).strip()
        for citation in raw.get("citations", [])
        if str(citation).strip()
    ]

    return {"draft_answer": draft_answer, "citations": citations}
