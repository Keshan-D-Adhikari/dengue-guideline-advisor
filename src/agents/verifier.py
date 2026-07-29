"""
Verifier agent (pattern: Reflection / self-critique).

Independently checks the draft answer against the cited chunks before
it reaches the user. Use config.VERIFIER_MODEL — deliberately a
DIFFERENT model than SYNTHESIS_MODEL, so this is a genuine second
opinion rather than the same model grading its own work.

Message in:  {"draft_answer": str, "citations": list[str], "chunks": list[dict]}
Message out: {"approved": bool, "issues": list[str],
              "revised_answer": str | None}

TODO: implement. Also enforce non-negotiable safety rules here in code
(not just via prompting) — e.g. reject/flag any draft that omits the
"seek medical attention" disclaimer for symptom_check answers, regardless
of what the model itself concludes.
"""


def verify_answer(draft_answer: str, citations: list, chunks: list) -> dict:
    raise NotImplementedError
