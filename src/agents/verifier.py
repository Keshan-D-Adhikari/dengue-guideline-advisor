"""
Verifier agent (pattern: Reflection / self-critique).

Independently checks the draft answer against the cited chunks before it
reaches the user. Uses config.VERIFIER_MODEL — deliberately a DIFFERENT
model, and a different vendor, than SYNTHESIS_MODEL, so this is a genuine
second opinion rather than the same model grading its own work.

Message in:  {"draft_answer": str, "citations": list[str], "chunks": list[dict]}
Message out: {"approved": bool, "issues": list[str],
              "revised_answer": str | None}

Two independent layers, and the code layer wins:

  1. safety_issues() — deterministic Python rules. No model involved, so
     no prompt injection, no sampling variance, no "the model was confident
     it was fine". These are the invariants that must hold for every answer
     the app will ever produce.

  2. an LLM critique for the things a rule cannot see — whether a claim is
     actually supported by the passage it cites, whether the answer drifted
     off the question.

approved requires BOTH. A model saying "looks good" cannot override a
failing rule; that inversion is the whole point of enforcing safety in
code rather than in a prompt.
"""

from __future__ import annotations

import re

from src import config
from src.models.state import VerifierOutput
from src.providers import chat_json

# "see a doctor", "seek medical attention", "go to hospital", ...
_ESCALATION_PATTERNS = [
    r"seek (?:medical|immediate|urgent|professional) ",
    r"(?:see|contact|consult|call) (?:a |your |the )?(?:doctor|physician|clinician|"
    r"health(?:care)? (?:professional|provider|worker)|nurse|emergency)",
    r"go to (?:a |the )?(?:hospital|clinic|emergency|nearest)",
    r"emergency (?:services|department|room|number)",
    r"medical attention",
]

# A dose is the single most dangerous thing to get wrong, and the corpus is
# a public-guidance corpus, not a prescribing manual.
_DOSE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|ml|g)\b(?:/(?:kg|day|dose))?", re.IGNORECASE
)

# Statements that assert a diagnosis rather than describe guidance.
_DIAGNOSIS_PATTERNS = [
    r"\byou (?:have|are suffering from|are infected with) (?:dengue|severe dengue|dhf)\b",
    r"\bthis (?:is|confirms) dengue\b",
    r"\byou (?:do not|don'?t) have dengue\b",
]

_CITATION_MARKER = re.compile(r"\[S\d+\]")


def has_escalation_advice(text: str) -> bool:
    """True if the answer tells the reader how and when to get real care."""
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in _ESCALATION_PATTERNS)


def safety_issues(
    draft_answer: str,
    chunks: list,
    category: str = "",
    warning_signs_matched: list | None = None,
) -> list[str]:
    """Deterministic safety rules. Returns a list of violations, [] if clean.

    Pure: no model, no network, no state. Every rule here is a unit test in
    tests/test_verifier.py.
    """
    issues: list[str] = []
    text = draft_answer or ""

    if not text.strip():
        return ["The answer is empty."]

    needs_escalation = (
        category in config.CATEGORIES_REQUIRING_ESCALATION_ADVICE
        or bool(warning_signs_matched)
    )
    if needs_escalation and not has_escalation_advice(text):
        issues.append(
            "Symptom-related answer does not tell the reader to seek medical "
            "attention. Add an explicit instruction to contact a health "
            "professional or emergency services."
        )

    if chunks and not _CITATION_MARKER.search(text):
        issues.append(
            "No [S#] source markers in the answer, so no claim is traceable to "
            "a source passage. Cite the passage behind each claim."
        )

    corpus = " ".join(str(chunk.get("text", "")) for chunk in chunks).lower()
    for dose in set(_DOSE_PATTERN.findall(text)):
        if dose.lower() not in corpus:
            issues.append(
                f"Answer states a dose ('{dose}') that does not appear in the "
                f"source passages. Remove it."
            )

    for pattern in _DIAGNOSIS_PATTERNS:
        if re.search(pattern, text.lower()):
            issues.append(
                "Answer asserts a diagnosis. This tool is decision support only "
                "— describe what the guidelines say instead."
            )
            break

    for sign in warning_signs_matched or []:
        if sign.lower() not in text.lower():
            issues.append(
                f"The user described the warning sign '{sign}' and the answer "
                f"does not address it."
            )

    return issues


CRITIQUE_SYSTEM_PROMPT = """You are reviewing another model's answer to a
question about dengue, checking it against the source passages it was given.
You did not write it. Be strict.

Reply with JSON only:
{"approved": true|false, "issues": ["..."], "revised_answer": null}

Reject (approved=false) if any of these hold:
- A factual claim is not supported by the passage it cites, or by any passage.
- A citation marker points at a passage that does not say what is claimed.
- The answer contradicts a passage.
- The answer answers a different question than the one asked.
- A warning sign present in the passages is omitted or downplayed.

Do NOT reject for style, length, tone, or for saying "the source documents
do not cover this" — that is the correct behaviour when they don't.

issues: one short, specific sentence per problem. Empty list if approved.
revised_answer: null unless the fix is a small factual correction you can
make using only the passages, in which case supply the corrected answer."""


def verify_answer(
    draft_answer: str,
    citations: list,
    chunks: list,
    category: str = "",
    warning_signs_matched: list | None = None,
) -> dict:
    """Check a draft against the passages and the hard safety rules."""
    hard_issues = safety_issues(draft_answer, chunks, category, warning_signs_matched)

    sources = "\n\n".join(
        f"[S{index}] {chunk.get('source_doc', 'unknown')}"
        f"{' § ' + chunk['section'] if chunk.get('section') else ''}\n"
        f"{chunk.get('text', '')}"
        for index, chunk in enumerate(chunks, start=1)
    )

    try:
        raw = chat_json(
            provider="openrouter",
            model=config.VERIFIER_MODEL,
            system=CRITIQUE_SYSTEM_PROMPT,
            user=(
                f"Source passages:\n{sources or '(none provided)'}\n\n"
                f"Cited by the answer: {', '.join(map(str, citations)) or '(none)'}\n\n"
                f"Answer under review:\n{draft_answer}"
            ),
            max_tokens=900,
        )
        parsed = VerifierOutput.model_validate(
            {
                "approved": bool(raw.get("approved", False)),
                "issues": [str(issue) for issue in raw.get("issues", []) if str(issue).strip()],
                "revised_answer": raw.get("revised_answer") or None,
            }
        )
        model_approved, model_issues = parsed.approved, parsed.issues
        revised_answer = parsed.revised_answer
    except Exception as exc:
        # A dead verifier must not silently become an approval. Withhold
        # approval and say why, so the graph can retry and the trace shows it.
        model_approved = False
        model_issues = [f"Verification could not be completed ({type(exc).__name__})."]
        revised_answer = None

    # The code layer is not overridable by the model layer.
    return {
        "approved": model_approved and not hard_issues,
        "issues": hard_issues + model_issues,
        "revised_answer": revised_answer if not hard_issues else None,
    }


def enforce_disclaimer(answer: str, category: str, warning_signs_matched: list | None) -> str:
    """Last-resort guarantee, applied after retries are exhausted.

    If the pipeline still hasn't produced an answer carrying escalation
    advice on a symptom-related query, append it. The invariant holds for
    every answer the user sees, regardless of what any model did.
    """
    needs_escalation = (
        category in config.CATEGORIES_REQUIRING_ESCALATION_ADVICE
        or bool(warning_signs_matched)
    )
    if not needs_escalation or has_escalation_advice(answer):
        return answer

    return (
        f"{answer.rstrip()}\n\n"
        "**If you are worried, or if symptoms worsen, contact a health "
        "professional or your local emergency services immediately.**"
    )
