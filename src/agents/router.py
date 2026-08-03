"""
Router agent (pattern: Router).

Classifies an incoming user query so the orchestrator can pick a path.

Message in:  {"user_query": str}
Message out: {"category": "symptom_check" | "prevention_advice"
                        | "reporting_procedure" | "emergency",
              "urgency": "normal" | "emergency"}

Runs on config.ROUTER_MODEL (Groq) in JSON mode. This call is on the
critical path before any other work can start, so the prompt is kept short
and the model is the smallest one in the stack.

Safety note: when the model replies with something unusable, the fallback
is "emergency". A query wrongly sent down the escalation path costs the
user a redundant "seek care" message; a genuine emergency wrongly sent to
the RAG pipeline costs them time they may not have. The asymmetry decides
the default.

A ProviderError is a different thing — the API is down or unconfigured, not
an ambiguous query — so it propagates to app.py and is shown as an error.
Dressing an outage up as medical advice would be worse than saying nothing.
"""

from __future__ import annotations

from src import config
from src.models.state import RouterOutput
from src.providers import chat_json

SYSTEM_PROMPT = """You classify questions about dengue fever for a triage system.

Reply with JSON only, exactly these keys:
{"category": "...", "urgency": "...", "reason": "..."}

category must be one of:
- symptom_check       — asking about symptoms, warning signs, or whether to worry
- prevention_advice   — mosquito control, breeding sites, travel precautions, vaccines
- reporting_procedure — notifying authorities, case reporting, where to seek care
- emergency           — describing severe or rapidly worsening illness right now

urgency must be "emergency" if the person describes any of: severe abdominal
pain, persistent vomiting, bleeding (gums, nose, vomit, stool), difficulty
breathing, cold or clammy skin, collapse, confusion, drowsiness,
unresponsiveness, or passing little/no urine. Otherwise "normal".

Judge urgency from what is described, not from how alarmed the wording is.
A calm description of bleeding is an emergency; an anxious question about a
mild fever is not.

reason: at most 12 words."""


def route_query(user_query: str) -> dict:
    """Classify a query into a category and an urgency level."""
    raw = chat_json(  # ProviderError propagates — see module docstring
        provider="groq",
        model=config.ROUTER_MODEL,
        system=SYSTEM_PROMPT,
        user=user_query,
        max_tokens=120,
    )

    try:
        parsed = RouterOutput.model_validate(
            {
                "category": raw.get("category"),
                "urgency": raw.get("urgency"),
                "reason": str(raw.get("reason", ""))[:120],
            }
        )
    except Exception:
        # Unusable classification — fail toward escalation.
        return {
            "category": "emergency",
            "urgency": "emergency",
            "reason": "router returned an unrecognised category; escalating",
        }

    category = parsed.category
    urgency = parsed.urgency

    # Keep the two fields consistent: an emergency category always implies
    # emergency urgency, and vice versa. The graph branches on urgency, so a
    # disagreement between them would be a silent routing bug.
    if category == "emergency" or urgency == "emergency":
        category, urgency = "emergency", "emergency"

    if category not in config.VALID_CATEGORIES:
        category, urgency = "emergency", "emergency"

    return {"category": category, "urgency": urgency, "reason": parsed.reason}
