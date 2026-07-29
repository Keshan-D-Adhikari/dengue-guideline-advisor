"""
Router agent (pattern: Router).

Classifies an incoming user query so the orchestrator can pick a path.

Message in:  {"user_query": str}
Message out: {"category": "symptom_check" | "prevention_advice"
                        | "reporting_procedure" | "emergency",
              "urgency": "normal" | "emergency"}

TODO: implement using config.ROUTER_MODEL via the Groq client.
Keep the prompt short — this call must be fast and cheap. Consider a
structured-output / JSON-mode call so you get a reliable category back
instead of parsing free text.
"""


def route_query(user_query: str) -> dict:
    raise NotImplementedError
