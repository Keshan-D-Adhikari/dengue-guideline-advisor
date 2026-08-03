"""
Central configuration: model choices per sub-task, and shared constants.

Every model ID can be overridden with an environment variable of the same
name, so the deployed app can be re-pointed without a code change:

    set SYNTHESIS_MODEL=anthropic/claude-opus-4.1

Model IDs move faster than this file does. Verify against the live lists
before a demo:
  - Groq models:       https://console.groq.com/docs/models
  - OpenRouter models: https://openrouter.ai/models

The rationale for each choice is in the README "Model selection strategy"
table; the one-line summary is in the comment above each constant.
"""

import os

# --- Model selection -------------------------------------------------------
# Two providers on purpose: Groq for the latency-critical mechanical steps,
# OpenRouter for the two steps where a wrong answer has clinical consequences.

# Groq. 4-way classification in JSON mode — smallest model that can do it,
# and it sits on the critical path before anything else can start.
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "llama-3.1-8b-instant")

# Groq. Relevance scoring needs a little more judgement than routing, but
# still runs once per query over short text, so it stays on the cheap tier.
RERANK_MODEL = os.getenv("RERANK_MODEL", "llama-3.3-70b-versatile")

# OpenRouter. Highest-stakes step: grounded generation under a strict
# "only what's in the chunks" instruction. Strongest instruction-following
# model in the stack.
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", "anthropic/claude-sonnet-4.5")

# OpenRouter. MUST be a different model — and here a different vendor —
# than SYNTHESIS_MODEL, so verification is a genuine second opinion and not
# a model grading its own homework.
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "google/gemini-2.5-pro")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Generation settings. Temperature 0 everywhere: this is a guideline lookup
# tool, so we want the same question to produce the same answer.
TEMPERATURE = 0.0
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 2

# --- RAG constants ---------------------------------------------------------
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
TOP_K_RETRIEVED = 8

VECTOR_STORE_DIR = "chroma_db"
RAW_CORPUS_DIR = "data/raw"
COLLECTION_NAME = "dengue_guidelines"

# Local sentence-transformers model. No API cost and no per-query network
# hop. 384-dim, CPU-friendly. The SAME model must be used at ingestion and
# at query time — src/rag/vectorstore.py is the only place it is loaded.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Tokeniser used only to measure chunk length. Any consistent tokeniser
# works; cl100k_base is a reasonable proxy for the models above.
TOKENIZER_ENCODING = "cl100k_base"

# --- Safety ----------------------------------------------------------------
# Shown verbatim to the user on every answer. Enforced in app.py, not left
# to a prompt.
DISCLAIMER = (
    "Decision support only, not a diagnosis. For urgent symptoms, contact "
    "a health professional or emergency services immediately."
)

# Returned as-is when the router classifies a query as an emergency. No LLM
# generation happens on this path, so it cannot be hallucinated.
EMERGENCY_MESSAGE = (
    "**This may be a medical emergency. Seek care now.**\n\n"
    "Go to the nearest hospital emergency department, or call your local "
    "emergency number immediately. Do not wait for this tool's advice.\n\n"
    "Dengue warning signs that need immediate assessment include severe "
    "abdominal pain, persistent vomiting, bleeding from the gums or nose, "
    "blood in vomit or stool, difficulty breathing, cold or clammy skin, "
    "restlessness or drowsiness, and passing little or no urine.\n\n"
    "If you are caring for someone who is collapsing, bleeding heavily, or "
    "unresponsive, treat it as an emergency and get help now."
)

# Official red-flag list, kept as structured data rather than prose so the
# retrieval agent can check it deterministically instead of relying on an
# LLM to paraphrase it correctly. Source: WHO dengue guidelines, warning
# signs for severe dengue.
WARNING_SIGNS = [
    "severe abdominal pain",
    "persistent vomiting",
    "clinical fluid accumulation",
    "mucosal bleeding",
    "bleeding gums",
    "nosebleed",
    "blood in vomit",
    "blood in stool",
    "lethargy",
    "restlessness",
    "liver enlargement",
    "difficulty breathing",
    "cold clammy skin",
    "reduced urine output",
    "rapid weak pulse",
]

# Answers in these categories must carry an explicit "seek medical
# attention" instruction. Enforced in src/agents/verifier.py in Python.
CATEGORIES_REQUIRING_ESCALATION_ADVICE = {"symptom_check", "emergency"}

VALID_CATEGORIES = {
    "symptom_check",
    "prevention_advice",
    "reporting_procedure",
    "emergency",
}
