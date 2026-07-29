"""
Central configuration: model choices per sub-task, and shared constants.

TODO (you write this): pick one model per sub-task and record WHY in the
README comparison table (cost / latency / context window / reasoning quality).

Look up current model IDs here:
  - Groq models:       https://console.groq.com/docs/models
  - OpenRouter models: https://openrouter.ai/models
"""

# TODO: fast/cheap model for intent routing + urgency classification (Groq)
ROUTER_MODEL = ""

# TODO: small/mid model for scoring & re-ranking retrieved chunks (Groq)
RERANK_MODEL = ""

# TODO: strong reasoning model for final answer synthesis (OpenRouter)
SYNTHESIS_MODEL = ""

# TODO: a DIFFERENT model than SYNTHESIS_MODEL for independent verification
VERIFIER_MODEL = ""

# RAG constants — tune these once you've looked at your actual corpus
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
TOP_K_RETRIEVED = 8

VECTOR_STORE_DIR = "chroma_db"
RAW_CORPUS_DIR = "data/raw"
