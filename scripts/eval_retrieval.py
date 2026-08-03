"""
Retrieval evaluation harness — the five-query evaluation the README reports.

    python -m scripts.eval_retrieval

Retrieval is evaluated on its own, before any generation, because a
generation-level score cannot tell you whether a wrong answer came from bad
retrieval or bad synthesis. This measures only the first half.

Metrics, per query:
  hit@k      — did any retrieved chunk contain an expected keyword
  precision  — fraction of retrieved chunks that contained one
  MRR        — 1/rank of the first relevant chunk (0 if none)

Ground truth here is keyword-based rather than a hand-labelled chunk id,
because chunk ids change every time the chunker is re-tuned. Keywords
survive re-chunking, which makes the metric comparable across runs. It is a
weaker signal than human relevance labels — that limitation is recorded in
the README.

EVAL_QUERIES is the part you must review against your own corpus: the
expected keywords have to actually appear in the guideline PDFs you ingested.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src import config
from src.rag import vectorstore

EVAL_QUERIES: list[dict] = [
    {
        "id": 1,
        "query": "What are the warning signs of severe dengue?",
        "expect_any": [
            "abdominal pain",
            "persistent vomiting",
            "mucosal bleeding",
            "lethargy",
            "warning sign",
        ],
    },
    {
        "id": 2,
        "query": "How much fluid should a dengue patient drink at home?",
        "expect_any": ["oral fluid", "fluid intake", "rehydration", "ors", "drink"],
    },
    {
        "id": 3,
        "query": "How do I stop mosquitoes breeding around my house?",
        "expect_any": ["breeding site", "standing water", "container", "larval", "vector control"],
    },
    {
        "id": 4,
        "query": "When must a dengue case be notified to the health authorities?",
        "expect_any": ["notifi", "report", "surveillance", "notifiable"],
    },
    {
        "id": 5,
        "query": "Which criteria mean a dengue patient must be admitted to hospital?",
        "expect_any": ["admission", "referral", "hospitali", "group c", "severe dengue"],
    },
]


def score_query(case: dict, k: int) -> dict:
    chunks = vectorstore.similarity_search(case["query"], k=k)
    keywords = [keyword.lower() for keyword in case["expect_any"]]

    relevance = [
        any(keyword in chunk["text"].lower() for keyword in keywords) for chunk in chunks
    ]
    first_hit = next((index for index, hit in enumerate(relevance) if hit), None)

    return {
        "id": case["id"],
        "query": case["query"],
        "retrieved": len(chunks),
        "hit": first_hit is not None,
        "precision": round(sum(relevance) / len(relevance), 3) if relevance else 0.0,
        "mrr": round(1 / (first_hit + 1), 3) if first_hit is not None else 0.0,
        "top_source": chunks[0]["source_doc"] if chunks else None,
        "top_section": chunks[0]["section"] if chunks else None,
        "top_score": chunks[0]["score"] if chunks else None,
    }


def main() -> int:
    k = config.TOP_K_RETRIEVED
    indexed = vectorstore.collection_count()

    if indexed == 0:
        print(
            f"Collection is empty. Put the guideline PDFs in "
            f"{config.RAW_CORPUS_DIR}/ and run `python -m src.rag.ingest` first."
        )
        return 1

    print(f"Corpus: {indexed} chunks · embedding: {config.EMBEDDING_MODEL} · k={k}\n")

    results = [score_query(case, k) for case in EVAL_QUERIES]

    header = f"{'#':<3}{'hit':<6}{'prec':<7}{'MRR':<7}query"
    print(header)
    print("-" * (len(header) + 30))
    for result in results:
        print(
            f"{result['id']:<3}{'yes' if result['hit'] else 'NO':<6}"
            f"{result['precision']:<7}{result['mrr']:<7}{result['query'][:52]}"
        )

    count = len(results)
    print(
        f"\nhit@{k}: {sum(result['hit'] for result in results)}/{count}"
        f" · mean precision: {sum(r['precision'] for r in results) / count:.3f}"
        f" · mean MRR: {sum(r['mrr'] for r in results) / count:.3f}"
    )

    output = Path("data/processed/retrieval_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nPer-query detail written to {output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
