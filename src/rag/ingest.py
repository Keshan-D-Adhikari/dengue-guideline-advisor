"""
One-off ingestion script: reads PDFs from data/raw/, chunks them, embeds
each chunk, and writes them into the Chroma collection.

Run this manually whenever the corpus changes:
    python -m src.rag.ingest            # add to the existing collection
    python -m src.rag.ingest --reset    # rebuild from scratch

Chunking is heading-aware rather than a blind sliding window. That matters
for this corpus specifically: the guideline documents present warning signs
and referral criteria as lists, and a window that cuts such a list in half
produces a chunk that reads like a complete answer while silently omitting
half the red flags. Sections are detected first, and the sliding window is
applied within a section, never across one.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterator

from src import config
from src.rag import vectorstore

# A line that looks like a heading: numbered ("3.2 Warning signs"), or a
# short title-ish line with no terminal punctuation ("WARNING SIGNS").
_NUMBERED_HEADING = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+\S")
_MAX_HEADING_CHARS = 90


def looks_like_heading(line: str) -> bool:
    """Heuristic heading test, kept pure so tests can pin the behaviour."""
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_CHARS:
        return False
    if stripped.endswith((".", ",", ";", ":")) and not _NUMBERED_HEADING.match(stripped):
        return False

    letters = [char for char in stripped if char.isalpha()]
    if len(letters) < 3:
        return False

    if _NUMBERED_HEADING.match(stripped):
        return True

    # ALL CAPS or Title Case line, e.g. "Warning Signs" / "WARNING SIGNS"
    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    if uppercase_ratio > 0.8:
        return True

    words = stripped.split()
    if 1 < len(words) <= 8 and all(word[0].isupper() for word in words if word[0].isalpha()):
        return True

    return False


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split document text into (heading, body) pairs.

    Text before the first detected heading is returned under an empty
    heading so nothing is dropped.
    """
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []

    for line in text.splitlines():
        if looks_like_heading(line):
            if buffer and any(entry.strip() for entry in buffer):
                sections.append((heading, "\n".join(buffer).strip()))
            heading = line.strip()
            buffer = []
        else:
            buffer.append(line)

    if buffer and any(entry.strip() for entry in buffer):
        sections.append((heading, "\n".join(buffer).strip()))

    return sections


def _get_encoder():
    import tiktoken

    return tiktoken.get_encoding(config.TOKENIZER_ENCODING)


def chunk_text(
    text: str,
    size: int = config.CHUNK_SIZE_TOKENS,
    overlap: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Sliding-window token chunks over a single section's body."""
    if not text.strip():
        return []

    encoder = _get_encoder()
    tokens = encoder.encode(text)
    if len(tokens) <= size:
        return [text.strip()]

    step = max(1, size - overlap)
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]
        if not window:
            break
        chunks.append(encoder.decode(window).strip())
        if start + size >= len(tokens):
            break
    return [chunk for chunk in chunks if chunk]


def extract_pdf_pages(path: Path) -> list[str]:
    """Per-page text so a chunk can carry a page number for citation."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def iter_document_chunks(path: Path) -> Iterator[dict]:
    """Yield {text, metadata} for every chunk of one PDF."""
    chunk_index = 0
    for page_number, page_text in enumerate(extract_pdf_pages(path), start=1):
        if not page_text.strip():
            continue
        for heading, body in split_into_sections(page_text):
            for piece in chunk_text(body):
                yield {
                    "text": piece,
                    "metadata": {
                        "source_doc": path.name,
                        "section": heading,
                        "chunk_index": chunk_index,
                        "page": page_number,
                    },
                }
                chunk_index += 1


def ingest(corpus_dir: str | None = None, reset: bool = False) -> dict:
    """Ingest every PDF under `corpus_dir`. Returns a summary dict."""
    directory = Path(corpus_dir or config.RAW_CORPUS_DIR)
    pdfs = sorted(directory.glob("*.pdf")) if directory.is_dir() else []

    if not pdfs:
        return {"docs": 0, "chunks": 0, "error": f"No PDFs found in {directory}/"}

    collection = (
        vectorstore.reset_collection() if reset else vectorstore.get_or_create_collection()
    )

    total_chunks = 0
    for path in pdfs:
        texts: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for chunk in iter_document_chunks(path):
            texts.append(chunk["text"])
            metadatas.append(chunk["metadata"])
            ids.append(f"{path.name}::{chunk['metadata']['chunk_index']}")

        if not texts:
            print(f"  {path.name}: no extractable text (scanned PDF?) — skipped")
            continue

        # Batch the upsert; embedding all of a large guideline at once can
        # exhaust memory on the Streamlit Cloud tier.
        for start in range(0, len(texts), 128):
            end = start + 128
            collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
                embeddings=vectorstore.embed(texts[start:end]),
            )

        total_chunks += len(texts)
        print(f"  {path.name}: {len(texts)} chunks")

    return {"docs": len(pdfs), "chunks": total_chunks, "indexed": collection.count()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest dengue guideline PDFs into Chroma.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="drop the existing collection before ingesting",
    )
    parser.add_argument("--corpus", default=None, help="override the corpus directory")
    args = parser.parse_args()

    print(f"Embedding model: {config.EMBEDDING_MODEL}")
    print(f"Corpus: {args.corpus or config.RAW_CORPUS_DIR}")

    summary = ingest(args.corpus, reset=args.reset)

    if summary.get("error"):
        print(f"\n{summary['error']}")
        print("Add the official dengue guideline PDFs and re-run.")
        return 1

    print(
        f"\nIngested {summary['docs']} document(s), created {summary['chunks']} chunk(s). "
        f"Collection now holds {summary['indexed']} chunk(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
