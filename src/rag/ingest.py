"""
One-off ingestion script: reads PDFs from data/raw/, chunks them,
embeds each chunk, and writes them into the Chroma collection.

Run this manually whenever the corpus changes:
    python -m src.rag.ingest

TODO: implement
  1. Walk config.RAW_CORPUS_DIR for PDFs (pypdf to extract text).
  2. Chunk each document using config.CHUNK_SIZE_TOKENS /
     CHUNK_OVERLAP_TOKENS. Prefer splitting on the document's own
     headings/sections where possible instead of a blind sliding window,
     so a chunk doesn't cut a warning-signs list in half.
  3. Attach metadata to every chunk: source_doc (filename), section
     (heading if you can detect it), chunk_index.
  4. Embed each chunk (sentence-transformers or another model — record
     your choice and why in the README) and upsert into the collection
     from src/rag/vectorstore.py.
  5. Print a summary: number of docs ingested, number of chunks created.
"""

if __name__ == "__main__":
    raise NotImplementedError
