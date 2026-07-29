"""
Thin wrapper around the vector store (Chroma) so the rest of the app
doesn't need to know the client details.

TODO: implement
  - get_client(): returns/creates a persistent Chroma client at
    config.VECTOR_STORE_DIR
  - get_or_create_collection(name): returns a collection handle
  - similarity_search(query: str, k: int) -> list[dict]: embeds the
    query (same embedding model used at ingestion time!) and returns
    the top-k chunks with their metadata and similarity score
"""


def get_client():
    raise NotImplementedError


def similarity_search(query: str, k: int) -> list:
    raise NotImplementedError
