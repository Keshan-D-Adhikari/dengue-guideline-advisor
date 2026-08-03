"""
Streamlit entrypoint.

Renders the chat, calls the compiled agent graph, and shows the agent trace
in the sidebar. The trace is not decoration — it is the visible evidence
that five distinct agents ran, which path the router chose, and whether the
verifier approved the answer.

The disclaimer banner is rendered here, unconditionally, before any user
input is read. It is not something a model can decline to include.
"""

from __future__ import annotations

import streamlit as st

from src import config
from src.providers import ProviderError

st.set_page_config(page_title="Dengue Guideline Advisor", page_icon="🩺")

st.title("Dengue Guideline Advisor")
st.info(config.DISCLAIMER)


# --- Sidebar ---------------------------------------------------------------

def render_trace(trace: list) -> None:
    """Show which agents fired, in order, with their key outputs."""
    if not trace:
        st.caption("No run yet.")
        return

    for step, entry in enumerate(trace, start=1):
        with st.expander(f"{step}. {entry['agent']} — {entry['summary']}", expanded=False):
            detail = entry.get("detail") or {}
            if not detail:
                st.caption("no additional detail")
            for key, value in detail.items():
                if isinstance(value, list):
                    st.markdown(f"**{key}**")
                    if value:
                        for item in value:
                            st.markdown(f"- {item}")
                    else:
                        st.caption("(none)")
                else:
                    st.markdown(f"**{key}**: {value}")


def render_corpus_status() -> None:
    """Warn early if ingestion hasn't run — otherwise every answer is 'not found'."""
    from src.rag.vectorstore import collection_count

    try:
        count = collection_count()
    except Exception as exc:
        st.error(f"Vector store unavailable: {exc}")
        return

    if count == 0:
        st.warning(
            "No documents indexed. Put the guideline PDFs in "
            f"`{config.RAW_CORPUS_DIR}/` and run `python -m src.rag.ingest`."
        )
    else:
        st.success(f"{count} chunks indexed")


with st.sidebar:
    st.header("Corpus")
    render_corpus_status()

    st.header("Models")
    st.caption(f"**router** · {config.ROUTER_MODEL}")
    st.caption(f"**planner / rerank** · {config.RERANK_MODEL}")
    st.caption(f"**synthesis** · {config.SYNTHESIS_MODEL}")
    st.caption(f"**verifier** · {config.VERIFIER_MODEL}")

    st.header("Agent trace")
    st.caption("Last run")
    render_trace(st.session_state.get("last_trace", []))


# --- Chat ------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(source)


def format_sources(chunks: list) -> list[str]:
    """One markdown line per retrieved chunk, numbered to match the [S#] markers."""
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        location = chunk.get("source_doc", "unknown")
        if chunk.get("section"):
            location += f" § {chunk['section']}"
        if chunk.get("page"):
            location += f" (p.{chunk['page']})"
        snippet = " ".join(str(chunk.get("text", "")).split())[:280]
        lines.append(f"**[S{index}]** `{location}` — relevance {chunk.get('score', 0):.2f}\n\n> {snippet}…")
    return lines


prompt = st.chat_input("Ask about dengue symptoms, prevention, or reporting…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Routing, retrieving, drafting, verifying…"):
                from src.graph.build_graph import run

                result = run(prompt)
        except ProviderError as exc:
            # Missing key or provider outage — actionable, so say what to do.
            st.error(f"Model provider unavailable.\n\n{exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ Model provider unavailable: {exc}"}
            )
        except Exception as exc:
            st.error(
                "Something went wrong while answering. Nothing has been saved.\n\n"
                f"`{type(exc).__name__}: {exc}`"
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ Error: {type(exc).__name__}"}
            )
        else:
            answer = result.get("final_answer", "")
            sources = format_sources(result.get("chunks", []))

            st.markdown(answer)

            badge = "✅ verified" if result.get("approved") else "⚠️ unverified"
            st.caption(
                f"{badge} · {result.get('category', '?')} · "
                f"{len(result.get('chunks', []))} source passage(s)"
            )

            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(source)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            st.session_state["last_trace"] = result.get("trace", [])
            st.rerun()
