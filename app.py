"""
Streamlit entrypoint.

TODO: implement
  - Page config + a persistent, always-visible disclaimer banner:
    "Decision support only, not a diagnosis. For urgent symptoms,
    contact a health professional or emergency services immediately."
  - Chat input -> src.graph.build_graph.run(user_query)
  - Render the final answer + its source citations
  - Sidebar "agent trace": show which agents fired, in what order, and
    their key outputs (category, urgency, approved) — this doubles as
    your live demo of the multi-agent pipeline for the viva.
  - Wrap the graph call in a try/except so a model/API failure shows a
    graceful error in the UI instead of a stack trace (rubric: "handles
    errors gracefully").
"""

import streamlit as st

st.set_page_config(page_title="Dengue Guideline Advisor", page_icon="🩺")

st.title("Dengue Guideline Advisor")
st.info(
    "Decision support only, not a diagnosis. For urgent symptoms, contact "
    "a health professional or emergency services immediately."
)

# TODO: chat input, call the graph, render answer + citations + agent trace
