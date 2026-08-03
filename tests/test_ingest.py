"""Tests for heading-aware chunking — the part of ingestion that has no I/O."""

from src.rag.ingest import chunk_text, looks_like_heading, split_into_sections


def test_recognises_heading_shapes():
    assert looks_like_heading("3.2 Warning signs")
    assert looks_like_heading("WARNING SIGNS")
    assert looks_like_heading("Clinical Management Of Dengue")
    assert looks_like_heading("4) Case reporting")


def test_rejects_body_text():
    assert not looks_like_heading(
        "The patient should be monitored closely for signs of plasma leakage."
    )
    assert not looks_like_heading("")
    assert not looks_like_heading("   ")
    assert not looks_like_heading("a" * 200)


def test_splits_on_headings_and_keeps_preamble():
    document = (
        "Some front matter before any heading.\n"
        "1. Introduction\n"
        "Dengue is a mosquito-borne viral infection.\n"
        "WARNING SIGNS\n"
        "Severe abdominal pain.\n"
    )
    sections = split_into_sections(document)
    headings = [heading for heading, _ in sections]

    assert headings == ["", "1. Introduction", "WARNING SIGNS"]
    assert "front matter" in sections[0][1]
    assert "Severe abdominal pain" in sections[2][1]


def test_short_section_is_a_single_chunk():
    assert chunk_text("a short body") == ["a short body"]


def test_empty_section_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_long_section_is_windowed_with_overlap():
    chunks = chunk_text("word " * 1200, size=400, overlap=50)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_warning_sign_list_is_not_split_across_chunks():
    # The reason chunking is heading-aware: this list must survive intact.
    document = (
        "WARNING SIGNS\n"
        "Severe abdominal pain.\n"
        "Persistent vomiting.\n"
        "Mucosal bleeding.\n"
        "PREVENTION\n"
        "Remove standing water.\n"
    )
    sections = dict(split_into_sections(document))
    warning_body = sections["WARNING SIGNS"]

    assert chunk_text(warning_body) == [warning_body]
    assert "Remove standing water" not in warning_body
