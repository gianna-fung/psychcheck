"""
The Streamlit UI: upload a PDF, see its structured overview, then ask grounded
questions about it. This file only handles display and Streamlit session state —
all the actual PDF/LLM/vectorstore logic lives in pipeline.py, kept separate so the
pipeline can be tested (or reused in a script) without needing Streamlit running.
"""

from pathlib import Path

import streamlit as st

from contested_findings import load_contested_findings
from pipeline import (
    answer_question,
    build_vectorstore,
    generate_overview,
    get_llm,
    load_pdf_pages,
    split_into_chunks,
)

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"

st.set_page_config(page_title="PsychCheck", page_icon="🧠")
st.title("🧠 PsychCheck")
st.caption(
    "Upload a psychology research paper to get a structured overview, then ask "
    "grounded questions about it. Answers that touch a well-known contested or "
    "failed-replication finding get flagged automatically."
)


# --- Cached, expensive-to-create objects ---
# @st.cache_resource means these are built once per app process (not once per script
# re-run — Streamlit re-runs this whole file on every user interaction, like clicking
# a button). Without caching, every click would re-load the embedding model and
# re-create the Claude client, which is slow and pointless since neither changes.


@st.cache_resource
def load_findings():
    return load_contested_findings()


@st.cache_resource
def load_llm():
    return get_llm()


contested_findings = load_findings()
llm = load_llm()

# --- Session state ---
# Streamlit re-runs this file top to bottom on every interaction, so anything that
# needs to survive between interactions (the current paper's vectorstore, the
# overview, the chat history) has to live in st.session_state instead of a plain
# local variable.
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "overview" not in st.session_state:
    st.session_state.overview = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of AnswerResult-ish dicts, oldest first
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None


# --- Upload + process a PDF ---

uploaded_file = st.file_uploader("Upload a paper (PDF)", type="pdf")

if uploaded_file is not None and uploaded_file.name != st.session_state.current_filename:
    # A new file was uploaded (different from whatever we last processed) — run the
    # full pipeline once and cache the results in session_state so re-runs triggered
    # by later button clicks don't reprocess the PDF from scratch every time.
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = UPLOAD_DIR / uploaded_file.name
    pdf_path.write_bytes(uploaded_file.getvalue())

    with st.spinner("Reading and indexing the paper..."):
        pages = load_pdf_pages(pdf_path)
        chunks = split_into_chunks(pages)
        st.session_state.vectorstore = build_vectorstore(chunks)

    with st.spinner("Generating overview..."):
        st.session_state.overview = generate_overview(pages, llm)

    st.session_state.current_filename = uploaded_file.name
    st.session_state.chat_history = []  # new paper, so old Q&A no longer applies


# --- Show the overview ---

if st.session_state.overview is not None:
    overview = st.session_state.overview
    st.subheader("Overview")
    st.markdown(f"**Research question:** {overview.research_question}")
    st.markdown(f"**Method:** {overview.method}")
    st.markdown(f"**Sample size:** {overview.sample_size}")
    st.markdown(f"**Key finding:** {overview.key_finding}")
    st.markdown(f"**Limitations:** {overview.limitations}")

    st.divider()
    st.subheader("Ask a question")

    question = st.text_input("Your question about this paper", key="question_input")
    ask_clicked = st.button("Ask")

    if ask_clicked and question:
        with st.spinner("Thinking..."):
            result = answer_question(
                st.session_state.vectorstore, llm, question, contested_findings
            )
        st.session_state.chat_history.append({"question": question, "result": result})

    # Newest question first, so the answer you just asked for is right under the
    # input box instead of at the bottom of a growing list.
    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"**Q: {entry['question']}**")
        result = entry["result"]

        if result.flagged_findings:
            for finding in result.flagged_findings:
                st.warning(
                    f"⚠️ **This touches a contested finding: {finding.topic}**\n\n"
                    f"Original claim: {finding.original_claim}\n\n"
                    f"What later evidence found: {finding.evidence_summary}\n\n"
                    f"Source: {finding.original_citation}"
                    + (
                        "\n\nReplication evidence: "
                        + "; ".join(finding.replication_citations)
                        if finding.replication_citations
                        else ""
                    )
                )

        st.markdown(result.answer)

        with st.expander(f"Sources ({len(result.citations)} excerpts used)"):
            for citation in result.citations:
                st.markdown(f"**Page {citation.page}:** {citation.excerpt}...")

        st.divider()
else:
    st.info("Upload a PDF above to get started.")
