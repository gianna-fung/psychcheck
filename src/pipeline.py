"""
The core RAG pipeline: PDF -> chunks -> vectorstore -> structured overview -> grounded Q&A.

Single-pass by design (see CLAUDE.md): each function below does one step and hands its
output to the next. There's no loop that re-retrieves or re-asks Claude to check its own
work — if you trace a bug, you can always point at exactly which function produced the
bad output.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# Same-directory import — works both when this file is run directly (`python
# src/pipeline.py`, which puts src/ on sys.path automatically) and when Streamlit
# runs src/app.py (Streamlit also adds the script's own directory to sys.path).
from contested_findings import ContestedFinding, check_for_contested_findings

# Reads ANTHROPIC_API_KEY out of .env so ChatAnthropic() can find it without the key
# being hardcoded anywhere. Safe to call this more than once (it's a no-op if the
# variables are already loaded).
load_dotenv()

# --- Tunable constants, gathered here instead of scattered through the functions ---
# so changing any one of them is a one-line edit. See CLAUDE.md for the reasoning
# behind each choice.

# claude-sonnet-5: strong enough for structured extraction and grounded Q&A, and
# meaningfully cheaper than Opus for a project you'll be calling constantly while
# developing and testing.
MODEL_NAME = "claude-sonnet-5"

# Runs locally via sentence-transformers — free, no API key, no network call per chunk.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ~1000 characters is roughly a paragraph: enough for a chunk to contain a complete
# idea (so it's useful on its own when retrieved), without being so large that
# retrieval becomes imprecise (mixing multiple unrelated ideas into one chunk).
CHUNK_SIZE = 1000

# 20% overlap so a sentence or idea that falls right on a chunk boundary still shows
# up whole in at least one chunk, instead of being split and losing meaning in both.
CHUNK_OVERLAP = 200

# How many chunks to retrieve per question. 4 is enough context for most questions
# about a single paper without diluting the prompt with irrelevant chunks (more
# retrieved chunks isn't free — each one is a chance to pull in something off-topic
# that nudges the answer in the wrong direction).
RETRIEVAL_K = 4

# Overview generation reads the whole paper (not just retrieved chunks — see
# generate_overview's docstring for why) but is capped at this many characters so a
# single unusually long PDF can't blow up cost or latency unbounded. ~200k characters
# is roughly 40-50k tokens, comfortably more than a typical journal article.
MAX_OVERVIEW_CHARS = 200_000


class PaperOverview(BaseModel):
    """The five fields we ask Claude to extract from the paper. Using a pydantic model
    with `.with_structured_output()` (rather than asking for free text and hoping it's
    formatted consistently) means we get these fields back as real Python attributes,
    already validated — no fragile string-parsing of Claude's response."""

    research_question: str = Field(
        description="The core research question or hypothesis the study set out to test."
    )
    method: str = Field(
        description="The study design and method in a sentence or two (e.g. randomized "
        "controlled experiment, longitudinal survey, between-subjects design)."
    )
    sample_size: str = Field(
        description="The number and type of participants (e.g. '120 undergraduates'). "
        "Write 'Not clearly reported' if the paper doesn't state this."
    )
    key_finding: str = Field(
        description="The main result the authors report, in plain language."
    )
    limitations: str = Field(
        description="Limitations the authors themselves acknowledge. Write 'Not clearly "
        "reported' if the paper has no explicit limitations section or discussion."
    )


@dataclass
class Citation:
    """One retrieved chunk that fed into an answer, for showing the user where an
    answer came from. `page` is 1-indexed for display (PyPDFLoader's own page numbers
    are 0-indexed, matching Python's usual indexing, so we add 1 here)."""

    page: int
    excerpt: str


@dataclass
class AnswerResult:
    """Everything the UI needs to render one answer: the text, what it was grounded in,
    and whether it touched a contested finding."""

    answer: str
    citations: list[Citation]
    flagged_findings: list[ContestedFinding]


def load_pdf_pages(pdf_path: str | Path) -> list[Document]:
    """Loads a PDF as one Document per page. Kept separate from chunking (below)
    because generate_overview() wants whole-page text, while the vectorstore wants
    smaller chunks — both start from these same pages."""
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_into_chunks(pages: list[Document]) -> list[Document]:
    """Splits page-level Documents into smaller chunks for retrieval. Metadata (like
    which page a chunk came from) carries over automatically, which is how citations
    in answer_question() know their page numbers."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(pages)


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Embeds every chunk and stores them in an in-memory Chroma collection. No
    `persist_directory` is passed — see CLAUDE.md's "Vector store" section for why a
    fresh, throwaway vectorstore per upload is the right call here.

    `collection_name` is set to a fresh UUID on every call -- without it, Chroma
    defaults every collection to the same name ("langchain"), so a second call to
    this function in the same process doesn't create an isolated store, it ADDS to
    whatever's already in that shared collection. Confirmed by direct reproduction:
    two unrelated documents ended up in the same collection, and querying the second
    one returned chunks from the first. A unique name per call is what actually makes
    "fresh vectorstore per upload" true, rather than just true for the first upload
    in a given process."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name=str(uuid.uuid4())
    )


def get_llm() -> ChatAnthropic:
    """Creates the Claude client. No `temperature` argument: Claude Sonnet 5 (and the
    other current-generation Claude models) removed the temperature/top_p/top_k
    sampling controls — the API returns a 400 if you pass them — because these models
    always reason adaptively instead. That actually fits what we want here (faithful
    extraction and grounded answers, not creative writing) even without an explicit
    temperature knob to turn down."""
    return ChatAnthropic(model=MODEL_NAME)


def generate_overview(pages: list[Document], llm: ChatAnthropic) -> PaperOverview:
    """Generates the structured overview from the WHOLE paper, not from retrieved
    chunks. This is different from answer_question() on purpose: an overview needs
    the research question (often in the intro), method (methods section), sample
    size (methods section), key finding (results), and limitations (usually right
    before the conclusion) — fields that are scattered across the entire paper. A
    top-4-chunks retrieval built for a specific question would likely miss most of
    them. Overview generation happens once per upload, so reading the whole paper is
    affordable; per-question retrieval (below) happens on every question, so it stays
    narrow and cheap."""
    full_text = "\n\n".join(
        f"[Page {page.metadata.get('page', 0) + 1}]\n{page.page_content}" for page in pages
    )
    full_text = full_text[:MAX_OVERVIEW_CHARS]

    structured_llm = llm.with_structured_output(PaperOverview)
    return structured_llm.invoke(
        [
            (
                "system",
                "You are extracting a structured overview from a psychology research "
                "paper for a student reading it. Base every field ONLY on the text "
                "given to you — do not use outside knowledge about the study, and do "
                "not guess at numbers you don't see. If a field genuinely isn't in the "
                "text, say so instead of inventing an answer.",
            ),
            ("human", full_text),
        ]
    )


def answer_question(
    vectorstore: Chroma,
    llm: ChatAnthropic,
    question: str,
    contested_findings: list[ContestedFinding],
) -> AnswerResult:
    """Answers one question about the paper using retrieval-augmented generation:
    retrieve the most relevant chunks, ask Claude to answer using only those chunks,
    then check the answer text against the contested-findings list. This is the one
    function where the "single-pass" architecture is most visible — everything
    happens in this one retrieve-then-generate-then-check sequence, no loop."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    retrieved_chunks = retriever.invoke(question)

    context = "\n\n".join(
        f"[Page {chunk.metadata.get('page', 0) + 1}]\n{chunk.page_content}"
        for chunk in retrieved_chunks
    )

    response = llm.invoke(
        [
            (
                "system",
                "You answer questions about a research paper using ONLY the excerpts "
                "provided below — not any outside knowledge you may have about this "
                "topic or study. Reference page numbers from the excerpts when you "
                "use them. If the excerpts don't contain enough information to answer "
                "the question, say so plainly instead of guessing.",
            ),
            ("human", f"Excerpts from the paper:\n\n{context}\n\nQuestion: {question}"),
        ]
    )
    # .text (not .content!): Claude's response can come back as a list of content
    # blocks (e.g. a thinking block plus a text block) rather than a plain string,
    # depending on how much the model reasoned before answering. .text always
    # returns just the visible text portion as a real string, regardless of how
    # many blocks came back -- .content would sometimes hand back a list here and
    # break check_for_contested_findings()'s .lower() call below.
    answer_text = response.text

    citations = [
        Citation(page=chunk.metadata.get("page", 0) + 1, excerpt=chunk.page_content[:200])
        for chunk in retrieved_chunks
    ]
    flagged = check_for_contested_findings(answer_text, contested_findings)

    return AnswerResult(answer=answer_text, citations=citations, flagged_findings=flagged)
