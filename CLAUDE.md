# Design log

Running notes on *why* PsychCheck is built the way it is, so the project can be explained or
rebuilt from scratch later. Newest decisions at the bottom.

## Overall architecture: single-pass RAG (no retry/correction loop)

PDF → chunks → embed → store → retrieve top-4 chunks for a question → Claude generates an
answer from just those chunks → the answer text is checked against a small hand-curated list of
contested/failed-replication findings → flag if matched.

**Why no retry loop:** a "real" production RAG system might re-retrieve if the answer looks
ungrounded, or ask a second LLM call to grade its own answer. That adds real complexity (state
machines, more API calls, more failure modes to debug) for a portfolio project whose actual goal
is to demonstrate retrieval-augmented generation + a contested-findings safety check clearly.
Single-pass keeps every step traceable: you can point at exactly which chunks produced which
answer. This is a deliberate scope cut, not a missing feature.

## Model choice: Claude Sonnet 5 (`claude-sonnet-5`)

Anthropic's current lineup also has Opus 5 (higher quality, ~2.5x the price) and Haiku 4.5
(cheapest, weakest at nuanced reasoning). Sonnet 5 is the middle tier: strong enough for
structured extraction and grounded Q&A, and meaningfully cheaper than Opus — which matters
because a project like this gets called dozens of times per hour during development (every
PDF upload triggers an overview call; every typed question triggers a Q&A call). Model ID is
set in one place (`src/pipeline.py`) so it's a one-line change if you want to compare Opus or
Haiku later.

## Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, run locally

This model runs on your own machine via `sentence-transformers` — no API key, no per-call cost,
and no network dependency once it's downloaded. It's small (~80MB) and fast, and "good enough"
semantic similarity is all retrieval needs here (it doesn't have to write text, just rank chunks
by relevance to a question). The tradeoff: it's a smaller/older embedding model than something
like Voyage or OpenAI's embeddings, so retrieval quality is a notch below a hosted embedding API
— acceptable for a portfolio project, worth knowing if you ever see a retrieval miss.

## Vector store: Chroma, in-memory, rebuilt per upload

Chroma is used via `langchain-chroma`, with no `persist_directory` — each uploaded PDF gets a
fresh, ephemeral vectorstore in the Streamlit session. Papers aren't meant to accumulate across
sessions (this isn't a multi-document research library), and skipping persistence avoids having
to manage a growing on-disk index or clean up old collections. Chroma was chosen over
alternatives (FAISS, Pinecone, Weaviate) because it needs no separate server process and has the
most direct LangChain integration for a local prototype — you `pip install` it and it works.

## Contested-findings matching: keyword/phrase matching, not a second LLM call

`src/contested_findings.py` checks whether a generated answer's text contains keywords tied to
one of the 10 hand-curated contested findings in `data/contested_findings.json`, rather than
asking Claude "does this answer touch a contested finding?" in a second API call.

**Why:** it's deterministic and free — the same answer always gets the same flag decision, and
it costs no extra tokens. You can read the exact keyword list that triggered a flag, which is
easier to debug and explain than an LLM's judgment call.

**Tradeoff, stated plainly:** keyword matching will miss a paraphrase that discusses "the
2010 study on posture and hormones" without ever saying "power posing." It will also occasionally
false-positive on an unrelated mention of a keyword. This is the accuracy/complexity tradeoff of
the single-pass design — a second LLM-based classification call would catch more paraphrases at
the cost of another API call and another thing that can silently fail. Documented here so it's a
known limitation, not a bug someone finds later.

## The 10 contested findings

Chosen because each is a *real, published, citable* controversy (see `data/contested_findings.json`
for the exact citations) — not a subjective judgment call. The set intentionally spans different
*kinds* of contestedness, since "contested" isn't one thing:

- **Failed direct replication** (a later study tried to reproduce the exact effect and couldn't):
  Bem's psi effect, power posing, ego depletion, the facial feedback hypothesis, social priming.
- **Methodological critique** (the original study's methods/procedure were later shown to be
  flawed, e.g. the researcher influencing participants): the Stanford Prison Experiment, the
  Schnall handwashing study.
- **Reanalysis showing a much smaller effect than originally reported** (not "failed," but the
  popular-press version overstates it): the marshmallow test, growth mindset effect sizes.
- **Field-wide finding, not one study**: the Open Science Collaboration (2015) reproducibility
  project, which is the citation *for* the idea that psychology has a replication problem at all.

Every entry's citation was checked against what I could verify at write time; none are
fabricated. If you use this project for anything beyond a portfolio piece, double-check each
citation yourself before relying on it — see the note in `data/contested_findings.json`.

## Chunk size: 1000 characters, 200 overlap (20%)

~1000 characters is roughly a paragraph — big enough that a chunk usually contains one complete
idea (so it's useful on its own when retrieved in isolation), small enough that retrieval stays
precise (a bigger chunk risks mixing several unrelated ideas together, so a question about one
of them pulls in noise from the others). The 200-character overlap exists so a sentence sitting
right on a chunk boundary doesn't get half-cut in both neighboring chunks — with overlap, it
appears whole in at least one of them. Both are set as constants at the top of `src/pipeline.py`
rather than tuned per-paper; if retrieval quality turns out to matter for grading, `eval/` is
where that would get measured before touching these numbers.

## Overview generation reads the whole paper; Q&A retrieves top-4 chunks

These two features use different strategies on purpose, and it's not an inconsistency:

- **The overview** (`generate_overview` in `src/pipeline.py`) needs fields that are scattered
  across the entire paper — research question in the intro, sample size in the methods,
  limitations usually right before the conclusion. A top-4-chunks retrieval built around one
  question wouldn't have a "question" to retrieve against for a whole-paper summary, and would
  likely miss most of these fields. So it reads the (size-capped) full extracted text instead.
  This only happens once per upload, so the extra tokens are worth it.
- **Q&A** (`answer_question`) retrieves the 4 chunks most relevant to the *specific* question
  asked, because that happens on every question — reading the whole paper on every question
  would be slower and more expensive for no accuracy benefit once a question is narrow enough
  that 4 relevant chunks contain the answer.

The overview's full text is capped at `MAX_OVERVIEW_CHARS` (200,000 characters) so one unusually
long PDF can't cause an unbounded-cost API call — most journal articles are far under this.

## Structured output via a pydantic model, not free-text parsing

`PaperOverview` in `src/pipeline.py` is a pydantic `BaseModel` passed to
`llm.with_structured_output(...)`, instead of asking Claude for the five overview fields in a
paragraph and then trying to regex or string-split them back apart. This means Claude's response
comes back as real, validated Python attributes (`overview.method`, `overview.sample_size`, ...)
that can't silently be missing a field or be formatted inconsistently between runs — the
structured-output feature enforces the shape.

## `temperature=0` for both overview and Q&A

Both tasks this app uses Claude for are meant to be faithful to the source text — extracting
what a paper says, and answering a question from retrieved excerpts — not creative writing.
`temperature=0` makes the model's output as deterministic as an LLM's output can be: the same
paper text should produce close to the same overview if you run it twice.

## Streamlit state: `st.session_state` + `@st.cache_resource`

Streamlit re-runs the entire `src/app.py` script top-to-bottom on every user interaction (every
button click, every text input). Two consequences shaped `app.py`:

- Anything that must survive between interactions — the current paper's vectorstore, its
  overview, and the running chat history — is stored in `st.session_state` instead of a plain
  local variable, which would reset to empty on every re-run.
- Anything expensive to build that doesn't change per-interaction — the embedding model and the
  Claude client — is wrapped in `@st.cache_resource`, so it's created once per app process
  instead of once per click.

A new file upload only reprocesses the PDF (re-chunk, re-embed, re-generate the overview) when
the uploaded filename differs from the one already processed — clicking "Ask" on a question
doesn't accidentally re-run the whole pipeline, since Streamlit's rerun-on-every-interaction
model would otherwise make that easy to trigger by mistake.

---

*(This file grows as the project grows.)*
