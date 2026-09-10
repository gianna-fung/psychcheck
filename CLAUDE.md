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

---

*(This file grows as the project grows — the next entries will cover `src/pipeline.py`'s chunk
size/overlap choice and `src/app.py`'s UI decisions.)*
