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

## No `temperature` setting — it's not available on this model

The first version of `get_llm()` passed `temperature=0`, on the reasoning that both tasks this
app uses Claude for (extracting what a paper says, answering from retrieved excerpts) should be
faithful to the source text, not creative — so turn down the randomness. That broke immediately
against the real API: **`claude-sonnet-5` returns a 400 if you pass `temperature`, `top_p`, or
`top_k` at all.** Anthropic removed the sampling-control parameters for this model generation —
these models reason adaptively by default instead, and there's no manual dial to turn down.
Found this the first time the app actually ran a live API call, which is exactly the kind of
thing that's invisible until you test against the real service instead of just reading the
request code. `get_llm()` now passes only `model=MODEL_NAME`.

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

## `eval/`: a hand-written question set, graded by a second Claude call

`eval/questions.json` has 12 questions against a real paper (Mischel, Ebbesen, & Zeiss, 1972,
kept at `eval/fixtures/` so it's git-tracked and reproducible, unlike `data/uploads/` which is
intentionally gitignored) — 6 factual, 3 that should trigger a contested-findings flag, 3 that
shouldn't. `eval/run_eval.py` runs all 12 through the real pipeline and grades them.

Flag-correctness grading needs no extra API call: whether the right finding (or nothing) got
flagged is just checking a list the app already returned. Factual-correctness grading does use a
second Claude call — comparing "children waited much longer when distracted" against "distraction
significantly increased delay times" as the same fact requires semantic judgment a plain string
match can't make. This is the same second-LLM-call tradeoff documented above for
contested-findings matching, but it's the right call *inside an eval*, even though the shipped app
avoids it: `eval/run_eval.py` isn't part of what ships or what a user waits on, so paying for a
more careful judgment there doesn't cost the product anything.

## Bug found by the eval: `response.content` can be a list, not a string

The first version of `answer_question` set `answer_text = response.content` and passed it straight
to `check_for_contested_findings`, which calls `.lower()` on it. That's a safe assumption *most*
of the time — but running `eval/run_eval.py` hit `AttributeError: 'list' object has no attribute
'lower'` on the very first question. Claude's response can come back as a list of content blocks
(e.g. a thinking block plus a text block) instead of a plain string, depending on how much the
model reasoned before answering — something a few quick manual tests in the browser hadn't
happened to trigger, but a 12-question eval run did. Fixed by using `response.text` instead of
`response.content` — a LangChain accessor that always returns just the visible text as a real
string, no matter how many content blocks came back. This is exactly why the eval exists: it ran
enough real questions to surface a bug that manual spot-checking hadn't.

## Eval iteration: 58% → 75%, and why we stopped there

The first real eval run scored 7/12 (58%). Digging into *why* each question failed split them into
three different categories, and only one of the four failures was something worth changing:

- **One bad ground-truth question** (`f1`): asked "how many children participated in this study"
  with no experiment specified, but this paper reports three separate experiments with three
  different samples. The app correctly refused to conflate them; the question was ambiguous, not
  the answer wrong. Took two more attempts to get right — the first reword mislabeled which
  experiment the sample actually belonged to (fixed by grepping the primary source text directly
  instead of trusting an earlier assumption), and even after that fix, a later run showed the
  underlying issue was actually retrieval, not the question — see below.
- **Two overly strict grading criteria** (`f3`, `f5`): the expected facts required exact
  theoretical phrasing pulled from the paper's *abstract* (e.g. the term "frustrative nonreward
  theory"), but the app's answers were grounded in the more detailed *results* section instead
  (which is what top-4 retrieval actually returned) — correct, more specific answers marked wrong
  for not using my exact words. Loosened to the substantive claim; both now pass cleanly.
- **One real false negative** (`flag3`): the answer said "delay gratification" (no "of"), and the
  keyword matcher only recognized the exact phrase "delay of gratification". Fixed by adding
  "delay gratification" and "delaying gratification" as additional keywords in
  `data/contested_findings.json`.

After all three fixes: 9/12 (75%). The 3 remaining failures are now each a distinct, understood,
*reproducible* limitation rather than something to keep patching:

- **`flag3` still fails** on a later run, with an answer that used neither "delay...gratification"
  in any form nor any other listed keyword — just "self-control." No finite keyword list catches
  every paraphrase; this is the exact tradeoff documented above under "Contested-findings
  matching," now demonstrated twice.
- **`f1` and `f4` both fail on a genuine retrieval gap**, not a wrong answer. This paper has three
  separate "Subjects" sections and multiple similarly-structured multi-condition results, and
  top-4 retrieval doesn't reliably surface the *specific* one a narrow question asks about — the
  model's answers in both cases show real, careful reasoning about what it *was* given (in `f1`'s
  case, correctly inferring which experiment two other retrieved sections belonged to) rather than
  fabricating the missing piece. That's the `RETRIEVAL_K = 4` tradeoff from the chunking section
  above, made concrete.

Stopped iterating at 75% rather than continuing to tune the eval toward 100% — every remaining
failure is now a named, reproducible, honestly-explained limitation of the architecture, not a bug.
Chasing a higher number from here would mean either overfitting these specific 12 questions to this
one paper's quirks, or inflating `RETRIEVAL_K` for no reason beyond making one eval pass. A 75%
with three explained failure categories is a more credible number than a suspiciously clean 100%
would be after three rounds of tuning.

## Second eval set found a real production bug, not just an eval quirk

Added `eval/questions_social_priming.json` — 8 more questions against a second paper (Doyen,
Klein, Pichon, & Cleeremans, 2012, a failed replication of Bargh et al.'s "elderly words make you
walk slower" study), to check the app against a different paper and a different contested topic
than the marshmallow one. Unlike the first paper, this one is CC-BY licensed (PLOS ONE), so
`eval/fixtures/doyen_klein_pichon_cleeremans_2012.pdf` is safely committed, not gitignored.

`eval/run_eval.py` was generalized to discover and run every `eval/questions*.json` file
automatically — adding a new paper to compare against is just dropping in a new file, no changes
needed to the runner.

**The first run scored 3/8, and one answer was visibly wrong in an alarming way**: asked about
Experiment 2's experimenter conditions, the app's answer said it found content from "two different
papers" — one about delayed gratification in children (the *other* eval's paper), mixed into an
answer that should only have seen the priming paper. That's not a subtle grading issue, that's the
wrong paper's content leaking into an answer.

Traced it to `build_vectorstore()` in `src/pipeline.py`: `Chroma.from_documents(...)` was called
with no `collection_name`, so every call defaulted to the same collection name Chroma uses when
none is given (`"langchain"`). Confirmed by direct reproduction — building two vectorstores from
two unrelated one-sentence documents in the same Python process, then querying the *second* one,
returned content from the *first* one too. **This is a real bug in the shipped app, not just an
eval artifact**: in a single long-running Streamlit server process, every new paper a user uploads
would silently add its chunks to the same collection as every paper uploaded before it in that
session, instead of getting a genuinely isolated vectorstore — meaning retrieval for a later
upload could pull in chunks from an earlier, completely unrelated paper. `eval/run_eval.py`
running two papers back-to-back in one process is exactly the condition that exposes this; asking
one question about one paper right after upload (the only thing manually tested in the browser so
far) never would have. Fixed by generating a fresh UUID as `collection_name` on every call — this
is the same "the eval ran enough real cases to surface something manual testing hadn't" lesson as
the `response.content`-is-a-list bug above, at a higher stakes level this time.

**After the fix, the social-priming score didn't move (still 3/8)** — worth stating plainly,
because it proves contamination wasn't the cause of most of those failures, and it would have been
easy to declare victory on the fix and not check. Inspecting each answer individually: `sp_f4`'s
answer was now completely clean (no more mixed papers) and factually excellent, but still failed
grading for omitting one secondary detail ("5 participants per experimenter") that wasn't really
what the question asked about — the exact same over-strict-grading mistake already fixed once in
`questions.json`, repeated here because this eval set was written faster. `sp_f3` had the same
issue. `sp_flag1` and `sp_flag2` both failed because their (accurate, well-reasoned) answers cited
"Bargh et al." — completely normal academic shorthand — while the keyword list only recognized the
full string "bargh chen burrows". Fixed the two grading criteria (loosened to substance) and added
"bargh" and "behavioral priming" as keywords for `social_priming_elderly_walking`.

**That keyword fix immediately created a new, different failure**, and this is worth keeping
rather than chasing away: `sp_noflag1` ("how far apart were the two infrared sensors") started
failing because its answer mentioned "matching the distance used in the original Bargh et al.
study" — a purely incidental citation, now enough to trip the broadened keyword. This is the exact
false-positive half of the tradeoff the false-negative fix was always going to risk; "bargh" is a
distinctive enough surname that keeping it is still a net improvement (fixed two real misses,
costs one narrow false positive), but the tradeoff itself doesn't disappear just because one
instance of it was fixed. Final combined result across both eval sets: **15/20 (75%)**, with every
remaining failure now a named example of one of four categories: two retrieval-completeness gaps
(`f1`/`f4`, `sp_f2`), one keyword false negative (`flag3`), and one keyword false positive
(`sp_noflag1`) — both directions of the same documented tradeoff, each demonstrated concretely
rather than just described in the abstract.

## Note: this repo's git history was rewritten on 2026-09-23

Two things got fixed after the fact, before this repo was ever made public:

1. **Commit author identity.** This machine's global git config had an unrelated name/email
   (leftover from something else set up on it previously) that every commit in this repo had been
   silently inheriting, since nothing here had ever overridden it locally. Fixed going forward by
   setting the identity at the repo level instead of relying on the global default, and rewrote
   the 6 existing commits to match.
2. **A copyrighted PDF.** `eval/fixtures/` briefly held a full copy of the 1972 journal article
   the eval set is written against, added for reproducibility without thinking through that
   redistributing a full copy of a copyrighted paper isn't something to do casually in a public
   repo. Removed from every commit, not just the latest one — see the note in `eval/questions.json`
   for how to re-run the eval against your own copy.

Both were fixed with `git filter-repo` (strips content from every commit, not just adds a new one
on top that merely hides it going forward) and a force-push, done before this repo had any forks,
external clones, or open pull requests — the one window where a history rewrite is actually final
rather than just cosmetic. Original commit dates were preserved; only the author identity and the
one file changed.

---

*(This file grows as the project grows.)*
