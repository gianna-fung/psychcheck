# eval

A 12-question eval set that runs the real pipeline against a real paper and grades the results —
6 factual questions, 3 that should trigger a contested-findings flag, 3 that shouldn't.

- `questions.json` — the questions, expected facts/flags, and notes on a couple of known risks
- `run_eval.py` — runs all 12 through `src/pipeline.py` for real (makes real, billed API calls)
  and grades them: deterministic checks for flag correctness, a second Claude call for factual
  correctness (see `CLAUDE.md` for why grading needs its own LLM call)
- `results.json` — the last real run's full results (answers, pass/fail, grading detail)

**The source PDF isn't included in this repo** — see the `_fixture_removed` note at the top of
`questions.json` for why (it's a copyrighted journal article) and how to re-run this yourself
against your own copy.

Run with:

```bash
python eval/run_eval.py
```
