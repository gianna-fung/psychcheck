"""
Runs eval/questions.json against the real pipeline and grades the results.

Usage: python eval/run_eval.py
(Run from the repo root, or anywhere -- paths below are resolved relative to this
file, not the current working directory, same reasoning as contested_findings.py.)

This makes REAL, BILLED API calls: one per question to generate an answer, plus one
more per factual question to grade that answer. For the 12 questions in
questions.json that's 12 + 6 = 18 calls -- at the per-call costs estimated during
development (a few cents at most per call), a full run costs well under a dollar,
so don't be afraid to re-run this after every prompt tweak.

Why grading needs a second LLM call, not just string matching:
Checking whether an answer correctly conveys "children waited much longer when
distracted from the rewards" against an answer that says "distraction significantly
increased delay times" is a semantic judgment -- those are the same fact in different
words, and no substring check reliably catches that kind of paraphrase. A second call
to Claude, given the question, the expected facts, and the answer, can make that
judgment the way a human grader would. This is the same "second LLM call" tradeoff
CLAUDE.md discusses for contested-findings matching, but it's the right call *here*:
this script isn't part of the shipped app's runtime (so an extra call per question
doesn't slow down or complicate the product), and an eval is exactly the place where
you want a more careful, more expensive judgment than the app itself makes.

Flag-correctness grading, by contrast, needs NO extra LLM call: whether the app
flagged the right (or no) contested finding is just checking a list the app already
returned, in code -- deterministic and free.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

# Make src/ importable regardless of where this script is run from.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from contested_findings import load_contested_findings  # noqa: E402
from pipeline import (  # noqa: E402
    answer_question,
    build_vectorstore,
    get_llm,
    load_pdf_pages,
    split_into_chunks,
)

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"


class FactualGrade(BaseModel):
    """What the grading LLM call returns for one factual question. Structured
    output again (see PaperOverview in pipeline.py for the same reasoning) so we get
    a real boolean to sum up at the end, not a paragraph we'd have to interpret."""

    facts_covered: list[str] = Field(
        description="Which of the expected facts the answer actually states, "
        "paraphrased however the answer stated them."
    )
    facts_missing: list[str] = Field(
        description="Which expected facts the answer does not state, or states "
        "incorrectly. Empty list if the answer covers everything."
    )
    passed: bool = Field(
        description="True only if every expected fact is correctly covered by the "
        "answer. A partially correct answer is a failure -- this is a strict grade "
        "on purpose, so the eval accuracy number means what it says."
    )


@dataclass
class QuestionResult:
    """One graded question, kept simple so the report-printing code at the bottom
    doesn't need to know about pydantic models or pipeline internals."""

    id: str
    category: str
    question: str
    answer: str
    passed: bool
    detail: str  # human-readable reason for the pass/fail, shown in the report


def grade_factual_answer(
    question: str, expected_key_facts: list[str], answer: str, llm
) -> FactualGrade:
    """The second LLM call described in this file's docstring: judges whether
    `answer` actually conveys every fact in `expected_key_facts`, regardless of
    exact wording."""
    structured_llm = llm.with_structured_output(FactualGrade)
    return structured_llm.invoke(
        [
            (
                "system",
                "You are grading whether an answer correctly covers a list of "
                "required facts. The answer doesn't need to match wording exactly "
                "-- paraphrases count as covered. But a fact that's absent, vague "
                "where the expected fact is specific (e.g. missing a number), or "
                "contradicted counts as missing. Be strict: this grade feeds an "
                "accuracy metric, so don't be generous with partial credit.",
            ),
            (
                "human",
                f"Question asked: {question}\n\n"
                f"Facts the answer should cover:\n"
                + "\n".join(f"- {fact}" for fact in expected_key_facts)
                + f"\n\nAnswer to grade:\n{answer}",
            ),
        ]
    )


def run_eval() -> list[QuestionResult]:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        eval_data = json.load(f)

    fixture_path = Path(__file__).resolve().parent.parent / eval_data["fixture_path"]
    print(f"Loading fixture paper: {fixture_path.name}")
    pages = load_pdf_pages(fixture_path)
    chunks = split_into_chunks(pages)
    vectorstore = build_vectorstore(chunks)

    contested_findings = load_contested_findings()
    llm = get_llm()

    results: list[QuestionResult] = []

    for q in eval_data["questions"]:
        print(f"  asking {q['id']}...")
        answer_result = answer_question(vectorstore, llm, q["question"], contested_findings)
        flagged_ids = [f.id for f in answer_result.flagged_findings]

        if q["category"] == "factual":
            grade = grade_factual_answer(
                q["question"], q["expected_key_facts"], answer_result.answer, llm
            )
            detail = (
                "covered: " + ", ".join(grade.facts_covered)
                if grade.passed
                else "MISSING: " + ", ".join(grade.facts_missing)
            )
            results.append(
                QuestionResult(q["id"], q["category"], q["question"], answer_result.answer, grade.passed, detail)
            )

        elif q["category"] == "should_flag":
            expected_id = q["expected_flag_id"]
            passed = expected_id in flagged_ids
            detail = (
                f"expected flag '{expected_id}', got {flagged_ids or '(none)'}"
            )
            results.append(
                QuestionResult(q["id"], q["category"], q["question"], answer_result.answer, passed, detail)
            )

        elif q["category"] == "should_not_flag":
            passed = len(flagged_ids) == 0
            detail = "no flags fired" if passed else f"unexpectedly flagged: {flagged_ids}"
            results.append(
                QuestionResult(q["id"], q["category"], q["question"], answer_result.answer, passed, detail)
            )

        else:
            raise ValueError(f"Unknown question category: {q['category']!r}")

    return results


def print_report(results: list[QuestionResult]) -> None:
    print("\n" + "=" * 70)
    print("EVAL REPORT")
    print("=" * 70)

    for category in ["factual", "should_flag", "should_not_flag"]:
        category_results = [r for r in results if r.category == category]
        if not category_results:
            continue
        passed_count = sum(r.passed for r in category_results)
        print(f"\n--- {category} ({passed_count}/{len(category_results)} passed) ---")
        for r in category_results:
            status = "PASS" if r.passed else "FAIL"
            print(f"[{status}] {r.id}: {r.question}")
            print(f"       {r.detail}")

    total_passed = sum(r.passed for r in results)
    print(f"\n{'=' * 70}")
    print(f"OVERALL: {total_passed}/{len(results)} passed ({100 * total_passed / len(results):.0f}%)")
    print("=" * 70)


if __name__ == "__main__":
    results = run_eval()
    print_report(results)

    # Also save the full detail (including complete answer text, not just the
    # pass/fail summary printed above) so a specific failure can be inspected later
    # without re-running the whole eval.
    output_path = Path(__file__).resolve().parent / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "id": r.id,
                    "category": r.category,
                    "question": r.question,
                    "answer": r.answer,
                    "passed": r.passed,
                    "detail": r.detail,
                }
                for r in results
            ],
            f,
            indent=2,
        )
    print(f"\nFull results written to {output_path}")
