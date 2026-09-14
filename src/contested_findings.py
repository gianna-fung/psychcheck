"""
Loads the hand-curated list of contested / failed-replication psychology findings
(data/contested_findings.json) and checks generated text against it.

Design note (see CLAUDE.md for the full reasoning): this uses plain keyword/phrase
matching, not a second Claude API call. That's a deliberate tradeoff for a single-pass
pipeline — deterministic, free, and easy to debug, at the cost of missing paraphrases
that don't use any of the listed keywords.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Resolve the JSON path relative to this file, not the current working directory,
# so this module works the same whether you run it from src/, the repo root, or
# import it from app.py running under Streamlit (which can change the cwd).
DEFAULT_FINDINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "contested_findings.json"


@dataclass
class ContestedFinding:
    """One entry from contested_findings.json, as a typed object instead of a raw dict
    so the rest of the codebase gets autocomplete and typo-checking on field names."""

    id: str
    topic: str
    original_claim: str
    original_citation: str
    status: str
    evidence_summary: str
    replication_citations: list[str]
    keywords: list[str]

    @classmethod
    def from_dict(cls, data: dict) -> "ContestedFinding":
        return cls(
            id=data["id"],
            topic=data["topic"],
            original_claim=data["original_claim"],
            original_citation=data["original_citation"],
            status=data["status"],
            evidence_summary=data["evidence_summary"],
            replication_citations=data["replication_citations"],
            keywords=data["keywords"],
        )


def load_contested_findings(path: Path = DEFAULT_FINDINGS_PATH) -> list[ContestedFinding]:
    """Reads contested_findings.json and returns it as a list of ContestedFinding objects.

    This is called once per app session (see app.py), not per question — the file is
    small and doesn't change while the app is running, so there's no need to re-read
    it from disk on every question.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [ContestedFinding.from_dict(entry) for entry in raw["findings"]]


def check_for_contested_findings(
    text: str, findings: list[ContestedFinding]
) -> list[ContestedFinding]:
    """Returns every ContestedFinding whose keywords appear in `text`.

    Matching is case-insensitive substring matching on whole keyword phrases (e.g.
    "power posing"), not single-word matching. Single-word matching would flag far too
    much unrelated text (imagine flagging every mention of the word "priming"); requiring
    the fuller phrase trades recall for precision, which matters more for a flag that
    interrupts the user with a warning.
    """
    text_lower = text.lower()
    matched = []
    for finding in findings:
        if any(keyword.lower() in text_lower for keyword in finding.keywords):
            matched.append(finding)
    return matched


if __name__ == "__main__":
    # Quick manual sanity check: `python src/contested_findings.py`
    # Loads the file and runs a couple of example checks so you can see the matcher
    # working without needing the full Streamlit app or an API key.
    findings = load_contested_findings()
    print(f"Loaded {len(findings)} contested findings.\n")

    examples = [
        "The study found that holding a power pose for two minutes increased confidence.",
        "Participants who practiced growth mindset techniques showed no change in test scores.",
        "The paper's method used a between-subjects design with 200 undergraduates.",
    ]
    for example in examples:
        matches = check_for_contested_findings(example, findings)
        match_ids = [m.id for m in matches] or ["(no match)"]
        print(f"Text: {example!r}\n  -> {match_ids}\n")
