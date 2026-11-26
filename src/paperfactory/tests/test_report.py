"""report: marks from spans (priority, no-position list, purple from read-back quotes) and formatting (no reportlab)."""
from __future__ import annotations

import json
from pathlib import Path

from paperfactory import report


def _domain(tmp: Path) -> Path:
    data = tmp / "data"
    seed, ds = data / "seed" / "bench_x", data / "dataset" / "bench_x" / "P1"
    seed.mkdir(parents=True)
    ds.mkdir(parents=True)
    md = "A found B. C found B with 12 units. Extra claim here."
    (ds / "paper.md").write_text(md)
    facts = [{"id": "A-f-B", "kind": "claim", "subject": "A", "relation": "f", "object": "B", "spans": [[0, 10]]},
             {"id": "C-f-B", "kind": "decoy", "subject": "C", "relation": "f", "object": "B", "spans": [[11, 35]]},
             {"id": "D-f-B", "kind": "decoy", "subject": "D", "relation": "f", "object": "B", "spans": []}]
    nums = [{"label": "C: units", "value": "12", "unit": "units", "spans": [[27, 29]]},
            {"label": "A: n", "value": "5", "spans": []}]
    (seed / "paperfactory.json").write_text(json.dumps({"papers": [{"id": "P1", "title": "T", "facts": facts,
                                                                      "numbers": nums, "writer": "w"}]}))
    (seed / "facts.jsonl").write_text(json.dumps({"id": "P1:A-f-B", "paper": "P1", "what": "relation",
                                                  "conflict": True}))
    (seed / "readback.json").write_text(json.dumps({"papers": [{"paper": "P1", "writer": "w", "claims": [
        {"sort": "possible_invented", "quote": "Extra claim here.", "section_start": 0}]}]}))
    return data


def test_marks_priority_nopos_and_invented(tmp_path):
    data = _domain(tmp_path)
    md, mark, nk, nopos = report.marks_for(data, "bench_x", "P1")
    assert mark[0] == "conflict"                       # the claim is flagged conflict in facts.jsonl
    assert mark[12] == "decoy" and mark[27] == "number"  # number wins inside the distractor span
    assert set(mark[md.index("Extra"):]) == {"invented"}
    assert nk == {"invented": 1, "claim": 0, "decoy": 1, "conflict": 1, "number": 1}
    assert nopos == ["distractor (true near-miss fact): D f B", "number: A: n = 5"]
    assert report.pick_samples(data, "bench_x") == ["P1"]


def test_formatting():
    assert report.pct(0.9512) == "95.1%" and report.ci([0.946, 0.9549]) == "[94.6, 95.5]"
    assert report.pct(None) == "n/a" and report.ci(None) == ""
