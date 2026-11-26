"""Number facts with no position, and fact recall (metric 1) with three buckets.

numspans.py writes `no_span` in seed/<domain>/numspans.json: number facts (attr, effect, pool) that got no span. Most of
them have their value in the paper, but the code rules and the verifier could not prove which place is theirs (same
value as another fact, subject not named in the passage, verifier said no). Those are "position unknown", not "the
writer left the fact out". Only a fact whose value is nowhere in the paper ("value_not_found") or that has no value
stays "missing".

  reason_of(notes)        -> (reason class, status): a short reason per fact from its numspans notes
  position_unknown(rep)   -> marks, one per no_span fact whose value is in the paper
  summary(marks)          -> {count, by_reason, by_kind, ids}: the key to merge into numspans.json
  merge(rep, marks)       -> rep with "position_unknown" set (the caller writes numspans.json)
  fact_recall(man, ids)   -> planted facts in placed / position unknown / missing;
                             recall = placed / (planted - position unknown), position unknown shown beside it

Run: python -m paperfactory.numspans_lost <seed_dir>... writes <seed_dir>/numspans_position_unknown.json
(marks, the summary to merge, fact recall). It reads numspans.json and paperfactory.json and writes nothing else.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from paperfactory.evaluate import wilson

UNKNOWN, MISSING = "position unknown", "missing"
OUT_NAME = "numspans_position_unknown.json"

# reason classes, most specific first; the verifier ones say why the fact went to the verifier
R_SAME = "same value as another fact, verifier said no"
R_SEVERAL = "value in several places, verifier said no"
R_VER = "verifier said no"
R_NO_ANSWER = "verifier gave no answer"
R_GATE_SUBJ = "gate: subject not named"
R_GATE_STAT = "gate: value has another role here (estimate vs CI bound)"
R_GATE_BOTH = "gate: subject not named or value has another role"
R_NONE = "no place found"
R_NOT_FOUND = "value not in paper"
R_NO_VALUE = "fact has no value"

_VER_NO = re.compile(r"verifier no on \[(.*)\]")


def reason_of(notes: list[str]) -> tuple[str, str]:
    """(reason class, status) from numspans notes; status is "missing" only if the value is not in the paper."""
    if "no value" in notes:
        return R_NO_VALUE, MISSING
    if "value_not_found" in notes:
        return R_NOT_FOUND, MISSING
    ver_no = any(_VER_NO.search(n) for n in notes)
    if ver_no:
        if any(n.startswith("rule 4: same value as") for n in notes):
            return R_SAME, UNKNOWN
        if any(n.startswith("rule 4:") for n in notes):
            return R_SEVERAL, UNKNOWN
        return R_VER, UNKNOWN
    if any("no verifier answer" in n for n in notes):
        return R_NO_ANSWER, UNKNOWN
    gates = [n for n in notes if ": gate (" in n]
    subj = any("gate (subject not named)" in n for n in gates)
    stat = any("gate (stat:" in n for n in gates)
    if subj and stat:
        return R_GATE_BOTH, UNKNOWN
    if subj:
        return R_GATE_SUBJ, UNKNOWN
    if stat:
        return R_GATE_STAT, UNKNOWN
    return R_NONE, UNKNOWN


def verifier_no_on(notes: list[str]) -> list[str]:
    """The fields (subject, stat, value) the verifier said no on, over all places."""
    out: set[str] = set()
    for n in notes:
        if m := _VER_NO.search(n):
            out |= {x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()}
    return sorted(out)


def classify(rep: dict[str, Any]) -> list[dict[str, Any]]:
    """Every no_span fact with its reason and status (position unknown or missing)."""
    out = []
    for x in rep.get("no_span", []):
        reason, status = reason_of(x.get("notes", []))
        out.append({"id": x["id"], "paper": x["paper"], "kind": x["kind"], "value": x.get("value"),
                    "status": status, "reason": reason, "verifier_no_on": verifier_no_on(x.get("notes", [])),
                    "notes": x.get("notes", [])})
    return out


def position_unknown(rep: dict[str, Any]) -> list[dict[str, Any]]:
    """The marks: no_span facts whose value is in the paper but whose place could not be proven."""
    return [m for m in classify(rep) if m["status"] == UNKNOWN]


def summary(marks: list[dict[str, Any]]) -> dict[str, Any]:
    """The key to merge into numspans.json as "position_unknown"."""
    return {"count": len(marks), "by_reason": dict(Counter(m["reason"] for m in marks).most_common()),
            "by_kind": dict(Counter(m["kind"] for m in marks).most_common()),
            "ids": [m["id"] for m in marks],
            "note": "number facts with no span whose value is in the paper: the code rules and the verifier could "
                    "not prove which place is theirs. Fact recall leaves them out of the denominator "
                    "(numspans_lost.fact_recall); they are not counted as left out by the writer."}


def merge(rep: dict[str, Any], marks: list[dict[str, Any]]) -> dict[str, Any]:
    """numspans.json report with "position_unknown" added; returns a new dict, the caller writes the file."""
    return rep | {"position_unknown": summary(marks)}


def merge_file(numspans_path: Path, marks_path: Path | None = None) -> dict[str, Any]:
    """Merge the summary of <seed_dir>/numspans_position_unknown.json (or of numspans.json itself) into numspans.json
    and write it back."""
    rep = json.loads(numspans_path.read_text(encoding="utf-8"))
    marks = json.loads(marks_path.read_text(encoding="utf-8"))["facts"] if marks_path else position_unknown(rep)
    rep = merge(rep, marks)
    numspans_path.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    return rep


def _bucket(n: int, placed: int, unknown: int) -> dict[str, Any]:
    den = n - unknown
    return {"planted": n, "placed": placed, "position_unknown": unknown, "missing": den - placed,
            "recall": round(placed / den, 3) if den else None, "ci95": wilson(placed, den)}


def fact_recall(man: dict[str, Any], unknown_ids: set[str]) -> dict[str, Any]:
    """Metric 1 from paperfactory.json: each paper's relation facts ("facts") and number facts ("numbers") with their
    tag-step spans. placed = has a span; position unknown = number fact in unknown_ids with no span; missing = the
    rest. recall = placed / (planted - position unknown), Wilson 95% interval; by kind too (claim/decoy/... and
    attr/effect/pool)."""
    rows: list[tuple[str, str]] = []                 # (kind, bucket)
    for p in man["papers"]:
        for fa in p.get("facts", []) + p.get("numbers", []):
            b = "placed" if fa.get("spans") else UNKNOWN if fa["id"] in unknown_ids else MISSING
            rows.append((fa["kind"], b))

    def part(xs: list[tuple[str, str]]) -> dict[str, Any]:
        return _bucket(len(xs), sum(b == "placed" for _, b in xs), sum(b == UNKNOWN for _, b in xs))
    return part(rows) | {"by_kind": {k: part([x for x in rows if x[0] == k]) for k in sorted({k for k, _ in rows})},
                         "rule": "recall = placed / (planted - position unknown); position unknown = number facts "
                                 "whose value is in the paper but whose place could not be proven (numspans_lost)"}


def run(seed_dir: Path) -> dict[str, Any]:
    """Marks, merge snippet and fact recall for one domain; writes <seed_dir>/numspans_position_unknown.json."""
    rep = json.loads((seed_dir / "numspans.json").read_text(encoding="utf-8"))
    man = json.loads((seed_dir / "paperfactory.json").read_text(encoding="utf-8"))
    every = classify(rep)
    marks = [m for m in every if m["status"] == UNKNOWN]
    out = {"source": "numspans_lost.py from numspans.json no_span", "domain": seed_dir.name,
           "no_span": len(every), "missing_value_not_in_paper": [m["id"] for m in every if m["status"] == MISSING],
           "position_unknown": summary(marks), "fact_recall": fact_recall(man, {m["id"] for m in marks}),
           "facts": marks}
    (seed_dir / OUT_NAME).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seed_dir", nargs="+")
    for d in ap.parse_args().seed_dir:
        o = run(Path(d))
        print(json.dumps({"domain": o["domain"], "no_span": o["no_span"], "position_unknown": o["position_unknown"]
                          ["by_reason"], "fact_recall": {k: v for k, v in o["fact_recall"].items()
                                                         if k not in ("by_kind", "rule")}}))
