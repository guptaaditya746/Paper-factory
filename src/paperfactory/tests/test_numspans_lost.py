"""numspans_lost.py: reason classes from numspans notes, marks, merge snippet, fact recall with three buckets."""
from __future__ import annotations

import json

import pytest

from paperfactory import numspans_lost as nl


@pytest.mark.parametrize("notes, reason, status", [
    (["rule 4: same value as R-1, R-2 -> verifier", "place 193: verifier no on ['subject', 'stat', 'value']"],
     nl.R_SAME, nl.UNKNOWN),
    (["rule 4: 2 places -> verifier", "place 3: verifier no on ['stat']"], nl.R_SEVERAL, nl.UNKNOWN),
    (["place 82: gate (subject not named)", "place 81: verifier no on ['subject']"], nl.R_VER, nl.UNKNOWN),
    (["place 88: gate (stat: the value is estimate here, the fact is ci_low)"], nl.R_GATE_STAT, nl.UNKNOWN),
    (["place 82: gate (subject not named)"], nl.R_GATE_SUBJ, nl.UNKNOWN),
    (["place 82: gate (subject not named)", "place 107: gate (stat: the value is estimate here, the fact is ci_high)"],
     nl.R_GATE_BOTH, nl.UNKNOWN),
    (["place 5: no verifier answer"], nl.R_NO_ANSWER, nl.UNKNOWN),
    (["12 places, first 8 sent to the verifier"], nl.R_NONE, nl.UNKNOWN),
    ([], nl.R_NONE, nl.UNKNOWN),
    (["value_not_found"], nl.R_NOT_FOUND, nl.MISSING),
    (["no value"], nl.R_NO_VALUE, nl.MISSING),
])
def test_reason_of(notes, reason, status):
    assert nl.reason_of(notes) == (reason, status)


def test_verifier_no_on():
    assert nl.verifier_no_on(["place 89: verifier no on ['subject', 'stat']", "place 193: verifier no on ['value']",
                              "rule 4: 2 places -> verifier"]) == ["stat", "subject", "value"]


REP = {"coverage": {}, "no_span": [
    {"id": "N1", "paper": "P1", "kind": "attr", "value": "3",
     "notes": ["rule 4: same value as N9 -> verifier", "place 4: verifier no on ['stat']"]},
    {"id": "N2", "paper": "P1", "kind": "effect", "value": "0.6", "notes": ["place 2: gate (subject not named)"]},
    {"id": "N3", "paper": "P2", "kind": "effect", "value": "7.7", "notes": ["value_not_found"]}]}

MAN = {"papers": [
    {"id": "P1", "facts": [{"id": "F1", "kind": "claim", "spans": [[0, 5]]}, {"id": "F2", "kind": "decoy", "spans": []}],
     "numbers": [{"id": "N1", "kind": "attr", "spans": []}, {"id": "N2", "kind": "effect", "spans": []},
                 {"id": "N4", "kind": "attr", "spans": [[9, 10]]}]},
    {"id": "P2", "facts": [], "numbers": [{"id": "N3", "kind": "effect", "spans": []}]}]}


def test_marks_and_summary():
    marks = nl.position_unknown(REP)
    assert [m["id"] for m in marks] == ["N1", "N2"]                 # N3: value not in paper -> missing, not marked
    assert marks[0] | {"notes": None} == {"id": "N1", "paper": "P1", "kind": "attr", "value": "3",
                                          "status": "position unknown", "reason": nl.R_SAME,
                                          "verifier_no_on": ["stat"], "notes": None}
    s = nl.summary(marks)
    assert (s["count"], s["by_reason"], s["ids"]) == (2, {nl.R_SAME: 1, nl.R_GATE_SUBJ: 1}, ["N1", "N2"])
    merged = nl.merge(REP, marks)
    assert merged["position_unknown"]["count"] == 2 and "position_unknown" not in REP


def test_fact_recall_three_buckets():
    r = nl.fact_recall(MAN, {"N1", "N2"})
    # 6 planted: placed F1, N4; position unknown N1, N2; missing F2, N3 -> recall 2 / (6 - 2)
    assert (r["planted"], r["placed"], r["position_unknown"], r["missing"], r["recall"]) == (6, 2, 2, 2, 0.5)
    assert r["by_kind"]["attr"] == r["by_kind"]["attr"] | {"planted": 2, "placed": 1, "position_unknown": 1,
                                                           "missing": 0, "recall": 1.0}
    assert r["by_kind"]["effect"]["recall"] == 0.0 and r["by_kind"]["claim"]["recall"] == 1.0
    lo, hi = r["ci95"]
    assert lo < 0.5 < hi


def test_fact_recall_all_unknown_has_no_rate():
    man = {"papers": [{"id": "P", "facts": [], "numbers": [{"id": "N", "kind": "attr", "spans": []}]}]}
    assert nl.fact_recall(man, {"N"})["recall"] is None


def test_run_and_merge_file(tmp_path):
    (tmp_path / "numspans.json").write_text(json.dumps(REP), encoding="utf-8")
    (tmp_path / "paperfactory.json").write_text(json.dumps(MAN), encoding="utf-8")
    out = nl.run(tmp_path)
    assert out["missing_value_not_in_paper"] == ["N3"] and out["fact_recall"]["recall"] == 0.5
    saved = json.loads((tmp_path / nl.OUT_NAME).read_text(encoding="utf-8"))
    assert [m["id"] for m in saved["facts"]] == ["N1", "N2"]
    rep = nl.merge_file(tmp_path / "numspans.json", tmp_path / nl.OUT_NAME)
    assert json.loads((tmp_path / "numspans.json").read_text())["position_unknown"]["ids"] == ["N1", "N2"]
    assert rep["no_span"] == REP["no_span"]
