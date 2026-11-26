"""seedcheck: seed lines in words, answer checks, one repair, labels in key + review sheet (fake backend)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from paperfactory import config, llm, seedcheck as sc

ENTS = {"TR1": {"kind": "trial", "label": "Velor-2", "alt": [], "attrs": {"sites": 51}},
        "T1": {"kind": "treatment", "label": "bramacet", "alt": [], "attrs": {}}}
ROWS = [{"id": "P1:TR1-finds_benefit-T1", "paper": "P1", "what": "relation", "subject": "TR1",
         "relation": "finds_benefit", "object": "T1", "role": "claim", "conflict": False},
        {"id": "P1:TR1-n", "paper": "P1", "what": "attr", "subject": "TR1", "attr": "enrolled", "value": "210",
         "unit": "patients"},
        {"id": "P2:x", "paper": "P2", "what": "attr", "subject": "TR1", "attr": "enrolled", "value": "9"}]


def test_seed_lines_use_names_and_only_this_paper():
    lines = sc.seed_lines(ROWS, ENTS, "P1", ["T1"])
    assert lines["TR1-finds_benefit-T1"] == "Velor-2 finds_benefit bramacet"
    assert lines["TR1-n"] == "Velor-2 enrolled = 210 patients"
    assert "x" not in lines and lines["E:TR1"] == "Velor-2 is a trial (sites = 51)"


def test_check_answer():
    ok = {"TR1-n"}
    assert sc.check_answer({"label": "extra_consistent", "facts": [], "reason": "r"}, ok) is None
    assert "needs" in sc.check_answer({"label": "contradicts_seed", "facts": [], "reason": "r"}, ok)
    assert "not given" in sc.check_answer({"label": "contradicts_seed", "facts": ["F9"], "reason": "r"}, ok)


def test_run_labels_key_and_sheet(tmp_path: Path):
    seed = tmp_path / "data" / "seed" / "bench_medicine"
    seed.mkdir(parents=True)
    (seed / "facts.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS))
    (seed / "entities.json").write_text(json.dumps(ENTS))
    key = {"rb-0001": {"kind": "possible_invented", "domain": "bench_medicine", "field": "medicine", "paper": "P1",
                       "writer": "claude-haiku-5-5", "quote": "Velor-2 found harm from bramacet.",
                       "ids": {"subject": ["TR1"], "object": ["T1"]},
                       "claim": {"subject": "Velor-2", "relation": "finds_harm", "object": "bramacet"}},
           "rb-0002": {"kind": "planned_control", "domain": "bench_medicine", "field": "medicine", "paper": "P1"}}
    kp = tmp_path / "data" / "seed" / "hc.key.json"
    kp.write_text(json.dumps(key))
    calls = []

    def backend(rm, system, user, schema):
        calls.append(user)
        raw = ({"label": "contradicts_seed", "facts": [], "reason": "x"} if len(calls) == 1 else
               {"label": "contradicts_seed", "facts": ["TR1-finds_benefit-T1"], "reason": "benefit, not harm"})
        return {"raw": raw, "cost_usd": 0.0, "served_model": rm.model, "input_tokens": 1, "output_tokens": 1}
    s = config.Settings("medicine", "argument", out=str(tmp_path / "run"), slot_dir=str(tmp_path / "slots"),
                        models={sc.ROLE: "claude-haiku-5-5"})
    rep = sc.run(kp, tmp_path / "data", llm.Caller(s, backend=backend))
    assert rep["counts"] == {"contradicts_seed": 1} and len(calls) == 2          # one repair after the bad answer
    got = json.loads(kp.read_text())["rb-0001"]["haiku_first_pass"]
    assert got["label_text"] == "contradicts seed" and got["judge_note"] == "same model as writer"
    assert "haiku_first_pass" not in json.loads(kp.read_text())["rb-0002"]
    assert got["status"] == "first pass, not the hand check"
    rows = list(csv.DictReader(Path(rep["sheet"]).open()))
    assert [r["id"] for r in rows] == ["rb-0001"] and rows[0]["contradicted_seed_lines"] == "Velor-2 finds_benefit bramacet"
    assert rows[0]["haiku_first_pass"] == "contradicts seed" and rows[0]["your_seed_label"] == ""
