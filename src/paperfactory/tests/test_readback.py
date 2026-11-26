"""Blind read-back (readback.py): sections, names, value matching, sorting, recall, repair, judge and cache on a
hand-made paper and a tiny world, with a fake backend (no model calls)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("thesis_scorer.facttools")   # thesis-scorer on PYTHONPATH (no package dependency)

from paperfactory import config, labels, llm, packs, readback as rb  # noqa: E402
from paperfactory.text import alias_table  # noqa: E402

F = packs.field("medicine")
ENTS = {"TR1": {"kind": "trial", "label": "the Velor-2 trial", "alt": ["Velor-2"], "attrs": {"year": 2019}},
        "TR2": {"kind": "trial", "label": "the Kesto trial", "alt": ["Kesto"], "attrs": {}},
        "TR3": {"kind": "trial", "label": "the Norvane trial", "alt": ["Norvane"], "attrs": {}},
        "T1": {"kind": "treatment", "label": "bramacet", "alt": [], "attrs": {"drug_class": "SGLT2 inhibitor"}}}
W = {"entities": ENTS}
MD = """# Bramacet trials: a review

## Abstract

The Velor-2 trial found that bramacet lowered HbA1c.

## Results

The Velor-2 trial randomised 210 patients. In Velor-2, bramacet gave a hazard ratio of 0.98 (95% CI 0.52-1.41).

Pooling Velor-2 and Kesto, the hazard ratio for bramacet was 0.75.

The Kesto trial found harm from bramacet. Velor-2 also reported harm from bramacet.

Velor-2 was stopped early in 2021. Most trials enrol adults over 50. The first trial was large.

## References

- Smith 2020.
"""
ROWS = [
    {"id": "P1:TR1-finds_benefit-T1", "paper": "P1", "what": "relation", "subject": "TR1", "relation": "finds_benefit",
     "object": "T1", "role": "claim", "conflict": False},
    {"id": "R-1", "paper": "P1", "what": "attr", "subject": "TR1", "attr": "patients", "value": "210", "unit": "n"},
    {"id": "R-2", "paper": "P1", "what": "effect", "subject": "TR1", "relation": "finds_benefit", "object": "T1",
     "stat": "estimate", "value": "0.98", "unit": "HR"},
    {"id": "R-3", "paper": "P1", "what": "effect", "subject": "TR1", "relation": "finds_benefit", "object": "T1",
     "stat": "ci_low", "value": "0.52", "unit": "HR"},
    {"id": "R-4", "paper": "P1", "what": "effect", "subject": "TR1", "relation": "finds_benefit", "object": "T1",
     "stat": "ci_high", "value": "1.41", "unit": "HR"},
    {"id": "R-5", "paper": "P1", "what": "pool", "subject": "T1", "stat": "estimate", "value": "0.75", "unit": "HR",
     "studies": ["TR1", "TR2"]},
    {"id": "P2:TR2-finds_harm-T1", "paper": "P2", "what": "relation", "subject": "TR2", "relation": "finds_harm",
     "object": "T1", "role": "claim", "conflict": False},
]


def C(tool, quote, **kw):
    return {"tool": tool, "paper": "P1", "quote": quote} | kw


READS = {   # the fake reader's calls per section (by heading word)
    "Abstract": [C("add_relation", "The Velor-2 trial found that bramacet lowered HbA1c.", subject="the Velor-2 trial",
                   relation="finds_benefit", object="bramacet")],
    "Results": [
        C("add_attr", "The Velor-2 trial randomised 210 patients.", subject="Velor-2", attr="patients", value="210"),
        C("add_effect", "In Velor-2, bramacet gave a hazard ratio of 0.98 (95% CI 0.52-1.41).", subject="Velor-2",
          object="bramacet", estimate="0.98", ci_low="0.52", ci_high="1.41"),
        C("add_pool", "Pooling Velor-2 and Kesto, the hazard ratio for bramacet was 0.75.", subject="bramacet",
          studies=["Velor-2", "Kesto"], estimate="0.57"),                     # wrong value: repaired below
        C("add_relation", "The Kesto trial found harm from bramacet.", subject="the Kesto trial",
          relation="finds_harm", object="bramacet"),
        C("add_relation", "Velor-2 also reported harm from bramacet.", subject="Velor-2", relation="finds_harm",
          object="bramacet"),
        C("add_custom", "Velor-2 was stopped early in 2021.", statement="Velor-2 was stopped early in 2021.",
          names=["Velor-2"], numbers=[{"value": "2021"}]),
        C("add_custom", "Most trials enrol adults over 50.", statement="Most trials enrol adults over 50.", names=[],
          numbers=[{"value": "50", "unit": "years"}]),
        C("add_custom", "The first trial was large.", statement="The first trial was large.",
          names=["The first trial"], numbers=[]),
        C("add_attr", "Smith 2020", subject="Smith", attr="year", value="2020")],   # not in the section: dropped
}
REPAIRED = [C("add_pool", "Pooling Velor-2 and Kesto, the hazard ratio for bramacet was 0.75.", subject="bramacet",
              studies=["Velor-2", "Kesto"], estimate="0.75")]


def fake():
    seen: list[tuple[str, str]] = []

    def backend(rm, system, user, schema):
        if system.startswith("Task: for each name"):
            items = json.loads(user)
            raw = {"answers": [{"id": it["id"], "entity": "TR1" if it["name"] == "The first trial" else "none"}
                               for it in items]}
            seen.append(("judge", user))
        elif system.startswith("Task: fix"):
            raw = {"calls": REPAIRED}
            seen.append(("repair", user))
        else:
            sec = json.loads(user)["section"]
            raw = {"calls": next((v for k, v in READS.items() if f"## {k}" in sec), [])}
            seen.append(("read", user))
        return {"raw": raw, "cost_usd": 0.01, "served_model": rm.model, "input_tokens": 1, "output_tokens": 1}
    backend.seen = seen
    return backend


def caller(tmp_path: Path, backend) -> llm.Caller:
    s = config.Settings("medicine", "argument", out=str(tmp_path / "run"), slot_dir=str(tmp_path / "slots"),
                        models={rb.READER_ROLE: "claude-sonnet-5-5", rb.JUDGE_ROLE: "claude-sonnet-5-5"})
    return llm.Caller(s, backend=backend)


def domain(tmp_path: Path) -> tuple[Path, Path]:
    seed, ds = tmp_path / "seed" / "bench_x", tmp_path / "dataset" / "bench_x"
    (ds / "P1").mkdir(parents=True)
    seed.mkdir(parents=True)
    (ds / "P1" / "paper.md").write_text(MD, encoding="utf-8")
    sp = lambda s: [[MD.index(s), MD.index(s) + len(s)]]  # noqa: E731
    man = {"field": "medicine", "genre": "argument", "papers": [
        {"id": "P1", "writer": "claude-haiku-5-5",
         "facts": [{"id": "TR1-finds_benefit-T1", "kind": "claim",
                    "spans": sp("The Velor-2 trial found that bramacet lowered HbA1c.")}],
         "numbers": [{"id": "R-1", "kind": "attr", "spans": sp("210")}, {"id": "R-5", "kind": "pool", "spans": []},
                     {"id": "R-2", "kind": "effect", "spans": sp("0.98")}]}]}
    (seed / "paperfactory.json").write_text(json.dumps(man), encoding="utf-8")
    (seed / "entities.json").write_text(json.dumps(ENTS), encoding="utf-8")
    (seed / "facts.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS), encoding="utf-8")
    return seed, ds


# ------------------------------------------------------------------ units
def test_sections_title_abstract_references_split():
    secs = rb.sections(MD)
    assert [s.heading for s in secs] == ["Bramacet trials: a review / Abstract", "Results"]
    assert secs[0].start == 0 and "References" not in "".join(s.text(MD) for s in secs)
    long = "# T\n\n## A\n\n" + "\n\n".join("word " * 50 for _ in range(10))
    parts = rb.sections(long, max_chars=600)
    assert len(parts) > 1 and all(p.end - p.start <= 600 for p in parts)
    assert "".join(p.text(long) for p in parts) == long


def test_name_routes():
    t = alias_table(W, F)
    assert rb.resolve_code("the Velor-2 trial", t)[:2] == ("TR1", "exact")
    assert rb.resolve_code("Velor-2", t)[:2] == ("TR1", "exact")
    assert rb.resolve_code("Kesto cohort", t)[:2] == ("TR2", "mention")
    assert rb.resolve_code("Velor-2 and Kesto", t)[1:] == ("ambiguous", ["TR1", "TR2"])
    assert rb.resolve_code("the sponsor", t) == (None, "no_match", [])


def test_value_match():
    assert rb.value_match("0.98", "0.98") == (True, [])
    assert rb.value_match("0.8", "0.75") == (True, ["rounded"])
    assert rb.value_match("0.75", "0.8") == (False, [])          # the read value is more precise: no rounding
    assert rb.value_match("65.2", "0.652", "65.2%") == (True, ["pct_scaled"])
    assert rb.value_match("65.2", "0.652", "65.2 points") == (False, [])
    assert rb.value_match("1.2", "1200", "EUR 1.2 million", "eur thousand") == (True, ["scaled"])
    assert rb.value_match("-0.3", "0.3") == (False, [])


def test_sort_outcomes():
    own, others = [r for r in ROWS if r["paper"] == "P1"], [r for r in ROWS if r["paper"] != "P1"]

    def srt(c, **ids):
        return rb.sort_claim(c, {k: [v] for k, v in ids.items()}, own, others, ENTS)
    rel = {"kind": "relation", "relation": "finds_benefit", "quote": "q"}
    assert srt(rel, subject="TR1", object="T1")["sort"] == "planned"
    assert srt(rel | {"relation": "finds_harm"}, subject="TR1", object="T1") == {
        "sort": "possible_invented", "facts": [], "flags": ["wrong_relation"]}
    assert srt(rel | {"relation": "finds_harm"}, subject="TR2", object="T1")["flags"] == ["other_paper_fact"]
    eff = {"kind": "effect", "quote": "q", "stat": "ci_low", "value": "0.98"}
    assert srt(eff, subject="TR1", object="T1") == {"sort": "planned", "facts": ["R-2"], "flags": ["stat_mixup"]}
    att = {"kind": "attr", "quote": "q", "value": "0.98", "text_value": False}
    assert srt(att, subject="TR1")["flags"] == ["kind_mixup"]
    assert srt(att | {"value": "1.41"}, subject="TR2") == {"sort": "possible_invented", "facts": [], "flags": []}
    assert srt(att | {"value": "2019"}, subject="TR1")["flags"] == ["entity_attr"]
    assert srt(att | {"value": "SGLT2 inhibitor", "text_value": True}, subject="T1")["flags"] == ["entity_attr"]
    assert srt(att | {"value": "0.75"}, subject=None)["flags"] == ["value_only"]
    assert srt(att | {"value": "99"}, subject=None)["sort"] == "background"
    assert srt(rel, subject=None, object="T1")["flags"] == ["group_summary"]
    assert srt(rel | {"relation": "finds_harm"}, subject=None, object="T1")["flags"] == ["group_conflict"]
    cus = {"kind": "custom", "quote": "q", "names": ["a", "b"], "numbers": []}
    assert srt(cus, names="TR1")["sort"] == "possible_invented"
    assert rb.sort_claim(cus, {"names": ["TR1", "T1"]}, own, others, ENTS)["flags"] == ["custom_pair"]


def test_sample_stratified():
    ps = [{"id": f"H{i}", "writer": "h"} for i in range(20)] + [{"id": f"Q{i}", "writer": "q"} for i in range(35)]
    got = rb.sample_papers(ps, 50, seed=3)
    assert len(got) == 50 and sum(p["writer"] == "h" for p in got) == 18
    assert rb.sample_papers(ps, 50, seed=3) == got
    assert rb.sample_papers(ps[:30], 50) == ps[:30]


# ------------------------------------------------------------------ end to end
def test_run_sorts_repairs_judges_and_caches(tmp_path):
    seed, ds = domain(tmp_path)
    b = fake()
    out = rb.run(seed, ds, caller(tmp_path, b))
    claims = out["papers"][0]["claims"]
    by = {(c["kind"], c.get("relation") or c.get("stat") or c.get("statement") or c.get("attr")): c for c in claims}
    assert by[("relation", "finds_benefit")]["sort"] == "planned"
    assert by[("pool", "estimate")]["repaired"] and by[("pool", "estimate")]["facts"] == ["R-5"]
    assert by[("relation", "finds_harm")]["sort"] in ("planned", "possible_invented")
    assert by[("custom", "Velor-2 was stopped early in 2021.")]["sort"] == "possible_invented"
    assert by[("custom", "Most trials enrol adults over 50.")]["sort"] == "background"
    first = by[("custom", "The first trial was large.")]
    assert first["routes"] == {"names": ["judge"]} and first["ids"] == {"names": ["TR1"]}
    s = out["summary"]
    assert s["rejected_calls"] == {"first_check_failed": 2, "repaired": 1, "dropped": 1}
    assert s["claims"] == {"planned": 7, "possible_invented": 3, "background": 1}
    assert s["writer_precision"]["k"] == 7 and s["writer_precision"]["n"] == 10
    assert s["wrong_relation"] == 1 and s["background"] == 1
    assert s["reader_recall"] == {"k": 3, "n": 3, "rate": 1.0, "ci95_exact": [0.2924, 1.0], "judge": out["reader_model"]}
    assert s["add_custom_share"]["k"] == 3
    assert out["by_writer"]["claude-haiku-5-5"]["writer_precision"]["n"] == 10
    assert {x["flags"][0] if x.get("flags") else "" for x in out["possible_invented"]} == {"wrong_relation", ""}
    n_calls = len(b.seen)
    assert [k for k, _ in b.seen].count("judge") == 1 and [k for k, _ in b.seen].count("repair") == 1
    again = rb.run(seed, ds, caller(tmp_path, b))                  # second run: every reply from the cache
    assert len(b.seen) == n_calls and again["summary"] == s
    cached = rb.run(seed, ds, caller(tmp_path, fake()), cache_only=True)
    assert cached["not_cached"] == 0
    # the reader never sees facts or entities
    reads = [u for k, u in b.seen if k == "read"]
    assert all("R-1" not in u and "TR1" not in u for u in reads)


def test_recall_miss(tmp_path):
    seed, ds = domain(tmp_path)
    keep = dict(READS)
    READS["Results"] = [c for c in keep["Results"] if c["tool"] != "add_attr"]
    try:
        out = rb.run(seed, ds, caller(tmp_path, fake()))
    finally:
        READS.update(keep)
    assert out["summary"]["reader_recall"]["k"] == 2 and out["recall_misses"][0]["fact"] == "R-1"


def test_export_invented(tmp_path):
    seed, ds = domain(tmp_path)
    out = rb.run(seed, ds, caller(tmp_path, fake()))
    got = labels.export_invented([out], tmp_path / "hc.csv", n_invented=2, n_planned=1)
    assert got["by_kind"] == {"possible_invented": 2, "planned_control": 1}
    rows = list(csv.DictReader((tmp_path / "hc.csv").open()))
    assert list(rows[0]) == labels.COLUMNS and all(r["label"] == "" for r in rows)
    key = json.loads(labels.key_path(tmp_path / "hc.csv").read_text())
    assert sorted(v["kind"] for v in key.values()) == ["planned_control", "possible_invented", "possible_invented"]
    assert "kind" not in rows[0] and all(r["sentence"] and r["relation"] for r in rows)
    assert "statement: None" not in "".join(r["relation"] for r in rows)


def test_words_match_and_paper_names():
    e = {"label": "Retron v. Nosor Media", "alt": []}
    facts = "a state broadcaster used voice data for content moderation"
    assert rb.words_match(facts, "Retron concerns a state broadcaster that moderated content with voice data", e)
    assert rb.words_match(facts, "voice data", e)                # a part of the planted description
    assert not rb.words_match(facts, "a bank sold location data", e)
    ents = {"P1": {"label": "section 1 of the Tetre Data Privacy Act", "alt": []},
            "P9": {"label": "section 12 of the Tetre Data Privacy Act", "alt": []}}
    assert rb.paper_prefix("Section 1", ["P1", "P9"], ents) == ("P1", "paper_prefix")
    assert rb.paper_prefix("section", ["P1", "P9"], ents) == (None, "no_match")
