"""number-spans (paperfactory/numspans.py) on a hand-made paper: formats, code rules, verifier, controls, cache."""
from __future__ import annotations

import json
from pathlib import Path

from paperfactory import llm, numspans
from paperfactory.numspans import canon, ci_mismatches, numbers_in, wrong_value
from paperfactory.tests.test_llm import settings

MD = """# Decoding methods

| Study | Estimate | Lower | Upper |
|---|---|---|---|
| Lof 2020 | 2.25 | 1.75 | 2.75 |

## Results

Kor 2021 compared 3 models on 3,050 examples. The pooled gain was 1.20 percent overall. A total of 3 runs failed.

| Study | Method | Baseline accuracy (%) | Difference (%), estimate (lower to upper) |
|---|---|---|---|
| Kor 2021 | Ragrer decoding | 65.2 | 9.0 (7.3 to 10.8) |
| Bel 2019 | Ragrer decoding | 65.2 | −1.9 (−3.7 to −0.5) |

Seeds were 5 in every study except one. Pooled, Ragrer decoding gained 10.8 percent.
"""
ENTS = {"S1": {"kind": "study", "label": "Kor 2021"}, "S2": {"kind": "study", "label": "Bel 2019"},
        "S3": {"kind": "study", "label": "Lof 2020"},
        "M1": {"kind": "method", "label": "Ragrer decoding", "alt": ["Ragrer"]}}


def row(i, what, value, unit="%", **kw):
    return {"id": i, "paper": "P-01", "what": what, "value": value, "unit": unit, "label": i, **kw}


ROWS = [row("A1", "attr", "3", "models", subject="S1", attr="models_compared"),
        row("A2", "attr", "3050", "n", subject="S1", attr="examples"),
        row("P1", "pool", "1.2", subject="M1", stat="estimate", studies=["S1", "S2"]),
        row("E1", "effect", "65.2", subject="S1", relation="reports_gain", object="M1", stat="baseline"),
        row("E1b", "effect", "65.2", subject="S2", relation="reports_drop", object="M1", stat="baseline"),
        row("E2", "effect", "-1.9", subject="S2", relation="reports_drop", object="M1", stat="estimate"),
        row("E2l", "effect", "-3.7", subject="S2", relation="reports_drop", object="M1", stat="ci_low"),
        row("E2h", "effect", "-0.5", subject="S2", relation="reports_drop", object="M1", stat="ci_high"),
        row("E3", "effect", "9", subject="S1", relation="reports_gain", object="M1", stat="estimate"),
        row("A3", "attr", "5", "n", subject="S2", attr="seeds"),
        row("A4", "attr", "42", "n", subject="S1", attr="tasks"),
        row("P2", "pool", "10.8", subject="M1", stat="ci_high", studies=["S1", "S2"]),
        row("F", "effect", "2.25", subject="S3", relation="reports_gain", object="M1", stat="estimate"),
        row("Fl", "effect", "1.75", subject="S3", relation="reports_gain", object="M1", stat="ci_low"),
        row("Fh", "effect", "2.75", subject="S3", relation="reports_gain", object="M1", stat="ci_high")]


def domain(tmp: Path) -> tuple[Path, Path]:
    seed, ds = tmp / "seed" / "d", tmp / "dataset" / "d"
    (ds / "P-01").mkdir(parents=True)
    seed.mkdir(parents=True)
    (ds / "P-01" / "paper.md").write_text(MD, encoding="utf-8")
    (seed / "entities.json").write_text(json.dumps(ENTS), encoding="utf-8")
    rel = {"id": "P-01:S1-reports_gain-M1", "paper": "P-01", "what": "relation", "subject": "S1",
           "relation": "reports_gain", "object": "M1", "role": "claim", "conflict": False}
    (seed / "facts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [rel] + ROWS), encoding="utf-8")
    man = {"field": "llm", "genre": "argument", "papers": [{"id": "P-01", "title": "t", "facts": [],
                                                             "source": {"run": "/x/runs/paperfactory/bench20h-llm"}}]}
    (seed / "paperfactory.json").write_text(json.dumps(man), encoding="utf-8")
    return seed, ds


def honest():
    """A fake verifier: yes when the fact's value is written in the passage (so every control gets "no")."""
    seen = []

    def backend(rm, system, user, schema):
        items = json.loads(user)
        seen.append(items)
        ans = [{"id": it["id"], "verdict": "yes" if canon(it["fact"].split(" = ")[1].split()[0])
                in {v for _, _, v in numbers_in(it.get("marked") or it["passage"])} else "no"} for it in items]
        return {"raw": {"answers": ans}, "cost_usd": 0.0, "served_model": rm.model, "input_tokens": 1,
                "output_tokens": 1}
    backend.seen = seen
    return backend


def test_canon_formats() -> None:
    assert canon("9.0") == canon("9") == "9"
    assert canon(".5") == "0.5" and canon("0.50") == "0.5"
    assert canon("3,050") == "3050" and canon("100") == "100"
    assert canon("−1.8") == "-1.8" and canon("-0.0") == "0"


def test_numbers_in_signs_ranges_and_names() -> None:
    vals = [v for _, _, v in numbers_in("gain -1.9 (−3.7 to 0.5), range 7.3-10.8, Thipom-7B v2.3, Stetrugel-6, 3,050 cases")]
    assert vals == ["-1.9", "-3.7", "0.5", "7.3", "10.8", "3050"]     # 10.8 after a range dash is positive


def test_wrong_value_keeps_decimals_and_avoids_passage() -> None:
    w = wrong_value("65.2", {"65.9", "65.2"})
    assert w not in {"65.9", "65.2"} and "." in w


def test_code_rules_verifier_and_controls(tmp_path: Path) -> None:
    seed, ds = domain(tmp_path)
    b = honest()
    rep = numspans.run(seed, ds, llm.Caller(settings(tmp_path, verifier="v"), backend=b))
    man = json.loads((seed / "paperfactory.json").read_text())
    got = {n["id"]: n for n in man["papers"][0]["numbers"]}
    how = {i: n["found_by"] for i, n in got.items()}
    assert how["A1"] == ["several+name"] and len(got["A1"]["spans"]) == 1      # "3 runs failed" names no one
    assert how["A2"] == ["once+name"]                                          # 3,050 = 3050
    assert how["P1"] == ["once+unit"]                                          # 1.20 = 1.2, "percent"
    assert how["E1"] == ["several+name"]                                       # only the Kor 2021 row
    assert how["E2"] == how["E2l"] == how["E2h"] == ["once+name"]              # unicode minus
    assert how["E3"] == ["once+name"]                                          # 9.0 = 9
    assert how["A3"] == ["verifier"]                                           # unit "n" is no evidence
    assert got["A4"]["spans"] == []
    s, e = got["P2"]["spans"][0]                                           # not Kor 2021's own 10.8 in its row
    assert how["P2"] == ["several+name"] and MD[s:e].startswith("Pooled")
    s, e = got["E1"]["spans"][0]
    assert MD[s:e] == "65.2" and MD.rindex("Kor 2021", 0, s) > MD.index("| Study")   # raw offsets, Kor's row
    assert rep["controls"] == {"items": 1, "no": 1, "yes": 0, "no_answer": 0, "note": rep["controls"]["note"]}
    assert rep["controls_by_writer"]["other_model"]["items"] == 1         # verifier "v" did not write bench20h
    assert man["papers"][0]["writer"] == "claude-haiku-5-5" and man["number_spans"]["verifier_model"] == "v"
    assert man["number_spans"]["independence"]["same_model_as_a_writer"] is False
    # Lof 2020: three cells of one row = shared; the pool's estimate (P1) and upper bound (P2) are in two sentences
    assert [m["ids"] for m in rep["ci_mismatch"]] == [{"estimate": "P1", "ci_high": "P2"}] and len(b.seen) == 3   # spans, hard controls, code audit
    assert [x["id"] for x in rep["no_span"]] == ["A4"]
    assert rep["coverage"]["effect"] == {"facts": 9, "with_span": 9, "once+name": 7, "once+unit": 0,
                                         "several+name": 2, "verifier": 0, "share": 1.0}


def test_cache_only_reuses_answers_and_never_calls(tmp_path: Path) -> None:
    seed, ds = domain(tmp_path)
    s = settings(tmp_path, verifier="v")
    numspans.run(seed, ds, llm.Caller(s, backend=honest()))
    first = json.loads((seed / "paperfactory.json").read_text())["papers"][0]["numbers"]

    def boom(*_):
        raise AssertionError("cache_only made a model call")
    rep = numspans.run(seed, ds, llm.Caller(s, backend=boom), cache_only=True)
    assert json.loads((seed / "paperfactory.json").read_text())["papers"][0]["numbers"] == first
    assert rep["not_cached"] == 0
    rep = numspans.run(seed, ds, llm.Caller(settings(tmp_path / "fresh", verifier="v"), backend=boom),
                       cache_only=True)
    assert rep["not_cached"] == 1 and "A3" in {x["id"] for x in rep["no_span"]}


def test_ci_mismatch_reported() -> None:
    rows = [row(i, "effect", "1", subject="S1", relation="r", object="M1", stat=st)
            for i, st in (("e", "estimate"), ("l", "ci_low"), ("h", "ci_high"))]
    by = {"e": {"groups": [[0, 5]]}, "l": {"groups": [[0, 5]]}, "h": {"groups": [[9, 12]]}}
    assert ci_mismatches(rows, by)[0]["shared"] is False
    by["h"] = {"groups": []}
    assert ci_mismatches(rows, by)[0]["with_span"] == ["ci_low", "estimate"]    # one bound without a span


def test_same_model_recorded(tmp_path: Path) -> None:
    seed, ds = domain(tmp_path)
    rep = numspans.run(seed, ds, llm.Caller(settings(tmp_path, verifier="claude-haiku-5-5"), backend=honest()))
    ind = rep["independence"]
    assert ind["same_model_as_a_writer"] and ind["papers_same_model"] == 1 and ind["verifier_spans_same_model"] == 1
    assert rep["controls_by_writer"]["same_model"] == {"items": 1, "no": 1, "yes": 0, "no_answer": 0}


def test_hard_control_right_value_wrong_subject(tmp_path: Path) -> None:
    """Kor 2021 and Bel 2019 share 65.2: each one's row is a hard control for the other's fact."""
    seed, ds = domain(tmp_path)
    rep = numspans.run(seed, ds, llm.Caller(settings(tmp_path, verifier="v"), backend=honest()))
    h = rep["hard_controls"]
    assert h["candidates"] >= 1 and h["items"] == h["candidates"]
    assert h["yes"] == h["items"] and len(h["accepted"]) == h["items"]   # the value-only fake verifier fails them


def test_clopper_pearson_and_collisions() -> None:
    assert numspans.clopper_pearson(0, 100) == [0.0, 0.0362]
    assert numspans.clopper_pearson(1, 400) == [0.0001, 0.0138]
    assert numspans.clopper_pearson(5, 5)[1] == 1.0
    c = numspans.value_collisions([row("a", "attr", "3", subject="S1"), row("b", "attr", "3.0", subject="S2"),
                                   row("c", "attr", "3", subject="S1"), row("d", "attr", "4", subject="S1")])
    assert c == {"values": 1, "facts": 3, "pairs": 3, "different_subject_pairs": 2}


def test_report_keeps_both_hard_rules(tmp_path: Path) -> None:
    seed, ds = domain(tmp_path)
    rep = numspans.run(seed, ds, llm.Caller(settings(tmp_path, verifier="v"), backend=honest()))
    assert rep["hard_controls_v1"]["why_replaced"] and rep["hard_controls"]["rule"].startswith("current")
    assert rep["error_rates"]["hard_control"]["n"] == rep["hard_controls"]["items"]
    assert rep["value_collisions"]["pairs"] >= 1                          # 65.2 twice in P-01


def test_code_audit_flags_collisions(tmp_path: Path) -> None:
    """65.2 is both Kor 2021's and Bel 2019's baseline: both spans are collisions and are checked on the marked cell."""
    seed, ds = domain(tmp_path)
    b = honest()
    rep = numspans.run(seed, ds, llm.Caller(settings(tmp_path, verifier="v"), backend=b))
    a = rep["code_audit"]
    flagged = {x["fact"]: x for x in a["collision_spans"]}
    assert flagged["E1"]["collides_with"] == ["E1b"] and flagged["E1"]["span_text"].endswith(": 65.2")
    assert a["collision"]["n"] == len(a["collision_spans"]) and a["collision"]["errors"] == 0
    assert a["several_name_random"]["n"] == a["several_name_random"]["of"]   # fewer than 100: all of them
    audit_call = [it for call in b.seen for it in call if "marked" in it]
    assert audit_call and rep["verifier_error_quote"]["rate"] == "1.3% [0.4%, 2.9%]"
