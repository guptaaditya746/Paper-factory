"""number-spans (paperfactory/numspans.py) on a hand-made paper: formats, code rules, verifier, controls, cache."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

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

Seeds were 5 in every study except one. Pooled across 2 studies, Ragrer decoding gained 8.1 percent (95% CI 5.4 to 10.8).
Bel 2019 reported a baseline of 71.3 points. Another baseline was 71.3 in a pilot.
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
        row("E4", "effect", "71.3", subject="S2", relation="reports_gain", object="M1", stat="baseline"),
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
        def has(value: str, it: dict) -> bool:
            return canon(value.split()[0]) in {v for _, _, v in numbers_in(it.get("marked") or it["passage"])}
        if items and "about" in items[0]:      # three-field verifier: subject and stat always yes here
            ans = [{"id": it["id"], "subject": "yes", "stat": "yes", "value": "yes" if has(it["value"], it) else "no"}
                   for it in items]
        else:
            ans = [{"id": it["id"], "verdict": "yes" if has(it["fact"].split(" = ")[1], it) else "no"}
                   for it in items]
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
    got_notes = {x["id"]: x["notes"] for x in rep["no_span"]}
    assert how["A1"] == ["several+name"] and len(got["A1"]["spans"]) == 1      # "3 runs failed" names no one
    assert how["A2"] == ["once+name"]                                          # 3,050 = 3050
    assert how["P1"] == []                  # rule 3: says pooled but no study count, no 2 studies -> gate: no method
    assert how["E1"] == ["verifier"]        # rule 4: E1b has the same value -> the verifier, only Kor 2021's row
    assert how["E2"] == how["E2l"] == how["E2h"] == ["once+name"]              # unicode minus
    assert how["E3"] == ["once+name"]                                          # 9.0 = 9
    assert how["A3"] == []                                   # unit "n" is no evidence; the gate: no subject named
    assert any("gate (subject not named)" in n for n in got_notes["A3"])
    assert how["E4"] == ["verifier"] and MD[slice(*got["E4"]["spans"][0])].startswith("Bel 2019")   # not the pilot
    assert got["A4"]["spans"] == []
    s, e = got["P2"]["spans"][0]                                           # not Kor 2021's own 10.8 in its row
    assert how["P2"] == ["several+name"] and MD[s:e].startswith("Pooled across 2")    # rule 3, upper-bound role
    s, e = got["E1"]["spans"][0]
    assert MD[s:e] == "65.2" and MD.rindex("Kor 2021", 0, s) > MD.index("| Study")   # raw offsets, Kor's row
    c = rep["controls"]
    assert c["items"] >= 1 and c["no"] == c["items"] and c["yes"] == 0
    other = rep["controls_by_writer"]["other_model"]                      # verifier "v" did not write bench20h
    assert other["items"] == rep["controls"]["items"] and rep["controls_by_writer"]["same_model"]["items"] == 0
    assert man["papers"][0]["writer"] == "claude-haiku-5-5" and man["number_spans"]["verifier_model"] == "v"
    assert man["number_spans"]["independence"]["same_model_as_a_writer"] is False
    # Lof 2020: three cells of one row = shared; P1 has no span now, so no pair to compare
    assert rep["ci_mismatch"] == []
    assert [x["id"] for x in rep["no_span"]] == ["P1", "A3", "A4"]
    eff = rep["coverage"]["effect"]          # Lof 2020's rows name no method: an effect needs both -> verifier
    assert eff["facts"] == eff["with_span"] == 10 and eff["verifier"] == 6


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
    assert ind["same_model_as_a_writer"] and ind["papers_same_model"] == 1
    assert ind["verifier_spans_same_model"] == rep["coverage"]["effect"]["verifier"] + rep["coverage"]["pool"]["verifier"]
    c = rep["controls_by_writer"]["same_model"]
    assert c["items"] >= 1 and c["no"] == c["items"]


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
    lo, hi = numspans.clopper_pearson(2900, 3000)    # plain float terms overflowed above n ~ 1000
    assert 0.957 < lo < 0.967 < hi < 0.973
    assert numspans.clopper_pearson(3000, 3000)[1] == 1.0 and numspans.clopper_pearson(0, 3000)[0] == 0.0
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
    """65.2 is both Kor 2021's and Bel 2019's baseline: rule 4 sends both to the verifier (marked cell), so no
    collision span is left for code; the audit has nothing to flag."""
    seed, ds = domain(tmp_path)
    b = honest()
    rep = numspans.run(seed, ds, llm.Caller(settings(tmp_path, verifier="v"), backend=b))
    a = rep["code_audit"]
    assert a["collision_spans"] == [] and a["collision"]["n"] == 0
    notes = {x["id"]: x for x in json.loads((seed / "numspans.json").read_text())["no_span"]}
    assert "E1" not in notes                                                   # placed, by the verifier
    marked = [it for call in b.seen for it in call if "marked" in it]
    assert any(it["marked"].endswith(": 65.2") for it in marked)
    assert rep["verifier_error_quote"]["rate"] == "1.3% [0.4%, 2.9%]"


def test_value_roles_and_gate() -> None:
    t = "The pooled estimate is -3.7%, with a 95% CI from -5.7% to -1.8%. HR 1.01 (95% CI 0.80 to 1.26)."
    assert numspans.value_roles(t, "-3.7") == {"estimate"}
    assert numspans.value_roles(t, "-5.7") == {"ci_low"} and numspans.value_roles(t, "-1.8") == {"ci_high"}
    assert numspans.value_roles(t, "0.8") == {"ci_low"} and numspans.value_roles(t, "1.26") == {"ci_high"}
    assert numspans.value_roles("0.63 (0.46 to 0.86)", "0.86") == {"ci_high"}
    assert numspans.value_roles("4.4", "4.4", "95% CI upper bound (%)") == {"ci_high"}
    p = numspans.Place(0, 4, "1.02 (0.85 to 1.22)", "", "", "", "Agent: drug B; HR: 1.02 (0.85 to 1.22)",
                       header="HR, estimate (lower to upper)")
    r = row("x", "pool", "1.02", "HR", subject="AT1", stat="ci_high", studies=[])
    assert numspans.gate(r, p, {"AT2"}) == "subject not named"
    assert numspans.gate(r, p, {"AT1"}).startswith("stat: the value is estimate")
    assert numspans.gate(r | {"stat": "estimate"}, p, {"AT1"}) is None


def test_hard_controls_use_frozen_items(tmp_path: Path) -> None:
    seed, ds = domain(tmp_path)
    s = settings(tmp_path, verifier="v")
    numspans.run(seed, ds, llm.Caller(s, backend=honest()))
    frozen = {"v2_seen": [{"paper": "P-01", "fact": "E1", "other": "E1b", "same_model": False, "answer": True,
                           "context": "columns: Study | Method | Baseline accuracy (%) | Difference (%), estimate "
                                      "(lower to upper)", "text": "E1 = 65.2 %",
                           "passage": "Study: Bel 2019; Method: Ragrer decoding; Baseline accuracy (%): 65.2; "
                                      "Difference (%), estimate (lower to upper): -1.9 (-3.7 to -0.5)"}],
              "v1_row": []}
    (seed / "numspans_hard_frozen.json").write_text(json.dumps(frozen))
    rep = numspans.run(seed, ds, llm.Caller(s, backend=honest()))
    h = rep["hard_controls"]
    assert h["frozen"] and h["yes"] == 1                              # the old answer, kept
    assert h["after_fix"]["gate_alone"]["rejected"] == 1                # Kor 2021 is not named in Bel's row
    assert h["after_fix"]["gate_and_verifier3"]["no"] == 1
    assert (seed / "numspans_audit_frozen.json").is_file()


def test_frozen_cell_picks_the_cell_with_the_number() -> None:
    """Every cell of a table row shares (context, passage); the frozen item must get the cell that holds the value."""
    from paperfactory import packs
    from paperfactory.text import alias_table, split_rules
    f, w = packs.field("llm"), {"entities": ENTS}
    P = numspans.places(MD, split_rules(f, w))
    by = {r["id"]: r for r in ROWS}
    row_bel = next(q for q in P if q.row and "Bel 2019" in q.passage)
    h = {"fact": "E1", "other": "E1b", "context": row_bel.context, "passage": row_bel.passage}
    i, how = numspans.frozen_cell(h, "v2_seen", P, by, alias_table(w, f), {})
    assert "65.2" in P[i].marked and how == "value cell"                # not the row's last cell (-1.9 ...)
    assert numspans.frozen_cell(h | {"passage": "nowhere"}, "v2_seen", P, by, alias_table(w, f), {}) == (None, None)


def test_rules_v2_labels_years_counts_pools() -> None:
    r = row("c", "attr", "2", "annotators", subject="S1", attr="human_raters")
    assert numspans.value_hits("Table 2 lists 2 raters.", "2", r) == [(14, 15)]            # not "Table 2"
    r1 = row("k", "attr", "1", "countries", subject="S1", attr="countries")
    assert numspans.value_hits("The interval includes 1.00 in 1 country.", "1", r1) == [(30, 31)]   # 1.00 is no count
    r2 = row("n", "attr", "1200", "n", subject="S1", attr="patients")
    assert numspans.value_hits("It enrolled 1,200 patients.", "1200", r2) == [(12, 17)]
    r3 = row("y", "attr", "2021", "n", subject="S1", attr="sites")
    assert numspans.value_hits("Kor 2021 used 2021 sites (2021).", "2021", r3) == [(14, 18)]
    ry = row("yy", "attr", "2021", "", subject="S1", attr="publication_year")
    assert len(numspans.value_hits("Kor 2021 used 2021 sites.", "2021", ry)) == 2               # a year fact keeps years
    pool = row("p", "pool", "0.77", "HR", subject="M1", stat="estimate", studies=["S1", "S2", "S3"])
    kinds = {"S1": "trial", "S2": "trial", "S3": "trial"}
    assert numspans.pooled_check(pool, "The pooled HR across 3 trials is 0.77.", set(), kinds) is None
    assert numspans.pooled_check(pool, "The pooled HR across 2 trials is 0.77.", set(), kinds).startswith("pool: states")
    assert numspans.pooled_check(pool, "Pooled, Kor 2021 and Bel 2019 give 0.77.", {"S1", "S2"}, kinds) is None
    assert numspans.pooled_check(pool, "Trial Kor 2021 found 0.77.", {"S1"}, kinds).startswith("pool: place does not")


def test_cell_check_both_orientations() -> None:
    from paperfactory import packs
    from paperfactory.text import alias_table
    tab = alias_table({"entities": ENTS}, packs.field("llm"))
    r = row("j", "attr", "3", "judges", subject="S1", attr="judges", label="Kor 2021: size of the bench")
    claim = numspans.Place(0, 1, "3", "", "", "", "", ["Kor 2021", "3", "3"], header="Number of claimants (claimants)")
    bench = numspans.Place(0, 1, "3", "", "", "", "", ["Kor 2021", "3", "3"], header="Size of the bench (judges)")
    assert numspans.cell_check(r, claim, {"S1"}, tab).startswith("cell: quantity")
    assert numspans.cell_check(r, bench, {"S1"}, tab) is None
    side = numspans.Place(0, 1, "3", "", "", "", "", ["Judges sitting", "5", "3"], header="Kor 2021 (judges)")
    assert numspans.cell_check(r, side, {"S1"}, tab) is None                                # sideways table


REJECTED = Path(__file__).parent / "data" / "numspans_rejected_v1.json"
DATA = Path(__file__).resolve().parents[5] / "data"


@pytest.mark.skipif(not (DATA / "seed" / "bench_law").is_dir(), reason="needs the parent repo's data/")
def test_regression_172_rejected_spans_not_placed_by_code() -> None:
    """The 172 code-made spans the first audit rejected, of which the fixed judge also calls 159 wrong: today's code
    rules must not place any of those 159 by code (the verifier may still, after its gate). 4 llm spans were judged
    wrong only for the old label "...: accuracy" (the value is the change); after the label fix they are right."""
    from paperfactory import packs
    from paperfactory.text import alias_table, defined_terms, normalize_paper, split_rules
    items = [x for x in json.loads(REJECTED.read_text())["items"] if x["fixed_judge_wrong"]]
    left = []
    for dom in sorted({x["domain"] for x in items}):
        seed = DATA / "seed" / dom
        man = json.loads((seed / "paperfactory.json").read_text())
        ents = json.loads((seed / "entities.json").read_text())
        rows = [json.loads(x) for x in (seed / "facts.jsonl").read_text().splitlines()]
        rows = [r for r in rows if r["what"] in ("attr", "effect", "pool")]
        f, w = packs.field(man["field"]), {"entities": ents}
        base = alias_table(w, f)
        for pid in sorted({x["paper"] for x in items if x["domain"] == dom}):
            md = (DATA / "dataset" / dom / pid / "paper.md").read_text()
            tab = alias_table(w, f, defined_terms(normalize_paper(md).text, base, f))
            mine = [r for r in rows if r["paper"] == pid]
            recs, extra = numspans.span_paper(md, mine, tab, split_rules(f, w), _NoVerifier(), pid,
                                              {k: e["kind"] for k, e in ents.items()})
            P = extra["P"]
            placed = {(rec["id"], P[i].marked or P[i].text) for rec in recs
                      for i, how in zip(rec["places"], rec["found_by"]) if how != "verifier"}
            left += [x for x in items if x["domain"] == dom and x["paper"] == pid
                     and (x["fact"], x["span_text"]) in placed]
    assert left == [], f"{len(left)} of {len(items)} still placed by code, e.g. {left[:3]}"


class _NoVerifier:
    """No model: every verifier answer is missing (only code rules place spans)."""
    def ask3(self, items, step):
        return [None] * len(items)


def test_mark_report_tags_only_judged_rates() -> None:
    rep = {"verifier_model": "claude-haiku-5-5", "error_rates": {"v2": {"errors": 1, "n": 9, "rate": 0.11}},
           "code_error_rate_quote": {"rate": "3.9%", "n": 306}, "fact_recall": {"x": {"recall": 0.9}},
           "coverage": {"rate": 0.5}}
    numspans.mark_report(rep)
    assert rep["error_rates"]["v2"]["judge_note"] == numspans.SAME_MODEL_NOTE == "same model as writer"
    assert rep["code_error_rate_quote"]["judge"] == "claude-haiku-5-5"
    assert "judge" not in rep["coverage"] and "judge" not in rep["fact_recall"]["x"]
    assert "judge_note" not in numspans.mark_judged({"rate": 1}, "claude-sonnet-5-5")
