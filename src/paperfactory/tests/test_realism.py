"""Realism distribution check (metric 3): preparation, properties and statistics. No model calls."""
from __future__ import annotations

import itertools
import json
import math

import pytest

from paperfactory import packs, realism
from paperfactory.text import alias_table, split_rules

LAW = packs.field("law")
RULES = split_rules(LAW, {"entities": {}})
WORLD = {"entities": {
    "C1": {"kind": "court", "label": "Tetre District Court", "alt": ["District Court"], "attrs": {}},
    "P1": {"kind": "provision", "label": "section 1 of the Tetre Data Privacy Act", "alt": [], "attrs": {}},
    "K1": {"kind": "case", "label": "Laper v. Dravun Media", "alt": ["Laper"], "attrs": {}}}}


# ------------------------------------------------------------------ invented-world sentences
@pytest.mark.parametrize("s", [
    "This article asks how courts in the invented Tetre system have treated user data.",
    "It reads six decisions alongside two provisions of the Act, with no real legal system assumed.",
    "The jurisdiction is invented for this study.",
    "Its courts, statutes and party names do not correspond to any real legal system.",
    "All names are invented, and the findings are confined to them.",
    "The study examines the Tetre legal system, a hypothetical jurisdiction constructed for analysis.",
    "These invented trials and the associated drugs serve as a framework.",
    "A notable discrepancy has emerged between two recent trials in this synthetic dataset.",
    "The data presented are not derived from real-world clinical trials.",
])
def test_invented_world_sentences_are_found(s: str) -> None:
    assert realism.invented_world(s)


@pytest.mark.parametrize("s", [
    "Rotating beam tests replicate real-world loading conditions more closely.",
    "Synthetic data for instruction tuning has grown quickly since 2022.",
    "The technique was invented by Smith in 1998 and later refined.",
    "Retrieval was invented in 1960 for library catalogues.",
    "Consider a hypothetical patient who misses two doses.",
    "The court relied on section 1 of the Act in a real dispute over erasure.",
])
def test_ordinary_sentences_are_not_flagged(s: str) -> None:
    assert realism.invented_world(s) is None


# ------------------------------------------------------------------ citations
@pytest.mark.parametrize("s,n", [
    ("Prices rose sharply [13].", 1),
    ("Governments lack negotiating power [16, 17] and data [52-54].", 2),
    ("Risk relates to gender^(4-5)^ and access^(8)^.", 2),
    ("Earlier work agrees (Smith et al., 2019; Lee 2020).", 1),
    ("Smith et al. (2019) found the opposite.", 1),
    ("The Braliram 2023 study reported a gain.", 1),
    ("The Brudes Lab 2024 study measured corrosion.", 1),
    ("Laper v. Dravun Media (Tetre District Court, 2017) relies on section 1.", 1),
    ("Three trials are included: DRELA-3 (2019) and NESENAM-7 (2025).", 2),
    ("In 2019 the court ruled; in March 2020 it ruled again.", 0),
    ("The pooled hazard ratio was 0.71 (95% CI 0.53 to 0.96).", 0),
    ("Table 1 gives the length of each decision.", 0),
])
def test_citations(s: str, n: int) -> None:
    assert len(realism.find_citations(s)) == n


# ------------------------------------------------------------------ names
def _gen(s: str) -> list[str]:
    return [s[a:b] for a, b in realism.generic_names(s)]


def test_generic_names() -> None:
    assert _gen("The BRUMON-6 trial and the later ZIMOR-3 trial both examined tukrapurtinib.") == ["BRUMON-6",
                                                                                                    "ZIMOR-3"]
    assert _gen("The reports cover two models, Thipom-7B and Nimel-13B.") == ["Thipom-7B", "Nimel-13B"]
    assert _gen("Courts applied section 1 of the Tetre Data Privacy Act to insurers.") == [
        "section 1 of the Tetre Data Privacy Act"]
    assert _gen("Laper relies on the statute, with a 95% CI that is wide.") == []      # sentence start, CI
    assert _gen("However, the National Institute for Health and Care Excellence disagreed.") == [
        "National Institute for Health", "Care Excellence"]


def test_world_and_generic_masking_agree_on_a_simple_sentence() -> None:
    table = alias_table(WORLD, LAW)
    s = "Courts read section 1 of the Tetre Data Privacy Act broadly, and the Tetre District Court may agree."
    w = realism.mask_sentence(s, realism.world_names(table))
    g = realism.mask_sentence(s, realism.generic_names)
    assert w.text == g.text == "Courts read [NAME] broadly, and the [NAME] may agree."
    assert (w.words, w.hedges, w.numbers, w.cites) == (9, 1, 0, 0)


def test_mask_sentence_counts() -> None:
    m = realism.mask_sentence("The Braliram 2023 study suggests a gain of 9.0 % (95% CI 7.3 to 10.8) [4].",
                              realism.generic_names)
    assert m.text == "The [CIT] study suggests a gain of [NUM] % ([NUM]% CI [NUM] to [NUM]) [CIT] ."
    assert (m.cites, m.numbers, m.hedges) == (2, 4, 1)
    assert m.words == 14          # placeholders count as one word each


# ------------------------------------------------------------------ paragraphs
PAPER = """# A title that is a heading

## Abstract

This abstract paragraph is long enough to count but sits in a skipped section, so it must not be measured at all here.

## Introduction

The first body paragraph has enough words to be measured. It may suggest a pattern that later work could test with care [3]. The jurisdiction is invented for this study.

Too short to count.

Table 1: Pages of each decision in the case set, with one column per decision and many more words to pass the limit.

| A | B |
|---|---|
| 1 | 2 |

- a list item that is long enough to count but is a list item, so the body filter leaves it out of the measure.

## References

Smith J. A long reference entry that has more than thirty words so that only the section rule can remove it from the body paragraphs of this paper, which it does.
"""


def test_body_paragraphs_keep_only_body_text() -> None:
    paras = realism.body_paragraphs(PAPER)
    assert [s for s, _ in paras] == ["Introduction"]
    row, dropped = realism.measure_paragraph(paras[0][1], RULES, realism.generic_names)
    assert [s for _, s in dropped] == ["The jurisdiction is invented for this study."]
    assert row["sentences"] == 2 and row["cites"] > 0 and row["hedges"] > 0


def test_body_paragraphs_cap_is_evenly_spaced() -> None:
    md = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(100))
    paras = realism.body_paragraphs(md, max_paras=5)
    assert [p.split()[1] for _, p in paras] == ["0", "25", "50", "74", "99"]


# ------------------------------------------------------------------ statistics
def _brute_ks_p(n: int, m: int, d: float) -> float:
    vals = list(range(n + m))
    hits = total = 0
    for xs in itertools.combinations(vals, n):
        ys = [v for v in vals if v not in xs]
        total += 1
        hits += realism.ks_2samp(list(xs), ys)["D"] >= d - 1e-9
    return hits / total


def test_ks_exact_matches_brute_force() -> None:
    x, y = [0.1, 0.4, 0.5, 0.9], [0.2, 0.6, 0.7, 0.8, 1.2]
    r = realism.ks_2samp(x, y)
    assert r["method"] == "exact"
    assert r["D"] == pytest.approx(0.55)
    assert r["p"] == pytest.approx(_brute_ks_p(4, 5, r["D"]))
    assert realism.ks_2samp([1, 2, 3], [4, 5, 6])["p"] == pytest.approx(2 / 20)       # 2 of C(6,3) paths
    assert realism.ks_2samp([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])["p"] == pytest.approx(2 / 252)


def test_ks_ties_and_asymptotic() -> None:
    assert realism.ks_2samp([0, 0, 1], [0, 1, 1])["D"] == pytest.approx(1 / 3)
    assert realism._kolmogorov_sf(1.3581) == pytest.approx(0.05, abs=2e-4)
    assert realism._kolmogorov_sf(1.0) == pytest.approx(0.2699996716773546, abs=1e-9)
    x = [i / 150 for i in range(150)]
    y = [i / 150 + 0.1 for i in range(150)]
    r = realism.ks_2samp(x, y)
    assert r["method"] == "asymptotic" and r["D"] == pytest.approx(0.1, abs=0.01)
    # same D at the largest exact size: the two methods agree closely
    assert realism.ks_2samp(x[:100], y[:100])["p"] == pytest.approx(r["p"], abs=0.25)


@pytest.mark.parametrize("x,df,p", [(3.841458820694124, 1, 0.05), (5.991464547107979, 2, 0.05),
                                    (10.0, 4, 0.04042768199451279), (1.0, 1, 0.31731050786291415),
                                    (124.34211340400407, 100, 0.05), (0.0, 3, 1.0), (40.0, 2, 2.061153622438558e-09)])
def test_chi2_sf_known_values(x: float, df: int, p: float) -> None:
    assert realism.chi2_sf(x, df) == pytest.approx(p, rel=1e-6)


def test_chi2_cramers_v() -> None:
    r = realism.chi2_cramers_v({"attr": 10, "effect": 20}, {"attr": 20, "effect": 10})
    assert r["chi2"] == pytest.approx(20 / 3)
    assert r["V"] == pytest.approx(1 / 3)
    assert r["p"] == pytest.approx(realism.chi2_sf(20 / 3, 1)) and r["p"] == pytest.approx(0.009823, abs=1e-6)
    same = realism.chi2_cramers_v({"a": 5, "b": 10, "c": 15}, {"a": 10, "b": 20, "c": 30, "d": 0})
    assert same["chi2"] == pytest.approx(0) and same["p"] == pytest.approx(1) and same["kinds"] == ["a", "b", "c"]
    with pytest.raises(ValueError):
        realism.chi2_cramers_v({"a": 3}, {"a": 4})


def test_holm() -> None:
    assert realism.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert realism.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_compare_reports_effect_first_and_holm_over_all_tests() -> None:
    real = [{"paper": f"r{i}", "sent_len": 20 + i % 7, "hedges": i % 3, "cites": 2.0, "numbers": i % 5}
            for i in range(30)]
    synth = [{"paper": f"s{i}", "sent_len": 22 + i % 5, "hedges": 0.0, "cites": 0.0, "numbers": i % 4}
             for i in range(30)]
    t4 = realism.compare(real, synth)
    assert [r["holm_over"] for r in t4] == [4] * 4
    t5 = realism.compare(real, synth, {"attr": 30, "effect": 30}, {"attr": 10, "effect": 50})
    assert [r["property"] for r in t5][-1] == "fact_kind_share" and t5[0]["holm_over"] == 5
    assert list(t5[0])[:4] == ["property", "test", "effect", "effect_size"]
    assert all(r["p_holm"] >= r["p"] for r in t5)
    assert t5[2]["effect_size"] == 1.0          # cites: all real 2.0, all synth 0.0


# ------------------------------------------------------------------ run on a tiny data folder
def test_run_synth(tmp_path) -> None:
    seed, ds = tmp_path / "seed" / "bench_x", tmp_path / "dataset" / "bench_x"
    (ds / "X-01").mkdir(parents=True)
    seed.mkdir(parents=True)
    (seed / "paperfactory.json").write_text(json.dumps({"field": "law", "papers": [{"id": "X-01"}]}))
    (seed / "entities.json").write_text(json.dumps(WORLD["entities"]))
    body = ("Laper v. Dravun Media (Tetre District Court, 2017) relies on section 1 of the Tetre Data Privacy Act. "
            "The court may have read the provision broadly in 2017, and the reading could matter for 12 insurers. "
            "All names in this study are invented.")
    (ds / "X-01" / "paper.md").write_text(f"# T\n\n## Abstract\n\nThe setting is an invented country.\n\n## Part\n\n{body}\n")
    out, inv = realism.run_synth(seed, ds)
    assert out["summary"]["paragraphs"] == 1 and inv["sentences"] == 2 and inv["papers_with_any"] == 1
    assert {x["section"] for x in inv["items"]} == {"Abstract", "Part"}
    assert set(out["masking_check"]["properties"]) == set(realism.PROPS)
    assert not math.isnan(out["summary"]["sent_len"]["mean"])
