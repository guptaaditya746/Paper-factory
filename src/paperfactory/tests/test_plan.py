"""Content plan: papers of one world differ in columns, tables, fact order and fact placement (no model calls)."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

from paperfactory import materials, packs, plan, quant, style, world
from paperfactory.materials import _row_prefix

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "external" / "paperorchestra"))
from utils import fact_check  # noqa: E402

FIELDS = {"medicine": "synthesis", "llm": "synthesis", "materials": "synthesis", "law": "argument"}


def _world(f: packs.Field, g: packs.Genre, n: int = 24) -> dict:
    """n papers with 2 to 5 claims and 1 to 4 decoys over a pool of subjects and objects (ids only)."""
    kind = f.subject_kinds[0]
    rels = list(f.relations)
    okinds = sorted({k for r in rels for k in f.relations[r]["object_kinds"] if k not in f.subject_kinds})
    ents = {f"S{i}": {"kind": kind, "label": f"Study S{i}", "alt": [], "attrs": {}} for i in range(3 * n)}
    ents |= {f"O{k}{j}": {"kind": k, "label": f"Object {k} {j}", "alt": [], "attrs": {}}
             for k in okinds for j in range(6)}
    papers, counts = {}, world.plan_counts(g, f, 1, "T", n)
    for i, (c, d) in enumerate(counts):
        facts = []
        for j in range(c + d):
            r = rels[(i + j) % len(rels)]
            ok = [k for k in f.relations[r]["object_kinds"] if k in okinds] or okinds
            facts.append([f"S{(3 * i + j) % (3 * n)}", r, f"O{ok[0]}{(i + j // 2) % 6}"])
        papers[f"P-{i + 1:02d}"] = {"title_hint": "t", "issue": "the issue", "thesis": "the thesis",
                                    "claims": facts[:c], "decoys": facts[c:]}
    w = {"entities": ents, "papers": papers}
    w["numbers"] = quant.make(w, f, 1)
    return w


def _plans(name: str):
    f, g = packs.field(name), packs.genre(FIELDS[name])
    w = _world(f, g)
    cards = style.cards(list(w["papers"]), f, g, 1)
    out = {}
    for pid in w["papers"]:
        tmpl = (g.folder / "templates" / f"{cards[pid]['template']}.md").read_text(encoding="utf-8")
        cp = plan.make(w, f, g, pid, tmpl, 1)
        out[pid] = (cp, *plan.number_rows(w, f, g, pid, cp, _row_prefix(f, pid)), cards[pid])
    return f, g, w, out


def test_leaves_of_a_template() -> None:
    t = "# {TITLE}\n\n## Abstract\n\n## Introduction\n\n## Results\n\n### A\n\n### B\n\n## Conclusions\n"
    assert plan.leaves(t) == ["Introduction", "Results > A", "Results > B", "Conclusions"]


@pytest.mark.parametrize("name", FIELDS)
def test_papers_of_one_world_differ(name: str) -> None:
    f, g, w, plans = _plans(name)
    n = len(plans)
    sizes = {(len(p["claims"]), len(p["decoys"])) for p in w["papers"].values()}
    assert len(sizes) >= 6                                            # v2: one size (3 claims, 3 decoys)
    columns = Counter(tuple((c["attr"], c["label"]) for cols in cp["attrs"].values() for c in cols)
                      for cp, *_ in plans.values())
    assert len(columns) >= n * 0.8                                    # v2: one column set per field
    titles = Counter(tuple(t["title"] for t in tables) for _, _, tables, _ in plans.values())
    assert max(titles.values()) <= max(2, n // 6)                     # v2: the same two titles in every paper
    assert len({t["design"] for _, _, tables, _ in plans.values() for t in tables}) >= 3
    assert len({len(tables) for _, _, tables, _ in plans.values()}) >= 2
    assert {cp["lean"] for cp, *_ in plans.values()} == set(plan.LEANS)
    # where the planted facts go: no section takes most of them across the world
    where = Counter(x["section"] for cp, *_ in plans.values() for x in cp["placement"])
    assert max(where.values()) / sum(where.values()) < 0.4


@pytest.mark.parametrize("name", FIELDS)
def test_every_plan_is_complete_and_the_number_check_can_read_its_notes(name: str) -> None:
    f, g, w, plans = _plans(name)
    for pid, (cp, rows, tables, card) in plans.items():
        p = w["papers"][pid]
        assert sorted(map(tuple, cp["order"])) == sorted(map(tuple, p["claims"] + p["decoys"]))
        assert all(x["section"] in cp["sections"] for x in cp["placement"])
        assert any(x["abstract"] for x in cp["placement"])
        assert len(tables) >= g.raw["min_tables"], (pid, cp["attrs"])
        ids = [r for t in tables for r in t["rows"]]
        assert sorted(ids) == sorted(r["id"] for r in rows)           # every number in exactly one table
        assert all(t["section"] in cp["sections"] for t in tables)
        idea, log = materials.materials_v3(w, f, g, pid, card, cp)
        facts = fact_check.parse_facts(log)
        assert [(x["id"], x["value"], x["placement"]) for x in facts] == [(r["id"], r["value"], "table") for r in rows]
        assert "State it first in" in log and "## " in idea
        for rel in f.relations.values():                              # the outline carries no relation words
            outline = log.split("Outline (what goes in each section")[1].split("\n## ")[0]
            assert rel["words"] not in outline


def test_the_same_seed_gives_the_same_plan() -> None:
    a, b = _plans("medicine")[3], _plans("medicine")[3]
    assert {k: v[0] for k, v in a.items()} == {k: v[0] for k, v in b.items()}
    assert {k: v[2] for k, v in a.items()} == {k: v[2] for k, v in b.items()}


def test_fit_counts_trims_and_leaves_slots() -> None:
    q = {"focus": ["A1", "A2"], "claims": [["A1", "r", "B1"], ["A2", "r", "B1"], ["A2", "r", "B2"], ["A1", "r", "B3"]],
         "decoys": [["A3", "r", "B1"]], "conflict": [["A1", "r", "B3"], ["A2", "r", "B1"]]}
    world.fit_counts(q, 2, 3)
    assert q["claims"] == [["A2", "r", "B1"], ["A1", "r", "B3"]]      # the conflict facts stay
    assert q["decoys"] == [["A3", "r", "B1"], ["", "", ""], ["", "", ""]]
    q = {"focus": ["A1", "A2"], "claims": [["A1", "r", "B1"], ["A2", "r", "B1"]], "decoys": []}
    world.fit_counts(q, 1, 0)
    assert len(q["claims"]) == 2                                      # never a focus subject's only claim
    assert world.count_range(3) == (3, 3) and world.count_range([2, 5]) == (2, 5)
