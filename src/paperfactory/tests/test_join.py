"""join.py: two worlds of one field -> one world (no model calls)."""
from __future__ import annotations

from paperfactory import join, packs, world

F, G = packs.field("law"), packs.genre("argument")


def _court(i: int) -> dict:
    return {"kind": "court", "label": f"Tetre {['District Court', 'Court of Appeal', 'Supreme Court'][i - 1]}",
            "alt": [], "attrs": {"level": i}}


def _world(dec: dict, prov: dict, claims: list, decoys: list) -> dict:
    ents = {"C1": _court(1), "C2": _court(2), "C3": _court(3)} | dec | prov
    return {"field": "law", "genre": "argument", "entities": ents,
            "papers": {"LAW-01": {"title_hint": "t", "issue": "i", "thesis": "th", "claims": claims,
                                  "decoys": decoys}}}


def test_same_thing_unified_and_accident_renamed():
    a = world.copy.deepcopy(_world(
        {"D1": {"kind": "decision", "label": "Sadepo v. Rubrothes Bank", "alt": [],
                "attrs": {"court": "C1", "year": 2010, "facts": "x"}}}, {}, [], []))
    b = _world({"AD1": {"kind": "decision", "label": "Sadepo v. Rubrothes Bank", "alt": [],
                        "attrs": {"court": "C1", "year": 2012, "facts": "y"}}}, {}, [], [])
    w = join.join(a, b, F, 1, "b")
    assert set(w["joins"]["unified"]) == {"b:C1", "b:C2", "b:C3"}
    assert w["entities"]["D1"]["label"] == "Sadepo v. Rubrothes Bank"
    new = w["entities"]["AD1"]["label"]
    assert "Sadepo" not in new and "Rubrothes" not in new and new.endswith("Bank")
    assert w["entities"]["AD1"]["attrs"]["court"] == "C1"
    assert w["papers"]["LAW-02"]["source"] == "b:LAW-01"
    assert a["entities"].keys() == {"C1", "C2", "C3", "D1"}       # inputs unchanged


def test_fresh_word_never_inside_a_used_one():
    namer = join.names.Namer(1, F)
    for _ in range(50):
        new = join._fresh(namer, {"tukrapurtinib"})
        assert new not in "tukrapurtinib" and "tukrapurtinib" not in new


def test_clashes_listed():
    w = {"papers": {"P1": {"claims": [["D1", "relies_on", "P1"]]}, "P2": {"claims": [["D1", "rejects", "P1"]]}}}
    assert join.clashes(w) == [{"subject": "D1", "object": "P1",
                                "relations": {"relies_on": ["P1"], "rejects": ["P2"]}}]
