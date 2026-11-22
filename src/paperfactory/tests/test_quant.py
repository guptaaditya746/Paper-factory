"""Numbers: the CI width follows the study's size (se_from), for every pack that has effects."""
from __future__ import annotations

import copy
import math
import random

import pytest

from paperfactory import packs, quant

FIELDS = [n for n in ("medicine", "llm", "materials") if packs.field(n).raw.get("effect", {}).get("se_from")]


def _world(f: packs.Field, n: int = 60) -> dict:
    """n studies, each with one fact about one of 4 objects; directions in turn."""
    kind = f.subject_kinds[0]
    rels = [r for r, d in f.relations.items() if d.get("effect")]
    okind = f.relations[rels[0]]["object_kinds"][0]
    ents = {f"O{j}": {"kind": okind, "label": f"O{j}", "attrs": {}} for j in range(4)}
    ents |= {f"S{i}": {"kind": kind, "label": f"S{i}", "attrs": {}} for i in range(n)}
    facts = [[f"S{i}", rels[i % len(rels)], f"O{i % 4}"] for i in range(n)]
    return {"entities": ents, "papers": {"P1": {"claims": facts, "decoys": []}}}


@pytest.mark.parametrize("name", FIELDS)
def test_world_numbers_pass_their_own_check(name: str) -> None:
    f = packs.field(name)
    for seed in range(5):
        w = _world(f)
        w["numbers"] = quant.make(w, f, seed)
        assert quant.errors(w, f) == []
        assert set(w["numbers"]["basis"]) == {e for e in w["entities"] if e.startswith("S")}


@pytest.mark.parametrize("name", FIELDS)
def test_ci_width_follows_the_size(name: str) -> None:
    f = packs.field(name)
    eff, w = f.raw["effect"], _world(f, 200)
    nums = quant.make(w, f, 3)
    tf = math.log if eff["scale"] == "ratio" else (lambda x: x)
    for (s, r, o) in w["papers"]["P1"]["claims"]:
        x, b = nums["facts"][quant.fact_id(s, r, o)], nums["basis"][s]
        got = (tf(x["ci_high"]) - tf(x["ci_low"])) / (2 * quant.Z)
        assert got == pytest.approx(b["k"] / math.sqrt(b["n"] * b["rate"]), rel=0.1, abs=0.02)
    # same k and rate, four times the size: half the width
    b = {"attr": "n", "n": 400, "k": 2.0, "rate": 0.3}
    assert quant.se_of(b | {"n": 1600}) == pytest.approx(quant.se_of(b) / 2)


def test_a_study_too_small_for_its_claim_gets_larger() -> None:
    f = packs.field("medicine")
    eff = f.raw["effect"]
    n_min = quant.min_size(eff, {"better"}, 2.0, 0.12)
    assert n_min > 400                               # a hazard-ratio benefit needs hundreds of patients
    assert quant.min_size(eff, {"none"}, 2.0, 0.12) == 0
    w = _world(f, 90)
    nums = quant.make(w, f, 1)
    for s, r, _ in w["papers"]["P1"]["claims"]:
        if f.relations[r]["effect"] != "none":
            b = nums["basis"][s]
            assert nums["entities"][s]["patients"] >= quant.min_size(eff, {f.relations[r]["effect"]}, b["k"], b["rate"])
    with pytest.raises(ValueError, match="too small"):
        quant.effect_numbers(eff, "better", random.Random(0), se=2.0 / math.sqrt(30))


def test_check_catches_a_size_changed_after_the_cis_were_made() -> None:
    f = packs.field("medicine")
    w = _world(f, 12)
    w["numbers"] = quant.make(w, f, 2)
    bad = copy.deepcopy(w)
    bad["numbers"]["entities"]["S0"]["patients"] += 500
    assert any("S0: patients" in e for e in quant.errors(bad, f))
    bad = copy.deepcopy(w)
    k = quant.fact_id(*w["papers"]["P1"]["claims"][0])
    x = bad["numbers"]["facts"][k]
    x["ci_low"], x["ci_high"] = round(x["estimate"] * 0.99, 2), round(x["estimate"] * 1.01, 2)
    assert any("CI width" in e or "does not agree" in e for e in quant.errors(bad, f))


def test_without_se_from_the_old_fixed_width_still_works() -> None:
    f = packs.field("medicine")
    g = packs.Field({k: v for k, v in copy.deepcopy(f.raw).items()})
    del g.raw["effect"]["se_from"]
    w = _world(g, 12)
    w["numbers"] = quant.make(w, g, 2)
    assert "basis" not in w["numbers"] and quant.errors(w, g) == []
