"""Numbers of the world, made and checked by code (22 Nov 2026). The model never invents a number.

Why: the 40-paper pilot papers had no numbers at all, and the writer filled the gaps with invented details that
contradicted each other across papers (outside review). Now every number lives in the world, once, so every paper
that mentions it uses the same value.

  entity numbers  per entity, from the pack's "kinds".<kind>."numbers" specs, e.g. a trial's patients and follow-up
                  ({"int": [lo, hi]}, {"float": [lo, hi], "decimals": d}, {"choice": [...]},
                  {"int_from": <range key in packs/topics/ranges.json>, "lo": "p25", "hi": "p90"}; "round": n)
  fact numbers    per planted fact whose relation has "effect": better | none | worse, from the pack's "effect" spec:
                  an estimate with a 95% CI that agrees with the relation (a benefit's CI excludes the null; no
                  effect's CI contains it). scale "ratio" (e.g. a hazard ratio; null 1) or "difference" (e.g.
                  accuracy points; null 0; with a baseline "base" when given).
                  With "se_from" in the effect spec the CI width follows the study's size: se = k / sqrt(n * rate),
                  n = the subject's size attribute ({"attr": "patients", "k": 2.0, "rate": [0.12, 0.45]}; k and
                  rate are a number or a [lo, hi] range, drawn once per study). A study too small to show the
                  effects its facts state gets a larger size. Without "se_from": the old fixed "ci_rel"/"ci_half".
  pooled          per paper and object: inverse-variance pooled estimate over the paper's studies of that object,
                  with Cochran's Q and I^2 (computed here, never by the writer).

w["numbers"] = {"entities": {eid: {attr: value}}, "facts": {fact_id: {...}},
                "basis": {subject id: {"attr", "n", "k", "rate"}} (only with "se_from")}.
"""
from __future__ import annotations

import math
import random
from typing import Any

from paperfactory import topics
from paperfactory.packs import Field

Z = 1.959964


def fact_id(s: str, r: str, o: str) -> str:
    return f"{s}-{r}-{o}"


def _rnd(x: float, d: int) -> float:
    return round(x + 0.0, d)


def _value(spec: dict[str, Any], rng: random.Random, ranges: dict[str, Any]) -> Any:
    if "choice" in spec:
        return rng.choice(spec["choice"])
    if "int_from" in spec:
        r = ranges.get(spec["int_from"])
        if not r:
            raise ValueError(f"number range {spec['int_from']!r} missing: run `paperfactory fetch-topics`")
        lo, hi = r[spec.get("lo", "p25")], r[spec.get("hi", "p90")]
        v = int(math.exp(rng.uniform(math.log(max(lo, 1)), math.log(max(hi, lo + 1)))))
    elif "int" in spec:
        v = rng.randint(*spec["int"])
    elif "float" in spec:
        return _rnd(rng.uniform(*spec["float"]), spec.get("decimals", 1))
    else:
        raise ValueError(f"unknown number spec {spec}")
    n = spec.get("round", 1)
    return max(n, int(round(v / n)) * n) if n > 1 else v


def _tf(eff: dict[str, Any]):
    """(to the scale the CI is symmetric on, back, the null value on that scale)."""
    if eff["scale"] == "ratio":
        return math.log, math.exp, 0.0
    return (lambda x: x), (lambda x: x), 0.0


def _range(eff: dict[str, Any], direction: str, se: float) -> tuple[float, float] | None:
    """The part of the pack's range for `direction` whose 95% CI (half-width Z * se on the CI scale) agrees with the
    direction: excludes the null for better / worse, contains it for none. None when no value does."""
    tf, back, null = _tf(eff)
    lo, hi = sorted(tf(x) for x in eff[direction])
    half = Z * se
    if direction == "none":
        lo, hi = max(lo, null - half * 0.95), min(hi, null + half * 0.95)
    elif (direction == "better") == (eff.get("better_side", "higher") == "lower"):     # the effect lies below null
        hi = min(hi, null - half * 1.03)
    else:
        lo = max(lo, null + half * 1.03)
    return (back(lo), back(hi)) if lo < hi else None


def effect_numbers(eff: dict[str, Any], direction: str, rng: random.Random, se: float | None = None) -> dict[str, Any]:
    """One estimate with a 95% CI that agrees with `direction` (better | none | worse). se: the standard error on
    the CI scale (log for ratios), from the study's size; None = a width from the pack's fixed range."""
    d = eff.get("decimals", 2)
    lo, hi = eff[direction]
    if se is not None:
        r = _range(eff, direction, se)
        if r is None:
            raise ValueError(f"se {se:.4f} is too large for a {direction!r} effect in {eff[direction]} "
                             f"({eff.get('name')}): the study is too small")
        lo, hi = r
    for _ in range(200):
        est = rng.uniform(lo, hi)
        if eff["scale"] == "ratio":
            s_ = se if se is not None else math.log(1 + rng.uniform(*eff["ci_rel"]))
            cl, ch = _rnd(math.exp(math.log(est) - Z * s_), d), _rnd(math.exp(math.log(est) + Z * s_), d)
            null = 1.0
        else:
            half = Z * se if se is not None else rng.uniform(*eff["ci_half"])
            cl, ch = _rnd(est - half, d), _rnd(est + half, d)
            null = 0.0
        est = _rnd(est, d)
        if _agrees(eff, direction, est, cl, ch, null):
            out = {"estimate": est, "ci_low": cl, "ci_high": ch}
            if eff["scale"] == "difference" and "base" in eff:
                base = _rnd(rng.uniform(*eff["base"]), d)
                out |= {"baseline": base, "with": _rnd(base + est, d)}
            return out
    raise ValueError(f"no estimate agrees with direction {direction} for effect {eff.get('name')}")


def _draw(x: Any, rng: random.Random) -> float:
    return float(rng.uniform(*x)) if isinstance(x, (list, tuple)) else float(x)


def se_of(b: dict[str, Any]) -> float:
    """Standard error on the CI scale from a study's basis: k / sqrt(n * rate)."""
    return b["k"] / math.sqrt(max(b["n"] * b["rate"], 1e-9))


def min_size(eff: dict[str, Any], directions: set[str], k: float, rate: float) -> int:
    """The smallest size at which every stated direction still has room in the pack's range (the CI of the
    strongest allowed effect excludes the null with a fifth to spare). 'none' needs no minimum."""
    tf, _, null = _tf(eff)
    need = 0.0
    for direction in directions - {"none"}:
        far = max(abs(tf(x) - null) for x in eff[direction])
        need = max(need, (Z * 1.03 * k / (0.8 * far)) ** 2 / rate)
    return math.ceil(need)


def _at_least(spec: dict[str, Any], n_min: int, rng: random.Random, ranges: dict[str, Any]) -> int:
    """A size of at least n_min that the spec allows (or the smallest step above its range when none does)."""
    if "choice" in spec:
        ok = sorted(c for c in spec["choice"] if c >= n_min)
        return rng.choice(ok) if ok else max(spec["choice"])
    if "int_from" in spec:
        hi = ranges[spec["int_from"]][spec.get("hi", "p90")]
    else:
        hi = spec["int"][1]
    v = int(math.exp(rng.uniform(math.log(n_min), math.log(max(hi, n_min * 1.5)))))
    n = spec.get("round", 1)
    return max(n_min, int(math.ceil(v / n)) * n)


def _agrees(eff: dict[str, Any], direction: str, est: float, cl: float, ch: float, null: float) -> bool:
    better_low = eff.get("better_side", "higher") == "lower"
    if not cl < est < ch:
        return False
    if direction == "none":
        return cl < null < ch
    good = ch < null if better_low else cl > null
    bad = cl > null if better_low else ch < null
    return good if direction == "better" else bad


def make(w: dict[str, Any], f: Field, seed: Any) -> dict[str, Any]:
    rng, ranges = random.Random(f"{seed}:numbers:{f.name}"), topics.ranges()
    ents: dict[str, dict[str, Any]] = {}
    for eid in sorted(w["entities"]):
        spec = f.kinds[w["entities"][eid]["kind"]].get("numbers", {})
        if spec:
            ents[eid] = {a: _value(sp, rng, ranges) for a, sp in spec.items()}
    eff = f.raw.get("effect")
    todo: dict[str, tuple[str, str]] = {}          # fact id -> (subject, direction), in a fixed order
    by_subject: dict[str, set[str]] = {}
    for pid in sorted(w["papers"]):
        p = w["papers"][pid]
        for s, r, o in p["claims"] + p["decoys"]:
            direction = f.relations[r].get("effect")
            if eff and direction and fact_id(s, r, o) not in todo:
                todo[fact_id(s, r, o)] = (s, direction)
                by_subject.setdefault(s, set()).add(direction)
    basis: dict[str, dict[str, Any]] = {}
    sf = eff.get("se_from") if eff else None
    if sf:
        for s in sorted(by_subject):
            spec = f.kinds[w["entities"][s]["kind"]].get("numbers", {}).get(sf["attr"])
            if spec is None:
                raise ValueError(f"effect.se_from.attr {sf['attr']!r} is not a number of kind "
                                 f"{w['entities'][s]['kind']!r}")
            r2 = random.Random(f"{seed}:basis:{f.name}:{s}")         # per study: adding a study moves no other
            k, rate = _draw(sf.get("k", 1.0), r2), _draw(sf.get("rate", 1.0), r2)
            n_min = min_size(eff, by_subject[s], k, rate)
            if ents[s][sf["attr"]] < n_min:
                ents[s][sf["attr"]] = _at_least(spec, n_min, r2, ranges)
            basis[s] = {"attr": sf["attr"], "n": ents[s][sf["attr"]], "k": round(k, 4), "rate": round(rate, 4)}
    facts: dict[str, dict[str, Any]] = {}
    for k_, (s, direction) in todo.items():
        facts[k_] = effect_numbers(eff, direction, rng, se_of(basis[s]) if s in basis else None)
    out = {"entities": ents, "facts": facts}
    if basis:
        out["basis"] = basis
    return out


def errors(w: dict[str, Any], f: Field) -> list[str]:
    """Every fact with an effect relation has numbers that agree with it; every entity has its pack numbers."""
    nums, eff, err = w.get("numbers"), f.raw.get("effect"), []
    if nums is None:
        return ["world has no numbers"] if eff or any(k.get("numbers") for k in f.kinds.values()) else []
    for eid, e in w["entities"].items():
        want = set(f.kinds[e["kind"]].get("numbers", {}))
        if want - set(nums["entities"].get(eid, {})):
            err.append(f"{eid}: numbers missing {sorted(want - set(nums['entities'].get(eid, {})))}")
    for p in w["papers"].values():
        for s, r, o in p["claims"] + p["decoys"]:
            direction = f.relations[r].get("effect")
            if not (eff and direction):
                continue
            x = nums["facts"].get(fact_id(s, r, o))
            null = 1.0 if eff["scale"] == "ratio" else 0.0
            if x is None:
                err.append(f"{fact_id(s, r, o)}: no numbers")
            elif not _agrees(eff, direction, x["estimate"], x["ci_low"], x["ci_high"], null):
                err.append(f"{fact_id(s, r, o)}: estimate {x['estimate']} ({x['ci_low']}, {x['ci_high']}) does not "
                           f"agree with {r}")
            elif (b := nums.get("basis", {}).get(s)) is not None:
                # the CI width must follow the study's size (a small study cannot report a tight interval)
                tf = _tf(eff)[0]
                if nums["entities"].get(s, {}).get(b["attr"]) != b["n"]:
                    err.append(f"{s}: {b['attr']} is {nums['entities'].get(s, {}).get(b['attr'])}, its CIs were made "
                               f"for {b['n']}")
                got, want = (tf(x["ci_high"]) - tf(x["ci_low"])) / (2 * Z), se_of(b)
                slack = 0.1 * want + 10 ** -eff.get("decimals", 2) / (x["ci_low"] if eff["scale"] == "ratio" else 1)
                if abs(got - want) > slack:
                    err.append(f"{fact_id(s, r, o)}: CI width gives se {got:.4f}, the study's size gives {want:.4f}")
    return err


def pooled(rows: list[dict[str, Any]], eff: dict[str, Any]) -> dict[str, Any]:
    """Inverse-variance fixed-effect pool of estimates with 95% CIs (log scale for ratios), Cochran's Q and I^2 (%).
    rows: [{"estimate", "ci_low", "ci_high"}], at least 2."""
    d = eff.get("decimals", 2)
    ratio = eff["scale"] == "ratio"
    tf = math.log if ratio else (lambda x: x)
    ys, ws = [], []
    for x in rows:
        se = (tf(x["ci_high"]) - tf(x["ci_low"])) / (2 * Z)
        ys.append(tf(x["estimate"]))
        ws.append(1 / max(se, 1e-9) ** 2)
    m = sum(w_ * y for w_, y in zip(ws, ys)) / sum(ws)
    se = math.sqrt(1 / sum(ws))
    q = sum(w_ * (y - m) ** 2 for w_, y in zip(ws, ys))
    i2 = max(0.0, (q - (len(ys) - 1)) / q) * 100 if q > 0 else 0.0
    back = math.exp if ratio else (lambda x: x)
    return {"estimate": _rnd(back(m), d), "ci_low": _rnd(back(m - Z * se), d), "ci_high": _rnd(back(m + Z * se), d),
            "studies": len(ys), "q": _rnd(q, 2), "i2": _rnd(i2, 1)}
