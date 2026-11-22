"""Content plan: code decides, per paper, what the paper shows and where (23 Nov 2026).

Why: after the style cards the papers still looked alike. Every paper of a field got the same number columns (all of
the kind's numbers), the same two tables with the same titles, the same four plan lines, and its facts in the same
order with no place given, so the writer put them in the same sections every time. The style card only changed
adjectives. Now each paper gets its own plan, made here from the seed:

  attrs      per kind, which numbers this paper reports (a sample, in its own column order) and under which header
             wording (the pack's "labels"). The world keeps every number; a paper shows some.
  tables     how many tables and what each holds: one of several designs (setups and effects apart, one combined
             table plus pooled values, one effects table per object, two setup tables ...), each with its own title
             wording, orientation and CI layout, and the section it goes in.
  placement  for every planted fact: the section that states it first, the form (own sentence, comparison, clause
             ...) and whether the abstract names it. A paper leans early, late, spread or clustered, so the planted
             facts do not sit at the same depth in every paper.
  order      the facts in a shuffled order (claims and decoys mixed).
  outline    one line per section: what goes there. Replaces the genre's fixed plan lines.

The plan is saved beside the writer's notes (content_plan.json): the tagger and the leak test can compare where a
fact was planned with where it was found. This file has no field or genre names.
"""
from __future__ import annotations

import random
from typing import Any

from paperfactory import quant
from paperfactory.packs import Field, Genre

SETUP_TITLES = ["Setup of each {S} (one row per {S})", "Characteristics of the {SP}", "Design of the {SP} compared",
                "The {SP} at a glance", "How the {SP} were set up", "Key features of the {SP}"]
EFFECT_TITLES = ["{E} with 95% CI", "Reported effects ({E})", "Results by {S}: {E}", "What each {S} reported ({E})",
                 "Effect estimates: {E}"]
POOL_TITLES = ["Pooled values", "Pooled estimates and heterogeneity", "Summary across {SP}"]
COMBINED_TITLES = ["{SP_CAP}: setup and reported effect", "Overview of the {SP} and their results",
                   "Setup and results side by side"]
ATTR_TITLES = ["{L} in each {S}", "{L} by {S}", "{SP_CAP} compared: {L}"]
ORIENT = ["one row per {S}", "one row per {S}, sorted by year or size", "one column per {S} (the items are the rows)"]
CI_FORMS = ["the estimate and the two CI bounds in three columns",
            "one cell per estimate written as `estimate (lower to upper)`, with the word \"to\""]
CLAIM_FORMS = ["its own sentence", "a comparison with another {S} of the paper", "the first sentence of a paragraph",
               "a clause inside a longer sentence about that {S}", "the last sentence of a paragraph, as its point",
               "a sentence that first gives the setup, then the finding"]
DECOY_FORMS = ["a short sentence of context", "a clause in a sentence about a neighbouring {S}",
               "a contrast with one of the main findings", "its own sentence", "a sentence in a list-like run of "
               "{SP} (still prose)"]
LEANS = ("early", "late", "spread", "clustered")


def leaves(template: str) -> list[str]:
    """The sections of a template a fact or table can go in, in order: each level-2 heading, or its level-3
    children when it has some ("Results > Pooled estimates"). The abstract is left out."""
    out: list[str] = []
    parent, kids = None, 0
    for line in template.splitlines():
        if line.startswith("## "):
            if parent and not kids:
                out.append(parent)
            parent, kids = line[3:].strip(), 0
        elif line.startswith("### ") and parent:
            out.append(f"{parent} > {line[4:].strip()}")
            kids += 1
    if parent and not kids:
        out.append(parent)
    return [x for x in out if x.split(" > ")[0].lower() != "abstract"]


def _weights(lean: str, n: int, rng: random.Random) -> list[float]:
    if n <= 1:
        return [1.0] * n
    pos = [i / (n - 1) for i in range(n)]
    if lean == "early":
        return [1.6 - 1.3 * p for p in pos]
    if lean == "late":
        return [0.3 + 1.3 * p for p in pos]
    if lean == "clustered":
        c = rng.randrange(n)
        return [3.0 if i == c else 1.0 if abs(i - c) == 1 else 0.15 for i in range(n)]
    return [1.0] * n


def _labels(sp: dict[str, Any], attr: str) -> list[str]:
    return list(sp.get("labels") or [sp.get("label", attr.replace("_", " "))])


def make(w: dict[str, Any], f: Field, g: Genre, pid: str, template: str, seed: Any) -> dict[str, Any]:
    """The content plan of one paper. Same seed, same plan."""
    rng = random.Random(f"{seed}:plan:{f.name}:{pid}")
    p = w["papers"][pid]
    facts = [list(x) for x in p["claims"] + p["decoys"]]
    order = facts[:]
    rng.shuffle(order)
    ents = list(dict.fromkeys([x for s, _, o in order for x in (s, o)]))
    words = {"S": f.word("subject_kind"), "SP": f.word("subject_plural"),
             "SP_CAP": f.word("subject_plural").capitalize()}
    eff = f.raw.get("effect")
    keep_always = (eff or {}).get("se_from", {}).get("attr")

    # which numbers of each kind this paper reports, in which column order, under which wording
    attrs: dict[str, list[dict[str, str]]] = {}
    for kind in dict.fromkeys(w["entities"][e]["kind"] for e in ents):
        spec = f.kinds[kind].get("numbers", {})
        if not spec:
            continue
        names = list(spec)
        k = rng.randint(min(2, len(names)), min(len(names), 5))
        pick = rng.sample(names, k)
        if keep_always in names and keep_always not in pick:
            pick[rng.randrange(len(pick))] = keep_always
        attrs[kind] = [{"attr": a, "label": rng.choice(_labels(spec[a], a))} for a in pick]

    # sections
    secs = leaves(template)
    body = secs[1:-1] if len(secs) > 3 else secs          # tables never in the first or the last section
    lean = rng.choice(LEANS)
    wts = _weights(lean, len(secs), rng)
    placement = []
    claims = [list(x) for x in p["claims"]]
    for s, r, o in order:
        is_claim = [s, r, o] in claims
        form = rng.choice(CLAIM_FORMS if is_claim else DECOY_FORMS)
        placement.append({"fact": [s, r, o], "role": "claim" if is_claim else "decoy",
                          "section": rng.choices(secs, wts)[0] if secs else "",
                          "form": form.format(**words),
                          "abstract": bool(is_claim and rng.random() < 0.35)})
    if claims and not any(x["abstract"] for x in placement):      # an abstract names at least one finding
        rng.choice([x for x in placement if x["role"] == "claim"])["abstract"] = True

    return {"pid": pid, "lean": lean, "order": order, "attrs": attrs, "placement": placement, "sections": secs,
            "table_sections": body, "design_seed": rng.random(), "words": words,
            "orientation": rng.choice(ORIENT).format(**words), "ci_form": rng.choice(CI_FORMS)}


def _fmt(v: Any, decimals: int | None = None) -> str:
    if isinstance(v, float):
        return f"{v:.{decimals}f}" if decimals is not None else f"{v:g}"
    return str(v)


def number_rows(w: dict[str, Any], f: Field, g: Genre, pid: str, plan: dict[str, Any],
                row_prefix: str) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """(rows, tables) for a planned paper. rows: only the numbers this paper reports, each with a row id, a label in
    the plan's wording, the exact value, the unit and the placement (always a table: the number check needs it).
    tables: [{"title", "rows": [row ids], "section", "layout"}] in one of the designs; at least the genre's
    min_tables whenever the paper has enough numbers."""
    rng = random.Random(f"{plan['design_seed']}:{pid}")
    nums = w.get("numbers", {"entities": {}, "facts": {}})
    eff, words = f.raw.get("effect"), plan["words"]
    name = lambda e: w["entities"][e]["label"]  # noqa: E731
    rows: list[dict[str, str]] = []

    def add(label: str, value: Any, unit: str, decimals: int | None = None) -> str:
        rid = f"{row_prefix}-{len(rows) + 1:02d}"
        rows.append({"id": rid, "label": label, "value": _fmt(value, decimals), "unit": unit, "placement": "table"})
        return rid
    ents = list(dict.fromkeys([x for s, _, o in plan["order"] for x in (s, o)]))
    by_attr: dict[str, list[str]] = {}          # attr -> row ids
    by_ent: dict[str, list[str]] = {}           # entity -> its setup row ids
    label_of: dict[str, str] = {}
    for e in ents:
        kind = w["entities"][e]["kind"]
        spec = f.kinds[kind].get("numbers", {})
        for col in plan["attrs"].get(kind, []):
            v = nums["entities"].get(e, {}).get(col["attr"])
            if v is None:
                continue
            rid = add(f"{name(e)}: {col['label']}", v, spec[col["attr"]].get("unit", ""))
            by_attr.setdefault(col["attr"], []).append(rid)
            by_ent.setdefault(e, []).append(rid)
            label_of[col["attr"]] = col["label"]
    by_obj: dict[str, list[str]] = {}           # object -> effect row ids
    by_subj: dict[str, list[str]] = {}
    xs_obj: dict[str, list[dict[str, Any]]] = {}
    pooled: list[str] = []
    if eff:
        d, unit = eff.get("decimals", 2), eff["unit"]
        for s, r, o in plan["order"]:
            x = nums["facts"].get(quant.fact_id(s, r, o))
            if x is None:
                continue
            what = f"{name(s)} / {name(o)}"
            ids = []
            if "baseline" in x:
                ids.append(add(f"{what}: baseline {eff['short']}", x["baseline"], unit, d))
                ids.append(add(f"{what}: {eff['short']} with the method", x["with"], unit, d))
            ids.append(add(f"{what}: {eff['name']}", x["estimate"], unit, d))
            ids.append(add(f"{what}: 95% CI lower bound", x["ci_low"], unit, d))
            ids.append(add(f"{what}: 95% CI upper bound", x["ci_high"], unit, d))
            by_obj.setdefault(o, []).extend(ids)
            by_subj.setdefault(s, []).extend(ids)
            xs_obj.setdefault(o, []).append(x)
        for o, xs in xs_obj.items():
            if len(xs) < 2:
                continue
            pool = quant.pooled(xs, eff)
            lab = f"{eff.get('pool_label', 'pooled estimate')} for {name(o)} ({pool['studies']} {words['SP']})"
            pooled += [add(f"{lab}: estimate", pool["estimate"], unit, d),
                       add(f"{lab}: 95% CI lower bound", pool["ci_low"], unit, d),
                       add(f"{lab}: 95% CI upper bound", pool["ci_high"], unit, d),
                       add(f"{lab}: heterogeneity I²", pool["i2"], "%", 1)]
    setups = [r for ids in by_attr.values() for r in ids]
    effects = [r for ids in by_obj.values() for r in ids]
    tw = words | {"E": (eff or {}).get("name", "results")}
    t = lambda pool_, **kw: rng.choice(pool_).format(**(tw | kw))  # noqa: E731
    attr_names = list(by_attr)

    def split_attrs() -> list[list[str]]:
        cut = rng.randint(1, len(attr_names) - 1)
        return [attr_names[:cut], attr_names[cut:]]

    def attr_table(names: list[str]) -> dict[str, Any]:
        title = (t(ATTR_TITLES, L=label_of[names[0]].capitalize()) if len(names) == 1 else t(SETUP_TITLES))
        return {"title": title, "rows": [r for a in names for r in by_attr[a]], "layout": plan["orientation"]}

    designs: dict[str, list[dict[str, Any]]] = {}
    eff_layout = f"one row per {words['S']} and object; {plan['ci_form']}"
    if setups and effects:
        designs["split"] = [{"title": t(SETUP_TITLES), "rows": setups, "layout": plan["orientation"]},
                            {"title": t(EFFECT_TITLES), "rows": effects + pooled, "layout": eff_layout}]
        if pooled:
            designs["combined"] = [
                {"title": t(COMBINED_TITLES), "rows": setups + effects,
                 "layout": f"one row per {words['S']}: its setup columns, then its effect; {plan['ci_form']}"},
                {"title": t(POOL_TITLES), "rows": pooled, "layout": "one row per pooled object"}]
            designs["three"] = [{"title": t(SETUP_TITLES), "rows": setups, "layout": plan["orientation"]},
                                {"title": t(EFFECT_TITLES), "rows": effects, "layout": eff_layout},
                                {"title": t(POOL_TITLES), "rows": pooled, "layout": "one row per pooled object"}]
        if 2 <= len(by_obj) <= 3 and all(len(v) >= 2 for v in xs_obj.values()):   # no one-row tables
            per = [{"title": t(EFFECT_TITLES) + f": {name(o)}", "rows": ids, "layout": eff_layout}
                   for o, ids in by_obj.items()]
            if pooled:
                per.append({"title": t(POOL_TITLES), "rows": pooled, "layout": "one row per pooled object"})
            designs["by_object"] = [{"title": t(SETUP_TITLES), "rows": setups, "layout": plan["orientation"]}] + per
        if len(attr_names) >= 2:
            a, b = split_attrs()
            designs["two_setups"] = [attr_table(a), attr_table(b),
                                     {"title": t(EFFECT_TITLES), "rows": effects + pooled, "layout": eff_layout}]
    elif setups:                                   # no effect numbers (e.g. law): tables from the entity numbers
        if len(attr_names) >= 2:
            designs["by_attr"] = [attr_table([a]) for a in attr_names]
            a, b = split_attrs()
            designs["two_groups"] = [attr_table(a), attr_table(b)]
            if len(attr_names) >= 3:
                c = attr_names[:]
                rng.shuffle(c)
                designs["one_and_rest"] = [attr_table(c[:1]), attr_table(c[1:])]
        else:
            designs["single"] = [attr_table(attr_names)]
    need = g.raw.get("min_tables", 0)
    ok = {k: v for k, v in designs.items() if len(v) >= need} or designs
    if not ok:
        return rows, []
    design = rng.choice(sorted(ok))
    tables = ok[design]
    # sections: in order, never before the first table's section
    body = plan["table_sections"] or plan["sections"] or [""]
    idx = sorted(rng.randrange(len(body)) for _ in tables)
    for tb, i in zip(tables, idx):
        tb["section"] = body[i]
        tb["design"] = design
    return rows, tables


def outline(w: dict[str, Any], f: Field, plan: dict[str, Any], tables: list[dict[str, Any]], issue: str,
            thesis: str) -> list[str]:
    """One line per section: the tables and the findings that go there. No relation words (the writer would copy
    them): a finding is named by its two entities."""
    name = lambda e: w["entities"][e]["label"]  # noqa: E731
    secs = plan["sections"]
    lines = []
    for i, sec in enumerate(secs):
        parts = []
        if i == 0:
            parts.append(f"the question ({issue})")
        parts += [f"Table {j + 1}" for j, tb in enumerate(tables) if tb.get("section") == sec]
        here = [x for x in plan["placement"] if x["section"] == sec]
        parts += [f"the {f.word('fact')} of {name(x['fact'][0])} on {name(x['fact'][2])}, as {x['form']}" for x in here]
        if i == len(secs) - 1:
            parts.append(f"the conclusion ({thesis})")
        if parts:
            lines.append(f"**{sec}**: " + "; ".join(parts) + ".")
    named = [x for x in plan["placement"] if x["abstract"]]
    if named:
        lines.insert(0, "**Abstract**: names " + "; ".join(
            f"the {f.word('fact')} of {name(x['fact'][0])} on {name(x['fact'][2])}" for x in named) + ".")
    return lines
