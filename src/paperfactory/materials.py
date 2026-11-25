"""World -> PaperOrchestra inputs per paper (idea_sparse.md + experimental_log.md). Code only.

Facts go in as DATA (subject | relation words | source), never as finished sentences: in the law pilot a fixed
sentence per relation was copied by the writer 10-13 times per paper and made planted sentences easy to spot.
Every field word comes from the field pack ("words"; Field.word), so the shared code holds no field's wording.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from paperfactory import quant
from paperfactory.packs import Field, Genre, fill, word_values


def _name(w: dict[str, Any], eid: str) -> str:
    return w["entities"][eid]["label"]


def _attrs(w: dict[str, Any], f: Field, eid: str) -> str:
    e = w["entities"][eid]
    parts = []
    for a, typ in f.kinds[e["kind"]].get("attrs", {}).items():
        if a not in e["attrs"]:
            continue
        v = e["attrs"][a]
        parts.append(f"{a}: {_name(w, v) if typ.startswith('ref:') else v}")
    return "; ".join(parts)


def materials(w: dict[str, Any], f: Field, g: Genre, pid: str) -> tuple[str, str]:
    p = w["papers"][pid]
    fact, facts_w, action = f.word("fact"), f.word("facts"), f.word("action")
    facts = p["claims"] + p["decoys"]
    ents = sorted({x for s, _, o in facts for x in (s, o)})
    idea = f"""# {p['title_hint']}

## Problem Statement
{p['issue']}

## Core Thesis
{p['thesis']}

## Approach
An analysis of the {f.word('subject_plural')} listed in the log, read against the sources they deal with.

## Expected Contribution
{textwrap.fill(f"A clear account of how the {f.word('subject_plural')} treated these sources, and an argued view on "
                f"which {f.word('line_of_authority')} should be followed.", 100)}
"""
    lines = ["# Argument Log", "", "## 1. Setting",
             f"* **Genre:** argument paper ({f.word('paper_kind')}).",
             f"* **Setting:** {f.raw['setting']}",
             f"* **Issue:** {p['issue']}", f"* **Thesis to defend:** {p['thesis']}", "", "## 2. Sources"]
    for kind in f.kinds:
        ids = [e for e in ents if w["entities"][e]["kind"] == kind]
        if ids:
            lines += [f"### {kind.replace('_', ' ').capitalize()}", "| Name | Short names | Details |", "|---|---|---|"]
            lines += [f"| {_name(w, e)} | {', '.join(w['entities'][e].get('alt', []))} | {_attrs(w, f, e)} |"
                      for e in ids]
            lines.append("")
    lines += [f"## 3. {facts_w.capitalize()} (data; state each in your own words)",
              f"{action.capitalize()} meanings: " + "; ".join(f"{d['words']} = {d['meaning']}" for d in f.relations.values()) + ".",
              "", f"| {f.word('subject_kind').capitalize()} | {action.capitalize()} | Source |", "|---|---|---|"]
    lines += [f"| {_name(w, s)} | {f.rel_words(r)} | {_name(w, o)} |" for s, r, o in facts]
    if p.get("conflict"):
        (s1, r1, o1), (s2, r2, _) = p["conflict"]
        lines += ["", "## 4. Conflict", f"* {_name(w, s1)} ({f.rel_words(r1)}) and {_name(w, s2)} ({f.rel_words(r2)}) "
                  f"disagree about {_name(w, o1)}."]
    plan = [fill(x, word_values(f)) for x in g.raw.get("argument_plan", [])]
    lines += ["", "## 5. Argument Plan"] + [f"* {x}" for x in plan] + [
        "", "## 6. Rules",
        f"* Do not invent {f.word('source_kinds')}, {facts_w}, dates or quotations.",
        f"* Keep the meaning of every {action} exact, but phrase it in your own words and vary the wording; never "
        f"reuse one phrase for several {facts_w} and never copy this log's wording.",
        f"* State each {fact} at least once and at most {g.raw.get('max_mentions_per_fact', 3)} times."]
    return idea, "\n".join(lines) + "\n"


# ------------------------------------------------------------------ v2 (22 Nov 2026): numbers, tables, style
def _row_prefix(f: Field, pid: str) -> str:
    """PaperOrchestra's row id form R-<letters>-<paper>-<row> (utils/fact_check.py ROW_ID)."""
    letters = "".join(ch for ch in f.name.lower() if ch.isalpha())[:3]
    num = "".join(ch for ch in pid if ch.isdigit()) or "0"
    return f"R-{letters}-{int(num):02d}"


def _fmt(v: Any, decimals: int | None = None) -> str:
    if isinstance(v, float):
        return f"{v:.{decimals}f}" if decimals is not None else f"{v:g}"
    return str(v)


def number_rows(w: dict[str, Any], f: Field, pid: str) -> tuple[list[dict[str, str]], list[list[str]]]:
    """(rows, tables). rows: every number of the paper, each with a row id, a label, the value as the paper must
    print it, its unit and its placement. tables: [title, row ids] for each table to build (setups first, then
    effects with the pooled values). Pooled values are computed here (quant.pooled), never by the writer.
    Placement: every number in a table (the text may repeat any of them; PaperOrchestra checks the tables)."""
    p, nums = w["papers"][pid], w.get("numbers", {"entities": {}, "facts": {}})
    eff = f.raw.get("effect")
    pre, rows, n = _row_prefix(f, pid), [], 0
    setups, effects, by_attr = [], [], {}

    def add(label: str, value: Any, unit: str, place: str, bucket: list[str], decimals: int | None = None,
            about: dict[str, Any] | None = None) -> None:
        nonlocal n
        n += 1
        rid = f"{pre}-{n:02d}"
        # about (world ids): what the number is, for the benchmark's fact rows (workbench.number_rows); the
        # writer's brief prints only id, label, value, unit and placement
        rows.append({"id": rid, "label": label, "value": _fmt(value, decimals), "unit": unit, "placement": place,
                     "about": about or {}})
        bucket.append(rid)
    facts = p["claims"] + p["decoys"]
    ents = list(dict.fromkeys([x for s_, _, o in facts for x in (s_, o)]))
    for e in ents:
        spec = f.kinds[w["entities"][e]["kind"]].get("numbers", {})
        for attr, sp in spec.items():
            v = nums["entities"].get(e, {}).get(attr)
            if v is None:
                continue
            unit = sp.get("unit", "")
            # v2 pilot (22 Nov 2026): a column split between text and table made the writer drop it (MED-02,
            # MED-03: follow-up weeks of the other trials missing); every number now goes in a table
            place = "table"
            add(f"{_name(w, e)}: {sp.get('label', attr.replace('_', ' '))}", v, unit, place,
                setups if place == "table" else [], about={"entity": e, "attr": attr})
            if place == "table":
                by_attr.setdefault(attr, []).append(rows[-1]["id"])
    by_obj: dict[str, list[dict[str, Any]]] = {}
    if eff:
        d = eff.get("decimals", 2)
        unit = eff["unit"]
        for s_, r, o in facts:
            x = nums["facts"].get(f"{s_}-{r}-{o}")
            if x is None:
                continue
            what = f"{_name(w, s_)} / {_name(w, o)}"
            ab = lambda stat: {"fact": [s_, r, o], "stat": stat}  # noqa: E731
            if "baseline" in x:
                add(f"{what}: baseline {eff['short']}", x["baseline"], unit, "table", effects, d, ab("baseline"))
                add(f"{what}: {eff['short']} with the method", x["with"], unit, "table", effects, d, ab("with"))
            dunit = eff.get("delta_unit", unit)
            place = "table"     # MAT-04: the estimate column was dropped when the claims' estimates were text-only
            add(f"{what}: {eff['name']}", x["estimate"], dunit if place == "text" else unit, place,
                effects if place == "table" else [], d, ab("estimate"))
            add(f"{what}: 95% CI lower bound", x["ci_low"], unit, "table", effects, d, ab("ci_low"))
            add(f"{what}: 95% CI upper bound", x["ci_high"], unit, "table", effects, d, ab("ci_high"))
            by_obj.setdefault(o, []).append((s_, x))
        for o, sx in by_obj.items():
            if len(sx) < 2:
                continue
            pool = quant.pooled([x for _, x in sx], eff)
            studies = [s_ for s_, _ in sx]
            pa = lambda stat: {"pool": o, "stat": stat, "studies": studies}  # noqa: B023, E731
            lab = f"{eff.get('pool_label', 'pooled estimate')} for {_name(w, o)} ({pool['studies']} {f.word('subject_plural')})"
            add(f"{lab}: estimate", pool["estimate"], unit, "table", effects, d, pa("estimate"))
            add(f"{lab}: 95% CI lower bound", pool["ci_low"], unit, "table", effects, d, pa("ci_low"))
            add(f"{lab}: 95% CI upper bound", pool["ci_high"], unit, "table", effects, d, pa("ci_high"))
            add(f"{lab}: heterogeneity I²", pool["i2"], "%", "table", effects, 1, pa("i2"))
    tables = []
    if setups:
        tables.append([f"Setup of each {f.word('subject_kind')} (one row per {f.word('subject_kind')})", setups])
    if effects:
        tables.append([f"Effects ({eff['name']}, 95% CI) and pooled values" if eff else "Results", effects])
    if not effects and len(by_attr) >= 2:      # no effect numbers (e.g. law): one table per kind of number
        lab = {a: sp.get("label", a) for k in f.kinds.values() for a, sp in k.get("numbers", {}).items()}
        tables = [[f"{lab.get(a, a).capitalize()} in each {f.word('subject_kind')}", ids] for a, ids in by_attr.items()]
    return rows, tables


def materials_v2(w: dict[str, Any], f: Field, g: Genre, pid: str, card: dict[str, Any]) -> tuple[str, str]:
    """Notes for a genre with numbers (numeric_facts): the v1 notes plus the source summaries, every number as a
    row (PaperOrchestra checks each one), the tables to build, and the paper type from the style card."""
    p = w["papers"][pid]
    fact, facts_w, action = f.word("fact"), f.word("facts"), f.word("action")
    facts = p["claims"] + p["decoys"]
    ents = sorted({x for s, _, o in facts for x in (s, o)})
    rows, tables = number_rows(w, f, pid)
    idea = f"""# {p['title_hint']}

## Problem Statement
{p['issue']}

## Core Thesis
{p['thesis']}

## Approach
{card['paper_kind'][:1].upper() + card['paper_kind'][1:]} of the {f.word('subject_plural')} listed in the log.

## Expected Contribution
{textwrap.fill(f"An answer to the question: {p['issue']}", 100)}
"""
    lines = ["# Experimental Log", "", "## 1. Setting",
             f"* **Paper type:** {card['paper_kind']}.",
             f"* **Setting:** {f.raw['setting']}",
             f"* **Question:** {p['issue']}", f"* **Main conclusion to support:** {p['thesis']}", "", "## 2. Sources"]
    for kind in f.kinds:
        ids = [e for e in ents if w["entities"][e]["kind"] == kind]
        if ids:
            lines += [f"### {kind.replace('_', ' ').capitalize()}", "| Name | Short names | Details |", "|---|---|---|"]
            lines += [f"| {_name(w, e)} | {', '.join(w['entities'][e].get('alt', []))} | {_attrs(w, f, e)} |"
                      for e in ids]
            lines.append("")
    lines += [f"## 3. {facts_w.capitalize()} (data; state each in your own words)",
              f"{action.capitalize()} meanings: " + "; ".join(f"{d['words']} = {d['meaning']}"
                                                            for d in f.relations.values()) + ".",
              "", f"| {f.word('subject_kind').capitalize()} | {action.capitalize()} | Source |", "|---|---|---|"]
    lines += [f"| {_name(w, s)} | {f.rel_words(r)} | {_name(w, o)} |" for s, r, o in facts]
    lines += ["", "## 4. Numbers (data; every value exactly as written, in its placement)",
              "| Row id | Item | Value | Unit | Placement |", "|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['label']} | {r['value']} | {r['unit']} | "
              f"{'state in a sentence' if r['placement'] == 'text' else 'table'} |" for r in rows]
    lines += ["", "## 5. Tables to build"]
    # row ids appear only in the Numbers table (PaperOrchestra reads every line holding one as a fact): here by
    # their last part
    lines += [f"* Table {i + 1}: {title}. Numbers rows ending in: {', '.join(x.rsplit('-', 1)[1] for x in ids)}."
              for i, (title, ids) in enumerate(tables)]
    lines += ["* Values marked 'state in a sentence' go in the text, not in a table.",
              "* Row ids are for you only: never print a row id in the paper."]
    k = 6
    if p.get("conflict"):
        (s1, r1, o1), (s2, r2, _) = p["conflict"]
        lines += ["", f"## {k}. Conflict", f"* {_name(w, s1)} ({f.rel_words(r1)}) and {_name(w, s2)} "
                  f"({f.rel_words(r2)}) disagree about {_name(w, o1)}."]
        k += 1
    plan = [fill(x, word_values(f)) for x in g.raw.get("plan", g.raw.get("argument_plan", []))]
    lines += ["", f"## {k}. Plan"] + [f"* {x}" for x in plan] + [
        "", f"## {k + 1}. Rules",
        f"* Do not invent {f.word('source_kinds')}, {facts_w}, numbers, dates, designs or quotations.",
        f"* Keep the meaning of every {action} exact, but phrase it in your own words and vary the wording.",
        f"* State each {fact} at least once and at most {g.raw.get('max_mentions_per_fact', 3)} times.",
        "* Copy every number exactly as the Numbers table gives it; pooled values are already computed."]
    return idea, "\n".join(lines) + "\n"


# ------------------------------------------------------------------ v3 (23 Nov 2026): content plan per paper
APPROACH = ["{K} of the {SP} listed in the log.", "{K}. It reads the {SP} of the log side by side.",
            "{K}, built on the {SP} that the log gives.", "{K}: what the {SP} of the log show, and where they part."]
CONTRIB = ["An answer to the question: {Q}", "A reasoned position on this question: {Q}",
           "The evidence on one question, weighed: {Q}", "{Q} The paper answers this from the {SP} it reads."]


def materials_v3(w: dict[str, Any], f: Field, g: Genre, pid: str, card: dict[str, Any],
                 cp: dict[str, Any]) -> tuple[str, str]:
    """Notes from the paper's content plan (plan.py): only the numbers this paper reports, its own table designs,
    the facts in the plan's order with the section that states each first, and an outline per section in place of
    the genre's fixed plan lines. The row format is the one PaperOrchestra's fact_check reads (as in v2)."""
    import random

    from paperfactory import plan as plan_
    p = w["papers"][pid]
    fact, facts_w, action = f.word("fact"), f.word("facts"), f.word("action")
    rng = random.Random(f"{cp['design_seed']}:notes:{pid}")
    ents = sorted({x for s, _, o in cp["order"] for x in (s, o)})
    rows, tables = plan_.number_rows(w, f, g, pid, cp, _row_prefix(f, pid))
    kind_text = card["paper_kind"][:1].upper() + card["paper_kind"][1:]
    fillw = {"K": kind_text, "SP": f.word("subject_plural"), "Q": p["issue"]}
    idea = f"""# {p['title_hint']}

## Problem Statement
{p['issue']}

## Core Thesis
{p['thesis']}

## Approach
{rng.choice(APPROACH).format(**fillw)}

## Expected Contribution
{textwrap.fill(rng.choice(CONTRIB).format(**fillw), 100)}
"""
    lines = ["# Experimental Log", "", "## 1. Setting",
             f"* **Paper type:** {card['paper_kind']}.",
             f"* **Setting:** {f.raw['setting']}",
             f"* **Question:** {p['issue']}", f"* **Main conclusion to support:** {p['thesis']}", "", "## 2. Sources"]
    kinds = list(f.kinds)
    rng.shuffle(kinds)
    for kind in kinds:
        ids = [e for e in ents if w["entities"][e]["kind"] == kind]
        rng.shuffle(ids)
        if ids:
            lines += [f"### {kind.replace('_', ' ').capitalize()}", "| Name | Short names | Details |", "|---|---|---|"]
            lines += [f"| {_name(w, e)} | {', '.join(w['entities'][e].get('alt', []))} | {_attrs(w, f, e)} |"
                      for e in ids]
            lines.append("")
    where = {tuple(x["fact"]): x for x in cp["placement"]}
    lines += [f"## 3. {facts_w.capitalize()} (data; state each in your own words)",
              f"{action.capitalize()} meanings: " + "; ".join(f"{d['words']} = {d['meaning']}"
                                                            for d in f.relations.values()) + ".",
              "", f"| {f.word('subject_kind').capitalize()} | {action.capitalize()} | Source | State it first in | As |",
              "|---|---|---|---|---|"]
    lines += [f"| {_name(w, s)} | {f.rel_words(r)} | {_name(w, o)} | {where[(s, r, o)]['section']} | "
              f"{where[(s, r, o)]['form']} |" for s, r, o in cp["order"]]
    lines += ["", "## 4. Numbers (data; every value exactly as written, in its placement)",
              "| Row id | Item | Value | Unit | Placement |", "|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['label']} | {r['value']} | {r['unit']} | table |" for r in rows]
    lines += ["", f"## 5. Tables to build (exactly {len(tables)}; this paper's own designs)"]
    lines += [f"* Table {i + 1}, in section \"{tb['section']}\": {tb['title']}. Layout: {tb['layout']}. Numbers rows "
              f"ending in: {', '.join(x.rsplit('-', 1)[1] for x in tb['rows'])}."
              for i, tb in enumerate(tables)]
    lines += ["* Write your own column headers from the item names; put each unit in its header.",
              "* Row ids are for you only: never print a row id in the paper."]
    k = 6
    if p.get("conflict"):
        (s1, r1, o1), (s2, r2, _) = p["conflict"]
        lines += ["", f"## {k}. Conflict", f"* {_name(w, s1)} ({f.rel_words(r1)}) and {_name(w, s2)} "
                  f"({f.rel_words(r2)}) disagree about {_name(w, o1)}."]
        k += 1
    lines += ["", f"## {k}. Outline (what goes in each section of this paper)"]
    lines += [f"* {x}" for x in plan_.outline(w, f, cp, tables, p["issue"], p["thesis"])]
    lines += ["", f"## {k + 1}. Rules",
              f"* Do not invent {f.word('source_kinds')}, {facts_w}, numbers, dates, designs or quotations.",
              f"* Keep the meaning of every {action} exact, but phrase it in your own words and vary the wording.",
              f"* State each {fact} first in the section its row names, in the form its row names. It may come up "
              f"again elsewhere: at most {g.raw.get('max_mentions_per_fact', 3)} times in all.",
              "* Copy every number exactly as the Numbers table gives it; pooled values are already computed.",
              "* Report only the numbers of the Numbers table: this paper leaves the others out."]
    return idea, "\n".join(lines) + "\n"


def write_all(w: dict[str, Any], f: Field, g: Genre, out: Path,
              cards: dict[str, dict[str, Any]] | None = None, seed: Any = None) -> list[str]:
    """cards (style.cards): with a genre that has numbers, the notes per paper from its content plan (v3, when the
    genre says "content_plan": true; the plan is saved as content_plan.json) or the v2 notes; else the v1 notes."""
    from paperfactory import plan as plan_
    done = []
    for pid in w["papers"]:
        d = out / "papers" / pid
        d.mkdir(parents=True, exist_ok=True)
        if cards and g.raw.get("numeric_facts") and g.raw.get("content_plan"):
            from paperfactory.packs import card_template
            tmpl = card_template(g, cards[pid])
            cp = plan_.make(w, f, g, pid, tmpl, seed)
            idea, log = materials_v3(w, f, g, pid, cards[pid], cp)
            (d / "content_plan.json").write_text(json.dumps(cp, indent=1, ensure_ascii=False), encoding="utf-8")
            # the number rows with what each one is (world ids): the benchmark's gold rows; never given to a writer
            rows = plan_.number_rows(w, f, g, pid, cp, _row_prefix(f, pid))[0]
            (d / "numbers.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        elif cards and g.raw.get("numeric_facts"):
            idea, log = materials_v2(w, f, g, pid, cards[pid])
        else:
            idea, log = materials(w, f, g, pid)
        (d / "idea_sparse.md").write_text(idea, encoding="utf-8")
        (d / "experimental_log.md").write_text(log, encoding="utf-8")
        done.append(pid)
    return done
