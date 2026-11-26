"""Blind read-back: writer precision of a merged benchmark domain (spec Part C, 26 Nov 2026).

A reader model that is not the writer (Sonnet, claude_cli) reads one paper section per call and extracts every
factual statement about named things with the five shared fact tools (thesis_scorer.facttools, facttools_v1.json).
It never sees the planted facts, the question or the entity list. Code then:

  checks      each call (quote in the section, values in the quote, numbers parse, ...); a failed call is sent back
              once with its errors (repair call); still failing -> dropped and counted as rejected
  resolves    every name to a world entity: exact (alias table with the paper's defined terms) -> mention (exactly
              one entity named inside the name) -> judge (separate call: name, quote, candidate entities of fitting
              kinds -> id | none | unknown) -> unresolved. The route is recorded.
  sorts       each read claim: planned (matches a planted claim, decoy, conflict or seed number of this paper; or,
              flagged other_paper_fact, of another paper; or a text property of the entity in entities.json),
              possible invented (matches nothing and names a world entity), background (no name resolves)
  measures    writer precision = planned / (planned + possible invented), exact (Clopper-Pearson) 95% interval, per
              writer model; reader recall on planted facts with a tag-step span inside a section that was read

A relation whose subject resolves to no entity but whose object does ("the studies found no gain from X") is a summary:
planned (flag group_summary) when every planted relation of the paper on that object has the read relation, else
possible invented (flag group_conflict).

Defaults for open questions (plan, 26 Nov 2026): a claim matching another paper's planted fact is planned (flag
other_paper_fact); a planted pair with the wrong relation is possible invented (flag wrong_relation, also counted
apart); the reader sees the field's relation list.

Sample: SAMPLE papers per domain, seed SEED, stratified by writer model (all papers if fewer). Every reply is cached by
input hash under runs/paperfactory/readback/<domain>/cache (llm.Caller) and reused.

  python -m paperfactory.readback --domain bench_medicine --data-dir ../../data [--sections 10] [--cache-only]
Output: <data>/seed/<domain>/readback.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from thesis_scorer import facttools as ft

from paperfactory import packs
from paperfactory.llm import json_schema, pmap
from paperfactory.numspans import clopper_pearson, mark_judged, writer_of
from paperfactory.text import AliasTable, _key, alias_table, defined_terms, find_mentions, norm, normalize_paper

READER_MODEL = "claude-haiku-5-5"      # Aditya (26 Nov 2026, later): no Sonnet; Haiku-judged rates are marked
# "same model as writer" (numspans.mark_judged). readback.json of 26 Nov was read by claude-sonnet-5-5 (v1 run).
READER_ROLE, JUDGE_ROLE = "tagger", "verifier"
PROMPT_VERSION = "readback_v1"
SAMPLE, SEED = 50, 2026
MAX_CHARS = 12000                       # a longer section is split at a paragraph
PARALLEL = 8
SKIP_HEADINGS = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:references|bibliography|works cited)\s*$", re.I)
SYMMETRIC = {"contradicts"}             # a reversed pair still states the fact
SUBJECT_TOOLS = {"relation": "subject", "effect": "subject"}

READER_SYS = """Task: extract every factual statement about named things from one section of a paper. Make one tool
call per statement. Use add_custom only when no other tool fits.

Input (JSON): "paper" is the paper id; "section" is the section text in Markdown (tables are rows between | bars).

Output: {{"calls": [...]}}, one item per tool call, in the order the statements appear. Each item has "tool" (the
tool name) and that tool's fields. Give an empty list when the section states no facts.

Rules:
- quote: copy one continuous span of the section character for character (a sentence or a table row); never "...".
- Names: write each name as it is written in the section.
- Numbers: copy each number as written (keep "%", "1,234", "-0.3"); one number per field; never compute, round or
  convert. Every table row that holds numbers is a statement.
- Use only the section; no outside knowledge.
- Skip sentences that only describe this paper itself (its aims, structure or plan) and headings.

Relations (for add_relation, and the optional relation of add_effect):
{relations}

Tools:

{tools}"""

REPAIR_SYS = """Task: fix tool calls that failed code checks. You get one section of a paper and the failed calls,
each with its errors. For each failed call, return one corrected call that passes the checks, or leave it out when the
section does not state it. Same tools and rules as before:
- quote: one continuous span of the section copied character for character; never "...".
- Names as written in the section; numbers copied as written, one number per field, and they must occur in the quote.

Input (JSON): "paper", "section", "failed" (list of {{"call": ..., "errors": [...]}}).
Output: {{"calls": [...]}} with the corrected calls only (empty when none can be fixed).

Relations: {relations}"""

JUDGE_SYS = """Task: for each name read from a paper, say which entity it refers to.

Input (JSON list): each item has "id", "name" (as written in the paper), "quote" (the sentence it came from) and
"candidates" (entities with "id", "names" and "kind").
Output: {"answers": [{"id": ..., "entity": ...}]} with one answer per item. "entity" is a candidate id when the name
clearly refers to that candidate; "none" when it refers to none of them (e.g. a generic phrase like "the trials" or a
thing not in the list); "unknown" when the quote does not settle it.

Examples:
- name "ZIMOR-3", candidates [{"id": "TR37", "names": ["the ZIMOR-3 trial"]}] -> "TR37"
- name "patients with type 2 diabetes", same candidates -> "none"
- name "the trial", quote "The trial was stopped early.", two trial candidates -> "unknown"
"""

JUDGE_OUT = {"type": "object", "properties": {"answers": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "entity": {"type": "string"}},
    "required": ["id", "entity"], "additionalProperties": False}}}, "required": ["answers"],
    "additionalProperties": False}


# ------------------------------------------------------------------ sections
@dataclass(frozen=True)
class Section:
    start: int                 # offsets in paper.md
    end: int
    heading: str

    def text(self, md: str) -> str:
        return md[self.start:self.end]


HEAD = re.compile(r"(?m)^(#{1,2})\s+(.*?)\s*#*\s*$")


def sections(md: str, max_chars: int = MAX_CHARS) -> list[Section]:
    """h1/h2 blocks (h3+ stay inside); the title block joins the next block (title + abstract); References are
    skipped; a block longer than max_chars is split at paragraph breaks."""
    heads = [(m.start(), m.group(2)) for m in HEAD.finditer(md)]
    if not heads or heads[0][0] > 0 and md[:heads[0][0]].strip():
        heads = [(0, "")] + heads
    blocks = [[a, heads[i + 1][0] if i + 1 < len(heads) else len(md), h] for i, (a, h) in enumerate(heads)]
    if len(blocks) > 1 and md[blocks[0][0]:].startswith("# "):
        blocks[1] = [blocks[0][0], blocks[1][1], f"{blocks[0][2]} / {blocks[1][2]}"]
        blocks = blocks[1:]
    out: list[Section] = []
    for a, b, h in blocks:
        if SKIP_HEADINGS.match(h) or not md[a:b].strip():
            continue
        while b - a > max_chars:
            cut = md.rfind("\n\n", a + 1, a + max_chars)
            cut = cut if cut > a else a + max_chars
            out.append(Section(a, cut, h))
            a = cut
        out.append(Section(a, b, h))
    return out


# ------------------------------------------------------------------ reading
def relations_text(f: packs.Field) -> str:
    return "\n".join(f"- {r}: {v.get('meaning', '')}" for r, v in f.relations.items())


class Reader:
    """Reader + repair + judge calls through one llm.Caller (cached by input hash)."""

    def __init__(self, caller, f: packs.Field, cache_only: bool = False):
        self.call, self.f, self.cache_only = caller, f, cache_only
        self.rel = list(f.relations)
        self.schema = ft.output_schema(f.relations)
        self.sys = READER_SYS.format(relations=relations_text(f), tools=ft.render(f.relations, f.name))
        self.repair_sys = REPAIR_SYS.format(relations=", ".join(self.rel))
        self.not_cached = 0

    def _ask(self, role: str, step: str, system: str, user: str, out: dict[str, Any], check=None):
        if not self.cache_only:
            return self.call(role, step, system, user, out, check)
        rm = self.call.s.role(role)
        got = self.call.cached(self.call.cache / f"{self.call.key(rm, system, user, json_schema(out))}.json", out,
                               check)
        if got is None:
            self.not_cached += 1
        return got

    def read(self, pid: str, sec: Section, md: str) -> dict[str, Any]:
        """{"read": bool, "calls": [valid calls with flags], "rejected": [...], "repaired": n, "first_errors": n}."""
        text = sec.text(md)
        user = json.dumps({"paper": pid, "section": text}, ensure_ascii=False)
        got = self._ask(READER_ROLE, f"readback:{pid}:{sec.start}", self.sys, user, self.schema)
        if got is None:
            return {"read": False, "calls": [], "rejected": [], "repaired": 0, "dropped": 0, "first_errors": 0}
        ok, bad = self._split(got["calls"], pid, text)
        res = {"read": True, "calls": ok, "rejected": [], "repaired": 0, "dropped": 0, "first_errors": len(bad)}
        if bad:
            ruser = json.dumps({"paper": pid, "section": text, "failed": [{"call": c, "errors": e} for c, e in bad]},
                               ensure_ascii=False)
            fixed = self._ask(READER_ROLE, f"readback-repair:{pid}:{sec.start}", self.repair_sys, ruser, self.schema)
            ok2, bad2 = self._split(fixed["calls"] if fixed else [], pid, text)
            res["calls"] += [c | {"repaired": True} for c in ok2]
            res["repaired"] = len(ok2)
            res["dropped"] = max(0, len(bad) - len(ok2))
            # the first errors of every failed call, plus the errors of repaired calls that still fail
            res["rejected"] = [{"call": c, "errors": e, "step": "first"} for c, e in bad] + [
                {"call": c, "errors": e, "step": "repair"} for c, e in bad2]
        return res

    def _split(self, calls: list[dict[str, Any]], pid: str, text: str) -> tuple[list, list]:
        ok, bad = [], []
        for c in calls:
            chk = ft.check_call(c, pid, text, self.rel)
            if chk.errors:
                bad.append((c, chk.errors))
            else:
                ok.append(c | {"flags": chk.flags})
        return ok, bad

    def judge(self, pid: str, sec_start: int, items: list[dict[str, Any]]) -> dict[str, str]:
        """items: {id, name, quote, candidates} -> {id: entity id | "none" | "unknown"}; missing on failure."""
        if not items:
            return {}
        allowed = {it["id"]: {c["id"] for c in it["candidates"]} | {"none", "unknown"} for it in items}

        def check(out: dict[str, Any]) -> str | None:
            got = {a["id"]: a["entity"] for a in out["answers"]}
            if set(got) != set(allowed):
                return f"answer every id exactly once: {sorted(allowed)}"
            wrong = [i for i, e in got.items() if e not in allowed[i]]
            return f"entity must be a candidate id, none or unknown (items {wrong})" if wrong else None
        got = self._ask(JUDGE_ROLE, f"readback-judge:{pid}:{sec_start}", JUDGE_SYS,
                        json.dumps(items, ensure_ascii=False, indent=1), JUDGE_OUT, check)
        return {a["id"]: a["entity"] for a in got["answers"]} if got else {}


# ------------------------------------------------------------------ names
def claim_names(c: dict[str, Any]) -> list[tuple[str, str]]:
    """(field, name) of a read claim."""
    if c["kind"] == "custom":
        return [("names", n) for n in c["names"]]
    out = [(k, c[k]) for k in ("subject", "object") if c.get(k)]
    return out + [("studies", n) for n in c.get("studies") or []]


def fitting_kinds(f: packs.Field, c: dict[str, Any], fld: str) -> tuple[str, set[str]] | None:
    """Entity kinds a name in this field can refer to: ("in", kinds), ("not_in", kinds) or None (any kind)."""
    subj = set(f.subject_kinds)
    if fld == "studies" or fld == "subject" and c["kind"] in ("relation", "effect"):
        return "in", subj
    if c["kind"] == "relation" and fld == "object" and c.get("relation") in f.relations:
        kinds = set(f.relations[c["relation"]].get("object_kinds", []))
        return ("in", kinds) if kinds else None
    if c["kind"] == "effect" and fld == "object" or c["kind"] == "pool" and fld == "subject":
        return "not_in", subj
    return None


def resolve_code(name: str, table: AliasTable) -> tuple[str | None, str, list[str]]:
    """(entity id or None, route, candidate ids for the judge)."""
    k = _key(name.strip())
    if k in table.by_key:
        return table.by_key[k], "exact", []
    for prefix in ("the ", "The "):            # "the KUSOBRER-3 trial" written without or with an article
        if k.startswith(prefix.lower()) and _key(k[4:]) in table.by_key:
            return table.by_key[_key(k[4:])], "exact", []
        if _key(prefix.lower() + name.strip()) in table.by_key:
            return table.by_key[_key(prefix.lower() + name.strip())], "exact", []
    if k in table.ambiguous:
        return None, "ambiguous", list(table.ambiguous[k])
    ids = list(dict.fromkeys(m.eid for m in find_mentions(name, table)))
    if len(ids) == 1:
        return ids[0], "mention", []
    if len(ids) > 1:
        return None, "ambiguous", ids
    return None, "no_match", []


def paper_prefix(name: str, in_paper: list[str], ents: dict[str, Any]) -> tuple[str | None, str]:
    """A short name that starts exactly one name of an entity this paper mentions ("Section 2" -> "section 2 of the
    Tetre Data Privacy Act" when the paper names no other section 2)."""
    k = norm(name).removeprefix("the ").strip()
    if len(k) < 3:
        return None, "no_match"
    hits = {e for e in in_paper for n in [ents[e]["label"], *ents[e].get("alt", [])]
            if norm(n).removeprefix("the ").startswith(k + " ")}
    return (hits.pop(), "paper_prefix") if len(hits) == 1 else (None, "no_match")


# ------------------------------------------------------------------ matching
def _dec(v: Any) -> Decimal | None:
    try:
        return Decimal(str(v)) if v is not None else None
    except InvalidOperation:
        return None


def value_match(read: str | None, fact: Any, quote: str = "", fact_unit: str = "") -> tuple[bool, list[str]]:
    """Read value vs planted value (both canonical). Rounding: within half a unit of the read value's last
    decimal (flag rounded). Percent and fraction only if the quote has "%" (flag pct_scaled); thousands written as
    millions (flag scaled)."""
    r, f = _dec(read), _dec(ft.canon(str(fact)) if fact is not None else None)
    if r is None or f is None:
        return False, []
    if r == f:
        return True, []
    exp = r.as_tuple().exponent
    half = Decimal(5).scaleb(exp - 1) if isinstance(exp, int) else Decimal(0)
    if abs(r - f) <= half and -exp < -(f.normalize().as_tuple().exponent):
        return True, ["rounded"]
    if "%" in quote and (r / 100 == f or r * 100 == f):
        return True, ["pct_scaled"]
    if "thousand" in (fact_unit or "").lower() and "million" in quote.lower() and r * 1000 == f:
        return True, ["scaled"]
    return False, []


def entities_of(r: dict[str, Any]) -> set[str]:
    return {x for x in [r.get("subject"), r.get("object"), *(r.get("studies") or [])] if x}


NUM_KINDS = ("attr", "effect", "pool")


def match_claim(c: dict[str, Any], ids: dict[str, list[str | None]], rows: list[dict[str, Any]],
                ents: dict[str, Any], loose: bool = True) -> tuple[list[str], list[str]] | None:
    """(matched fact ids, flags) or None. rows: planted facts (facts.jsonl) to match against. loose=False (other
    papers' facts): exact values only, no match across fact kinds."""
    first = lambda k: (ids.get(k) or [None])[0]  # noqa: E731
    s, o = first("subject"), first("object")
    named = {x for v in ids.values() for x in v if x}
    q = c.get("quote") or ""
    if c["kind"] == "relation":
        hit = [r["id"] for r in rows if r["what"] == "relation" and (r["subject"], r["relation"], r["object"])
               == (s, c["relation"], o)]
        if hit:
            return hit, []
        rev = [r["id"] for r in rows if r["what"] == "relation" and (r["subject"], r["relation"], r["object"])
               == (o, c["relation"], s)]
        if rev and c["relation"] in SYMMETRIC:
            return rev, ["reversed"]
        return None

    def number_rows(value: str | None, need: set[str], kinds=NUM_KINDS) -> tuple[list[str], list[str]]:
        hit, fl = [], []
        for r in rows:
            if r["what"] in kinds and need & entities_of(r):
                ok, f2 = value_match(value, r["value"], q, r.get("unit", ""))
                if ok and (loose or not f2):
                    hit.append(r["id"])
                    fl += f2
        return hit, sorted(set(fl))

    def entity_attr(value: Any, need: set[str], text: bool) -> bool:
        for e in need:
            for av in (ents.get(e, {}).get("attrs") or {}).values():
                if text and isinstance(av, str) and av in ents:          # an attribute that is another entity
                    if any(norm(n) in norm(str(value)) or norm(str(value)) in norm(n)
                           for n in [ents[av]["label"], *ents[av].get("alt", [])]):
                        return True
                    continue
                if text and isinstance(av, str) and (norm(str(value)) == norm(av) or norm(av) in norm(str(value))
                                                     or words_match(av, str(value), ents.get(e, {}))):
                    return True
                if not text and not isinstance(av, str) and value_match(value, av)[0]:
                    return True
                if not text and isinstance(av, str) and ft.value_of(av) is not None and \
                        value_match(value, ft.value_of(av))[0]:
                    return True
        return False
    if c["kind"] == "attr":
        if not s:
            return None
        if c.get("text_value"):
            return ([], ["entity_attr"]) if entity_attr(c["value"], {s}, True) else None
        hit, fl = number_rows(c["value"], {s}, ("attr",))
        hit = [h for h in hit if next(r for r in rows if r["id"] == h)["subject"] == s]
        if hit:
            return hit, fl
        hit, fl = number_rows(c["value"], {s}) if loose else ([], [])
        if hit:
            return hit, fl + ["kind_mixup"]
        return ([], ["entity_attr"]) if entity_attr(c["value"], {s}, False) else None
    if c["kind"] in ("effect", "pool"):
        if not s:
            return None
        cands = [r for r in rows if r["what"] == c["kind"] and r["subject"] == s
                 and (c["kind"] == "pool" or o is None or r.get("object") == o)]
        for same_stat in (True, False):
            hit, fl = [], []
            for r in cands:
                if (r.get("stat") == c["stat"]) == same_stat:
                    ok, f2 = value_match(c["value"], r["value"], q, r.get("unit", ""))
                    if ok and (loose or not f2):
                        hit.append(r["id"])
                        fl += f2
            if hit:
                extra = ([] if same_stat else ["stat_mixup"]) + (["partial"] if c["kind"] == "effect" and o is None
                                                                  else [])
                return hit, sorted(set(fl)) + extra
        hit, fl = number_rows(c["value"], named) if loose else ([], [])
        return (hit, fl + ["kind_mixup"]) if hit else None
    # custom
    if not named:
        return None
    nums = [n for n in c["numbers"] if n.get("value") is not None]
    if nums:
        hit, fl, ok_all = [], [], True
        for n in nums:
            h, f2 = number_rows(n["value"], named)
            if h:
                hit += h
                fl += f2
            elif entity_attr(n["value"], named, False):
                fl.append("entity_attr")
            else:
                ok_all = False
        return (hit, sorted(set(fl)) + ["custom_number"]) if ok_all else None
    pair = [r["id"] for r in rows if r["what"] == "relation" and {r["subject"], r["object"]} <= named]
    if pair:
        return pair, ["custom_pair"]
    if any(av in named for e in named for av in (ents.get(e, {}).get("attrs") or {}).values()
           if isinstance(av, str)):
        return [], ["entity_link"]            # e.g. a decision and its court, a study and its model
    if entity_attr(c.get("statement") or "", named, True):
        return [], ["entity_attr"]
    return None


STOP = {"with", "from", "that", "this", "which", "were", "used", "uses", "using", "their", "after", "before", "into",
        "about", "over", "under", "between", "than", "have", "been", "being", "each", "other", "case", "cases"}


def _stems(text: str, drop: set[str] = frozenset()) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z0-9]+", norm(text)) if len(w) >= 4 and w not in STOP} - drop


def words_match(attr_value: str, text: str, ent: dict[str, Any]) -> bool:
    """A text property of an entity restated in other words: at least 2/3 of its content words (first 5 letters,
    words of the entity's own names left out) occur in text, or text is a part of it (2+ content words, all in it)."""
    own = _stems(" ".join([ent.get("label", ""), *ent.get("alt", [])]))
    need, got = _stems(attr_value, own), _stems(text, own)
    return bool(need) and (len(need & got) >= max(1, math.ceil(2 * len(need) / 3)) or len(got) >= 2 and got <= need)


def sort_claim(c: dict[str, Any], ids: dict[str, list[str | None]], own: list[dict[str, Any]],
               others: list[dict[str, Any]], ents: dict[str, Any]) -> dict[str, Any]:
    """-> {"sort": planned | possible_invented | background, "facts": [...], "flags": [...]}."""
    named = {x for v in ids.values() for x in v if x}
    if (m := match_claim(c, ids, own, ents)) is not None:
        return {"sort": "planned", "facts": m[0], "flags": m[1]}
    if (m := match_claim(c, ids, others, ents, loose=False)) is not None:
        return {"sort": "planned", "facts": m[0], "flags": m[1] + ["other_paper_fact"]}
    if c["kind"] == "relation":
        s, o = (ids.get("subject") or [None])[0], (ids.get("object") or [None])[0]
        pair = [r["id"] for r in own if r["what"] == "relation" and {r["subject"], r["object"]} == {s, o}]
        if pair:
            same_dir = any((r["subject"], r["object"]) == (s, o) for r in own if r["id"] in pair)
            return {"sort": "possible_invented", "facts": [], "flags": ["wrong_relation" if same_dir else "reversed"]}
        if s is None and o is not None:       # "the studies found ...": a summary over planted facts on the object
            on = [r for r in own if r["what"] == "relation" and r["object"] == o]
            if on and all(r["relation"] == c["relation"] for r in on):
                return {"sort": "planned", "facts": [], "flags": ["group_summary"]}
            if on:
                return {"sort": "possible_invented", "facts": [], "flags": ["group_conflict"]}
    if named:
        return {"sort": "possible_invented", "facts": [], "flags": []}
    if c["kind"] in NUM_KINDS and c.get("value") is not None and not c.get("text_value"):
        hit = [r["id"] for r in own if r["what"] in NUM_KINDS and value_match(c["value"], r["value"],
                                                                                c.get("quote") or "")[0]]
        if hit:
            return {"sort": "planned", "facts": [], "flags": ["value_only"], "value_facts": hit}
    return {"sort": "background", "facts": [], "flags": []}


# ------------------------------------------------------------------ one paper
def read_paper(reader: Reader, f: packs.Field, w: dict[str, Any], base: AliasTable, p: dict[str, Any], md: str,
               own: list[dict[str, Any]], others: list[dict[str, Any]], secs: list[Section] | None = None
               ) -> dict[str, Any]:
    ents = w["entities"]
    nt = normalize_paper(md).text
    table = alias_table(w, f, defined_terms(nt, base, f))
    # entities of this paper: named in it by an unambiguous name, or in its planted facts (a name the world gives two
    # entities, e.g. "section 1" or duplicated provisions of the merged domain, is then settled by the paper)
    in_paper = list(dict.fromkeys([m.eid for m in find_mentions(nt, table)] + [e for r in own for e in
                                                                               sorted(entities_of(r))]))
    secs = sections(md) if secs is None else secs
    out_secs, claims = [], []
    for sec in secs:
        r = reader.read(p["id"], sec, md)
        out_secs.append({"start": sec.start, "end": sec.end, "heading": sec.heading, "read": r["read"],
                         "calls": len(r["calls"]), "first_errors": r["first_errors"], "repaired": r["repaired"],
                         "dropped": r["dropped"], "rejected": r["rejected"]})
        rows: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        for ci, call in enumerate(r["calls"]):
            for cl in ft.read_claims(call):
                cl |= {"paper": p["id"], "section": sec.heading, "section_start": sec.start, "tool": call["tool"],
                       "call_flags": call.get("flags", []), "repaired": call.get("repaired", False), "call": ci,
                       "ids": {}, "routes": {}}
                for fld, name in claim_names(cl):
                    eid, route, cand = resolve_code(name, table)
                    if eid is None and route == "ambiguous" and len(here := [e for e in cand if e in in_paper]) == 1:
                        eid, route = here[0], "ambiguous_paper"
                    if eid is None and route in ("no_match", "ambiguous"):
                        e2, r2 = paper_prefix(name, in_paper, ents)
                        eid, route = (e2, r2) if e2 else (eid, route)
                    cl["ids"].setdefault(fld, []).append(eid)
                    cl["routes"].setdefault(fld, []).append(route)
                    if eid is None:
                        kinds = fitting_kinds(f, cl, fld)
                        pool = cand or in_paper
                        cs = [e for e in pool if kinds is None or (ents[e]["kind"] in kinds[1]) == (kinds[0] == "in")]
                        if cs:
                            pending.append({"claim": cl, "fld": fld, "i": len(cl["ids"][fld]) - 1, "name": name,
                                            "cands": cs})
                rows.append(cl)
        items = [{"id": f"N{k + 1}", "name": x["name"], "quote": x["claim"]["quote"],
                  "candidates": [{"id": e, "names": [ents[e]["label"], *ents[e].get("alt", [])],
                                  "kind": ents[e]["kind"]} for e in x["cands"]]} for k, x in enumerate(pending)]
        ans = reader.judge(p["id"], sec.start, items)
        for k, x in enumerate(pending):
            a = ans.get(f"N{k + 1}")
            route = "judge" if a and a not in ("none", "unknown") else f"judge_{a}" if a else "judge_failed"
            x["claim"]["ids"][x["fld"]][x["i"]] = a if route == "judge" else None
            x["claim"]["routes"][x["fld"]][x["i"]] = route
        for cl in rows:
            cl |= sort_claim(cl, cl["ids"], own, others, ents)
        claims += rows
    return {"paper": p["id"], "writer": p.get("writer") or writer_of(p), "sections": out_secs, "claims": claims}


# ------------------------------------------------------------------ recall
def planted_spans(p: dict[str, Any]) -> dict[str, tuple[str, list[list[int]]]]:
    """facts.jsonl id -> (kind, raw spans) from paperfactory.json (relation facts and number facts)."""
    out = {f"{p['id']}:{x['id']}": (f"relation_{x['kind']}", x.get("spans") or []) for x in p.get("facts", [])}
    out |= {x["id"]: (x["kind"], x.get("spans") or []) for x in p.get("numbers", [])}
    return out


def recall(res: dict[str, Any], p: dict[str, Any]) -> list[dict[str, Any]]:
    """Per planted fact with a tag-step span inside a section that was read: found by a planned claim or not."""
    read = [(s["start"], s["end"]) for s in res["sections"] if s["read"]]
    found = {i for c in res["claims"] if c["sort"] == "planned" for i in c["facts"]}
    out = []
    for fid, (kind, spans) in planted_spans(p).items():
        if any(a <= x and y <= b for x, y in spans for a, b in read):
            out.append({"fact": fid, "kind": kind, "found": fid in found})
    return out


# ------------------------------------------------------------------ one domain
def sample_papers(papers: list[dict[str, Any]], n: int = SAMPLE, seed: int = SEED) -> list[dict[str, Any]]:
    """n papers, stratified by writer model (largest remainder), fixed seed; all papers if there are n or fewer."""
    if len(papers) <= n:
        return list(papers)
    by: dict[str, list[dict[str, Any]]] = {}
    for p in papers:
        by.setdefault(p.get("writer") or writer_of(p), []).append(p)
    quota = {w: n * len(ps) / len(papers) for w, ps in by.items()}
    take = {w: int(q) for w, q in quota.items()}
    for w in sorted(quota, key=lambda w: (-(quota[w] - take[w]), w))[:n - sum(take.values())]:
        take[w] += 1
    rng = random.Random(seed)
    keep = {p["id"] for w in sorted(by) for p in rng.sample(by[w], take[w])}
    return [p for p in papers if p["id"] in keep]


def rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95_exact": clopper_pearson(k, n)}


def summarize(results: list[dict[str, Any]], rec: list[dict[str, Any]]) -> dict[str, Any]:
    claims = [c for r in results for c in r["claims"]]
    sorts = Counter(c["sort"] for c in claims)
    calls = [(c["paper"], c["section_start"], c["call"], c["tool"]) for c in claims]
    tools = Counter(t for *_, t in set(calls))
    secs = [s for r in results for s in r["sections"]]
    routes = Counter(x for c in {(c["paper"], c["section_start"], c["call"]): c for c in claims}.values()
                     for v in c["routes"].values() for x in v)          # per call, not per read claim
    return {
        "writer_precision": rate(sorts["planned"], sorts["planned"] + sorts["possible_invented"]),
        "reader_recall": rate(sum(x["found"] for x in rec), len(rec)),
        "reader_recall_by_kind": {k: rate(sum(x["found"] for x in rec if x["kind"] == k),
                                          sum(x["kind"] == k for x in rec)) for k in sorted({x["kind"] for x in rec})},
        "claims": dict(sorts), "background": sorts["background"],
        "calls": sum(tools.values()), "calls_by_tool": dict(tools),
        "add_custom_share": rate(tools["add_custom"], sum(tools.values())),
        "fact_kind_shares": {k: round(v / len(claims), 4) for k, v in Counter(c["kind"] for c in claims).items()}
        if claims else {},
        "rejected_calls": {"first_check_failed": sum(s["first_errors"] for s in secs),
                           "repaired": sum(s["repaired"] for s in secs),
                           "dropped": sum(s["dropped"] for s in secs)},
        "name_routes": dict(routes),
        "flags": dict(Counter(fl for c in claims for fl in c["flags"])),
        "call_flags": dict(Counter(fl for c in claims for fl in c["call_flags"])),
        "wrong_relation": sum("wrong_relation" in c["flags"] for c in claims),
        "sections": len(secs), "sections_read": sum(s["read"] for s in secs), "papers": len(results)}


def run(seed_dir: Path, ds_dir: Path, caller, cache_only: bool = False, max_sections: int | None = None,
        papers: list[str] | None = None) -> dict[str, Any]:
    man = json.loads((seed_dir / "paperfactory.json").read_text(encoding="utf-8"))
    ents = json.loads((seed_dir / "entities.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (seed_dir / "facts.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    f = packs.field(man["field"])
    w = {"entities": ents}
    base = alias_table(w, f)
    reader = Reader(caller, f, cache_only)
    pick = sample_papers(man["papers"]) if papers is None else [p for p in man["papers"] if p["id"] in papers]
    mds = {p["id"]: (ds_dir / p["id"] / "paper.md").read_text(encoding="utf-8") for p in pick}
    plan = {p["id"]: sections(mds[p["id"]]) for p in pick}
    if max_sections is not None:                     # cost probe: the first sections of the first papers
        left, cut = max_sections, {}
        for p in pick:
            cut[p["id"]], left = plan[p["id"]][:left], left - len(plan[p["id"]][:left])
        plan = {k: v for k, v in cut.items() if v}
        pick = [p for p in pick if p["id"] in plan]

    def one(p: dict[str, Any]) -> dict[str, Any]:
        own = [r for r in rows if r["paper"] == p["id"]]
        others = [r for r in rows if r["paper"] != p["id"]]
        return read_paper(reader, f, w, base, p, mds[p["id"]], own, others, plan[p["id"]])
    results = pmap(one, pick, PARALLEL)
    rec_by = {r["paper"]: recall(r, p) for r, p in zip(results, pick)}
    rec = [x | {"paper": k} for k, v in rec_by.items() for x in v]
    by_writer = {}
    for wr in sorted({r["writer"] for r in results}):
        rs = [r for r in results if r["writer"] == wr]
        by_writer[wr] = summarize(rs, [x for x in rec if any(r["paper"] == x["paper"] for r in rs)])
    out = {"domain": seed_dir.name, "field": f.name, "prompt_version": PROMPT_VERSION, "tools_version": ft.VERSION,
           "reader_model": caller.s.model(READER_ROLE), "judge_model": caller.s.model(JUDGE_ROLE),
           "sample": {"n": SAMPLE, "seed": SEED, "papers": [p["id"] for p in pick],
                      "writers": dict(Counter(r["writer"] for r in results)), "of": len(man["papers"])},
           "defaults": {"other_paper_fact": "planned, flagged", "wrong_relation": "possible invented, counted apart",
                        "relation_list_shown": True,
                        "value_only": "a number claim with no resolved name whose value is a planted number of the "
                                      "paper: planned, flagged value_only (not used for recall)"},
           "interval": "exact (Clopper-Pearson) 95%",
           "summary": summarize(results, rec), "by_writer": by_writer,
           "not_cached": reader.not_cached, "cost": caller.cost("readback"),
           "possible_invented": [_brief(c) for r in results for c in r["claims"] if c["sort"] == "possible_invented"],
           "recall_misses": [x for x in rec if not x["found"]],
           "papers": results}
    mark_judged(out["summary"], out["reader_model"])
    mark_judged(out["by_writer"], out["reader_model"])
    return out


def pooled(readbacks: list[dict[str, Any]]) -> dict[str, Any]:
    """Writer precision and reader recall summed over domains (and per writer), exact 95% intervals."""
    def add(get) -> dict[str, Any]:
        ks = [get(rb) for rb in readbacks]
        ks = [x for x in ks if x]
        return rate(sum(x["k"] for x in ks), sum(x["n"] for x in ks))
    writers = sorted({w for rb in readbacks for w in rb["by_writer"]})
    calls = sum(rb["summary"]["calls"] for rb in readbacks)
    return {"domains": [rb["domain"] for rb in readbacks],
            "writer_precision": add(lambda rb: rb["summary"]["writer_precision"]),
            "reader_recall": add(lambda rb: rb["summary"]["reader_recall"]),
            "by_writer": {w: {"writer_precision": add(lambda rb: rb["by_writer"].get(w, {}).get("writer_precision")),
                              "reader_recall": add(lambda rb: rb["by_writer"].get(w, {}).get("reader_recall"))}
                          for w in writers},
            "add_custom_share": rate(sum(rb["summary"]["calls_by_tool"].get("add_custom", 0) for rb in readbacks),
                                     calls),
            "background": sum(rb["summary"]["background"] for rb in readbacks),
            "rejected_calls": dict(sum((Counter(rb["summary"]["rejected_calls"]) for rb in readbacks), Counter())),
            "name_routes": dict(sum((Counter(rb["summary"]["name_routes"]) for rb in readbacks), Counter())),
            "cost_usd": round(sum(rb["cost"]["usd_known"] for rb in readbacks), 2)}


def _brief(c: dict[str, Any]) -> dict[str, Any]:
    keep = ("paper", "section", "kind", "tool", "quote", "subject", "relation", "object", "attr", "value", "stat",
            "unit", "measure", "studies", "statement", "names", "numbers", "ids", "routes", "flags")
    return {k: c[k] for k in keep if c.get(k) not in (None, [], {})}


def caller_for(domain: str, field_name: str, genre: str):
    from paperfactory.config import Settings
    from paperfactory.llm import Caller
    s = Settings(field=field_name, genre=genre or "argument",
                 models={READER_ROLE: {"model": READER_MODEL, "max_parallel": PARALLEL},
                         JUDGE_ROLE: {"model": READER_MODEL, "max_parallel": PARALLEL}},
                 out=f"runs/paperfactory/readback/{domain}")
    return Caller(s)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m paperfactory.readback", description=__doc__.split("\n")[0])
    ap.add_argument("--domain", help="one domain (bench_law, ...)")
    ap.add_argument("--pool", nargs="+", metavar="DOMAIN", help="print the pooled summary of these domains' "
                    "readback.json (no model calls)")
    ap.add_argument("--data-dir", default="../../data")
    ap.add_argument("--sections", type=int, default=None, help="cost probe: read only the first N sections")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    data = Path(a.data_dir)
    if a.pool:
        rbs = [json.loads((data / "seed" / d / "readback.json").read_text(encoding="utf-8")) for d in a.pool]
        print(json.dumps(pooled(rbs), indent=1))
        return
    if not a.domain:
        ap.error("--domain or --pool is required")
    seed_dir, ds_dir = data / "seed" / a.domain, data / "dataset" / a.domain
    man = json.loads((seed_dir / "paperfactory.json").read_text(encoding="utf-8"))
    caller = caller_for(a.domain, man["field"], man.get("genre", "argument"))
    out = run(seed_dir, ds_dir, caller, a.cache_only, a.sections)
    s = out["summary"]
    print(json.dumps({"domain": a.domain, "sections": s["sections"], "read": s["sections_read"],
                      "precision": s["writer_precision"], "recall": s["reader_recall"], "cost": out["cost"],
                      "not_cached": out["not_cached"]}, indent=1))
    if not a.no_write and a.sections is None:
        (seed_dir / "readback.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
