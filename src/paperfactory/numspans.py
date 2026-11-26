"""Positions of number facts (attr, effect, pool) in a merged benchmark domain (26 Nov 2026).

Input: <data>/seed/<domain>/{facts.jsonl, entities.json, paperfactory.json} and <data>/dataset/<domain>/<pid>/paper.md.
Output: papers[].numbers in paperfactory.json (spans as character offsets in paper.md, like papers[].facts[].spans,
plus `found_by` per span) and a report, numspans.json, next to it.

A place is one sentence (paragraph or list item) or one table cell (its row and column header are its context).
Number formats are normalized before matching (canon): 9.0 = 9, .5 = 0.5, 3,050 = 3050, a unicode minus = "-".

Per fact, by code first:
  once+name   the value occurs once in the paper and the place names the subject, object or a pooled study
  once+unit   the value occurs once and the place states the unit (no name)
  several+name the value occurs several times; kept: the places that name the subject (an effect: subject and
              object; a pool: not a place naming exactly one study); a table cell only when no other cell of
              its row has the same value
  verifier    every other case: each place with the value goes to a separate yes/no model call. Each item has a
              code-made control: the same passage with a wrong value in the fact, which must get "no".
Hard control: the right value in a place about another subject's fact (e.g. another study's row with the same
value) must get "no" (up to HARD_MAX per domain, step "numhard:<pid>").
Checked after: the estimate and both CI bounds of one effect or pool should share a place (report "ci_mismatch").

Verifier answers are cached under runs/paperfactory/numspans/<domain>/cache (llm.Caller). cache_only=True (used by
ops/merge_bench.py) never calls a model: an uncached item stays unresolved and is counted as "not_cached".

  uv run paperfactory number-spans --domain bench_llm [--data-dir ../../data]
"""
from __future__ import annotations

import json
import math
from collections import Counter
import random
import re
import threading
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from paperfactory import packs
from paperfactory.llm import json_schema, pmap
from paperfactory.text import (AliasTable, SplitRules, _split_unit, alias_table, defined_terms, find_mentions,
                               normalize_paper, split_rules)

VERIFIER_MODEL = "claude-haiku-5-5"    # Aditya's choice (26 Nov 2026): the SAME model as the bench20h writers
# writer of each source run (configs/haiku/bench20-*.json: Claude Haiku; the others: Qwen behind the gateway)
WRITERS = (("bench20h-", "claude-haiku-5-5"), ("qwen40-", "qwen (orch-model)"), ("v2p3-", "qwen (orch-model)"))
BATCH = 25
PARALLEL = 24                          # papers at once (Aditya, 26 Nov 2026); the role's max_parallel caps calls
MAX_PLACES = 8                         # places per fact sent to the verifier
HARD_MAX, HARD_SEED = 100, 26          # hard controls per domain (right value, wrong thing)
AUDIT_RANDOM, AUDIT_SEED = 100, 27     # other several+name spans checked per domain
# the verifier's error rate as quoted (Aditya, 26 Nov 2026): first hard-control run, 13 'yes' of 400 classified by
# hand: 8 false controls (subject named in a column header), 5 real errors (LLM-06, MED-08, MED-31 x2, MED-44)
VERIFIER_ERROR_QUOTE = {"rate": "1.3% [0.4%, 2.9%]", "errors": 5, "n": 400, "interval": "exact (Clopper-Pearson) 95%",
                        "source": "hard_controls_v1, 13 accepted of 400, 8 judged false controls by hand"}
HARD_RULES = {
    "v1_row": "first run (26 Nov 2026): dropped a place when the fact's subject is named in its sentence or in the "
              "cells of its table row",
    "v2_seen": "current (26 Nov 2026): dropped a place when the fact's subject is named anywhere the verifier sees "
               "it (passage with column headers, and context)"}
HARD_WHY = ("v1 made false controls: in sideways tables (one column per case or study) the subject is named in the "
            "column header, so the passage did state the fact and 'yes' was right. v1 counted 13 'yes' of 400; most "
            "were such false controls; v2 dropped them")
CI = ("estimate", "ci_low", "ci_high")
NO_UNIT = {"n", "h", ""}               # too short or too common to count as unit evidence

NUM = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)?(?:\.\d+)?(?![\w]|\.\d)")
SIGN_BEFORE = set(" ([,:;=/\n")


# ------------------------------------------------------------------ numbers
def canon(s: str | None) -> str | None:
    """One form per number: no thousands commas, plain minus, no trailing zeros, leading zero. None if no number."""
    if s is None:
        return None
    t = s.strip().replace("−", "-").replace("‒", "-").replace("–", "-").replace(",", "").lstrip("+")
    try:
        d = Decimal(t)
    except InvalidOperation:
        return None
    out = format(d.normalize(), "f")
    return "0" if out in ("-0", "0") else out


def numbers_in(text: str) -> list[tuple[int, int, str]]:
    """(start, end, canonical value) of each number in text. A "-" right before a number is a minus sign only at
    the start or after a space or opening bracket (else it is a range dash: 7.3-10.8). A number joined to a word by
    "-" is part of a name and skipped."""
    out = []
    for m in NUM.finditer(text):
        if not m.group() or not any(c.isdigit() for c in m.group()):
            continue
        a = m.start()
        if a > 1 and text[a - 1] in "-−" and text[a - 2].isalpha():     # part of a name: Stetrugel-6, Thipom-7B
            continue
        neg = a > 0 and text[a - 1] in "-−" and (a == 1 or text[a - 2] in SIGN_BEFORE)
        v = canon(("-" if neg else "") + m.group())
        if v is not None:
            out.append((a - 1 if neg else a, m.end(), v))
    return out


def wrong_value(v: str, avoid: set[str]) -> str:
    """A code-made wrong number for the control: same number of decimals, not in `avoid` (values in the passage)."""
    d = Decimal(v)
    step = Decimal(1).scaleb(d.as_tuple().exponent) if d.as_tuple().exponent < 0 else Decimal(1)
    for k in (7, 13, 3, 29, 41, 53):
        for c in (d + k * step, d - k * step):
            if canon(str(c)) not in avoid and canon(str(c)) != canon(v):
                return canon(str(c))
    return canon(str(d * 10 + 1))


def unit_in(unit: str, text: str) -> bool:
    u = (unit or "").strip()
    if u.lower() in NO_UNIT:
        return False
    if u == "%":
        return bool(re.search(r"%|\bper ?cent", text, re.I))
    stem = u[:-1] if len(u) > 3 and u.endswith("s") else u
    return bool(re.search(rf"(?<!\w){re.escape(stem)}", text, re.I))


# ------------------------------------------------------------------ places
@dataclass
class Place:
    start: int                 # offsets in normalize_paper(md).text
    end: int
    text: str                  # the sentence, or the cell
    names_text: str            # where names count: the sentence, or the whole row
    unit_text: str             # where the unit counts: the sentence, or the cell + its column header
    context: str               # for the verifier: the sentence before, or the column headers
    passage: str               # for the verifier: the sentence, or "header: cell; ..." for the row
    row: list[str] = field(default_factory=list)   # cells of the row (table cells only)
    group: tuple[int, int] = (0, 0)                 # same group = same place for the CI check: the sentence, or the row
    marked: str | None = None                       # a cell as "header: cell" (the code audit asks about this cell)
    header: str = ""                                # a cell's column header


def places(md: str, rules: SplitRules) -> list[Place]:
    p = normalize_paper(md)
    line_of = lambda i: md.count("\n", 0, p.back[i])  # noqa: E731
    out: list[Place] = []
    prev = ""
    cells = [u for u in p.units if u.kind == "cell"]
    rows: dict[int, list[Any]] = {}
    for u in cells:
        rows.setdefault(line_of(u.start), []).append(u)
    header_of: dict[int, list[str]] = {}
    last, head = -9, None
    for ln in sorted(rows):
        if ln - last > 2:                        # a new table (rows are 1 apart; 2 across the |---| line)
            head = [p.text[u.start:u.end] for u in rows[ln]]
        header_of[ln] = head
        last = ln
    for u in p.units:
        if u.kind == "heading":
            continue
        if u.kind == "cell":
            ln = line_of(u.start)
            row = rows[ln]
            texts = [p.text[x.start:x.end] for x in row]
            head = header_of[ln]
            col = row.index(u)
            h = head[col] if head and len(head) == len(row) and head != texts else ""
            pairs = "; ".join(f"{a}: {b}" for a, b in zip(head, texts)) if head and len(head) == len(row) \
                else " | ".join(texts)
            out.append(Place(u.start, u.end, texts[col], " | ".join(texts), f"{h} {texts[col]}",
                             "columns: " + " | ".join(head or []), pairs, texts, (row[0].start, row[-1].end),
                             f"{h}: {texts[col]}" if h else texts[col], h))
            continue
        t = p.text[u.start:u.end]
        for a, b in _split_unit(t, rules):
            s = t[a:b]
            out.append(Place(u.start + a, u.start + b, s, s, s, prev, s, [], (u.start + a, u.start + b)))
            prev = s
    return out


# ------------------------------------------------------------------ verifier
class Verdict(BaseModel):
    id: str
    verdict: str


class VerOut(BaseModel):
    answers: list[Verdict]


VER_SYS = """You check number facts against a paper. Each item has a passage from the paper (one sentence with the
sentence before it as context, or one table row as "column: cell" pairs) and one number fact: what the number is
about, its value and its unit.
Answer "yes" only if the passage itself states this value for exactly this thing: the same study or entity and the
same quantity. The same value for another quantity or another entity is "no". A value missing from the passage is
"no". Formats count as equal: 9.0 = 9, .5 = 0.5, 3,050 = 3050.
Output: {"answers": [{"id": "<item id>", "verdict": "yes" | "no"}]} with one answer per item.

Examples:
passage "Kor 2021 enrolled 412 patients."; fact "Kor 2021: patients = 412 n" -> yes
passage "Kor 2021 enrolled 412 patients over 12 weeks."; fact "Kor 2021: follow-up = 412 weeks" -> no (412 is the
patients, not the follow-up)
passage "columns: Study | Estimate | Patients ... Study: Bel 2019; Estimate: 0.81; Patients: 300"; fact "Kor 2021:
estimate = 0.81 HR" -> no (the row is about Bel 2019)"""


AUDIT_SYS = VER_SYS + """

Some items also have "marked": one cell of the table row. Then judge only that cell: "yes" only if the marked cell
holds this value for exactly this thing. The right value in another cell of the row is "no".
passage "Study: Kor 2021; Models: 3; Seeds: 3"; marked "Seeds: 3"; fact "Kor 2021: models compared = 3 models" -> no
(the marked cell is the seeds)"""


# ------------------------------------------------------------------ gate before the verifier (v2, 26 Nov 2026)
INTERVAL = re.compile(r"(?:\(|\[|\bfrom\s|\bCI:?\s|\binterval\s(?:of\s)?|\bbetween\s)(?:95\s?%\s*CI[:,]?\s*)?"
                      r"(?P<lo>-?\d[\d.,]*\d|-?\d)\s*%?\s*(?:to|,|and|-)\s*(?P<hi>-?\d[\d.,]*\d|-?\d)", re.I)


def value_roles(text: str, v: str, header: str = "") -> set[str]:
    """The roles value v plays in text: ci_low / ci_high (first / second number of an interval: "(0.46 to 0.86)",
    "from -5.7% to -1.8%", "[1, 2]") or estimate (any other place). A lone number in a table cell takes its role
    from the column header ("lower" / "upper"). text is normalized (plain minus)."""
    lo = {m.start("lo") for m in INTERVAL.finditer(text)}
    hi = {m.start("hi") for m in INTERVAL.finditer(text)}
    h = header.lower()
    out = set()
    for a, _, x in numbers_in(text):
        if x != v:
            continue
        if a in lo or (a + 1 in lo and text[a] == "-"):
            out.add("ci_low")
        elif a in hi or (a + 1 in hi and text[a] == "-"):
            out.add("ci_high")
        elif not (lo or hi) and "lower" in h and "upper" not in h:
            out.add("ci_low")
        elif not (lo or hi) and "upper" in h and "lower" not in h:
            out.add("ci_high")
        else:
            out.add("estimate")
    return out


def gate(r: dict[str, Any], place: Place, seen: set[str]) -> str | None:
    """Code check before the verifier: None = pass, else the reason. The passage itself (a sentence, or a table row
    with its column headers; not the sentence before) must name the fact's subject; for an effect or pool estimate or CI bound, the value must play that role here."""
    if r.get("subject") not in seen:
        return "subject not named"
    if r["what"] in ("effect", "pool") and r.get("stat") in CI:
        roles = value_roles(place.text, canon(r["value"]) or "", place.header)
        if roles and r["stat"] not in roles:
            return f"stat: the value is {'/'.join(sorted(roles))} here, the fact is {r['stat']}"
    return None


class Verdict3(BaseModel):
    id: str
    subject: str
    stat: str
    value: str


class VerOut3(BaseModel):
    answers: list[Verdict3]


VER3_SYS = """You check number facts against a paper, in three separate parts. Each item has a passage from the
paper (one sentence with the sentence before it as context, or one table row as "column: cell" pairs, sometimes with
one "marked" cell) and one number fact with three fields: "about" (the study, trial, case or pooled set the number
belongs to, with its method, drug or material), "quantity" (what the number measures, e.g. estimate, 95% CI lower
bound, patients) and "value" (the number and unit).
Answer each part with "yes" or "no", looking only at the passage (if there is a "marked" cell: at that cell, with
its row and column header):
  subject: does the number in the passage belong to exactly the thing in "about"? Another study, another trial,
           another drug or a pooled value instead of a single study (or the reverse) is "no".
  stat:    does it measure exactly the "quantity"? An estimate instead of a CI bound, a lower instead of an upper
           bound, or another count is "no".
  value:   is this value written there? Formats count as equal: 9.0 = 9, .5 = 0.5, 3,050 = 3050.
Output: {"answers": [{"id": "<item id>", "subject": "yes"|"no", "stat": "yes"|"no", "value": "yes"|"no"}]}.

Examples:
passage "Kor 2021 enrolled 412 patients."; about "Kor 2021"; quantity "patients"; value "412 n"
-> subject yes, stat yes, value yes
passage "The pooled hazard ratio for drug B is 1.02 (0.85 to 1.22)."; about "pooled hazard ratio for drug A
(3 trials)"; quantity "95% CI upper bound"; value "1.02 HR" -> subject no (drug B), stat no (1.02 is the estimate),
value yes
passage "Trial T-1 found HR 0.63 (0.46 to 0.86) for drug A."; about "pooled hazard ratio for drug A (2 trials)";
quantity "estimate"; value "0.70 HR" -> subject no (one trial, not the pooled set), stat yes, value no"""


def fact_fields(r: dict[str, Any], value: str) -> tuple[str, str, str]:
    """(about, quantity, value) from the row label "<about>: <quantity>"."""
    about, sep, qty = r["label"].rpartition(": ")
    if not sep:
        about, qty = r["label"], ""
    stat = {"ci_low": "95% CI lower bound", "ci_high": "95% CI upper bound"}.get(r.get("stat"), r.get("stat"))
    return about, (qty or stat or str(r.get("attr") or "").replace("_", " ")), f"{value} {r.get('unit') or ''}".strip()


def ids3_check(ids: list[str]):
    def check(out: dict[str, Any]) -> str | None:
        if sorted(a["id"] for a in out["answers"]) != sorted(ids):
            return f"answer each id exactly once: {ids}"
        if any(a[k] not in ("yes", "no") for a in out["answers"] for k in ("subject", "stat", "value")):
            return 'subject, stat and value must each be "yes" or "no"'
        return None
    return check


def ids_check(ids: list[str]):
    def check(out: dict[str, Any]) -> str | None:
        got = [a["id"] for a in out["answers"]]
        if sorted(got) != sorted(ids):
            return f"answer each id exactly once: {ids}"
        if any(a["verdict"] not in ("yes", "no") for a in out["answers"]):
            return 'verdict must be "yes" or "no"'
        return None
    return check


class Verifier:
    """Yes/no calls in batches of BATCH; cache_only: answers from the cache only (None when not cached)."""

    def __init__(self, caller, cache_only: bool = False):
        self.call, self.cache_only = caller, cache_only
        self.not_cached = 0
        self.calls = 0
        self._lock = threading.Lock()

    def ask(self, items: list[tuple], step: str, system: str = "") -> list[bool | None]:
        """items: (context, passage, fact) or, for the code audit, (context, passage, fact, marked cell or None)."""
        system = system or VER_SYS
        out: list[bool | None] = []
        for i in range(0, len(items), BATCH):
            chunk = items[i:i + BATCH]
            ids = [f"V{j + 1}" for j in range(len(chunk))]
            user = json.dumps([{"id": k, "context": it[0], "passage": it[1], "fact": it[2]}
                               | ({"marked": it[3]} if len(it) > 3 and it[3] else {}) for k, it in zip(ids, chunk)],
                              indent=1, ensure_ascii=False)
            got = self._one(step, user, ids_check(ids), system)
            by = {a["id"]: a["verdict"] == "yes" for a in got["answers"]} if got else {}
            out += [by.get(k) for k in ids]
        return out

    def ask3(self, items: list[tuple], step: str) -> list[dict[str, bool] | None]:
        """items: (context, passage, about, quantity, value, marked or None) -> {subject, stat, value} per item."""
        out: list[dict[str, bool] | None] = []
        for i in range(0, len(items), BATCH):
            chunk = items[i:i + BATCH]
            ids = [f"V{j + 1}" for j in range(len(chunk))]
            user = json.dumps([{"id": k, "context": it[0], "passage": it[1], "about": it[2], "quantity": it[3],
                                "value": it[4]} | ({"marked": it[5]} if it[5] else {}) for k, it in zip(ids, chunk)],
                              indent=1, ensure_ascii=False)
            got = self._one(step, user, ids3_check(ids), VER3_SYS, VerOut3)
            by = {a["id"]: {k: a[k] == "yes" for k in ("subject", "stat", "value")} for a in got["answers"]} \
                if got else {}
            out += [by.get(k) for k in ids]
        return out

    def _one(self, step: str, user: str, check, system: str, out=VerOut) -> dict[str, Any] | None:
        with self._lock:
            self.calls += 1
        if not self.cache_only:
            return self.call("verifier", step, system, user, out, check)
        rm = self.call.s.role("verifier")
        path = self.call.cache / f"{self.call.key(rm, system, user, json_schema(out))}.json"
        got = self.call.cached(path, out, check)
        if got is None:
            with self._lock:
                self.not_cached += 1
        return got


# ------------------------------------------------------------------ one paper
def entities_of(r: dict[str, Any]) -> list[str]:
    return [x for x in [r.get("subject"), r.get("object"), *(r.get("studies") or [])] if x]


def fact_text(r: dict[str, Any], value: str) -> str:
    return f"{r['label']} = {value} {r.get('unit') or ''}".strip()


# ------------------------------------------------------------------ code rules v2 (26 Nov 2026)
LABEL_BEFORE = re.compile(r"(?:\b(?:Tables?|Figures?|Fig\.|[Ss]ections?|§|[Pp]hase|[Rr]ef\.?|Appendix|Supplement(?:ary)?|"
                          r"[Cc]hapter|[Ss]tep|[Ee]q\.?|[Ee]quation)\s*)$")
MEASURE_UNITS = {"%", "hr", "°c", "µm", "mm", "h", "gpu-h", "eur thousand", "days", "weeks", "months", "years"}
STOP = {"the", "of", "in", "for", "with", "and", "to", "a", "an", "by", "per", "study", "trial", "method", "number",
        "value", "reported", "total", "mean", "relative", "reference", "tested", "property"}
WORDNUM = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
POOLED = re.compile(r"\bpool(?:ed|ing)?\b|\bmeta-?analy|\binverse-variance\b|\bcombined (?:estimate|effect)", re.I)
N_BEFORE = r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\w+\s+)?"


def n_studies(kind: str | None) -> re.Pattern[str]:
    """"3 trials", "two included studies": a number before the plural of the pooled entities' kind (from the world,
    e.g. study, trial) or the generic "studies"."""
    plural = {(kind[:-1] + "ies") if kind.endswith("y") else kind + "s", "studies"} if kind else {"studies"}
    return re.compile(N_BEFORE + "(?:" + "|".join(sorted(map(re.escape, plural))) + r")\b", re.I)


def is_year_fact(r: dict[str, Any]) -> bool:
    """An attr whose value is a calendar year (attr name with "year", unit not a duration)."""
    return r["what"] == "attr" and "year" in str(r.get("attr", "")).lower() and \
        str(r.get("unit") or "").lower() not in ("years", "year")


def is_count_fact(r: dict[str, Any]) -> bool:
    """A whole-number count (patients, judges, models, ...): an integer value with a unit that is no measurement."""
    v = canon(r.get("value"))
    return v is not None and "." not in v and str(r.get("unit") or "").lower() not in MEASURE_UNITS


def value_hits(text: str, v: str, r: dict[str, Any]) -> list[tuple[int, int]]:
    """Where value v of fact r is written in text, without numbers that are labels (Table 2, section 9, Phase 3),
    years next to a name or in brackets (unless r is a year fact), and, for a count, numbers with decimals
    ("1.00" is no count; "1,200" is)."""
    out = []
    for a, b, x in numbers_in(text):
        if x != v:
            continue
        raw = text[a:b]
        if LABEL_BEFORE.search(text[:a]):
            continue
        if not is_year_fact(r) and re.fullmatch(r"(19|20)\d\d", raw) and (
                re.search(r"(?:[A-Z][\w.-]*\s|\(|\bin\s|,\s)$", text[:a]) or text[b:b + 1] == ")"):
            continue
        if is_count_fact(r) and "." in raw:
            continue
        out.append((a, b))
    return out


def words(s: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-zä-ü²]+", s.lower()) if w not in STOP and len(w) > 2}


def quantity_fits(r: dict[str, Any], axis: str) -> bool:
    """The other axis of a table cell (column header or row label) fits the fact's quantity: a content word of the
    label's quantity or of the attr name, or the unit (not "%" or "n", which are in most headers)."""
    _, qty, _ = fact_fields(r, r.get("value") or "")
    want = words(qty) | words(str(r.get("attr", "")).replace("_", " "))
    if want & words(axis):
        return True
    u = str(r.get("unit") or "")
    return u not in ("%",) and unit_in(u, axis)


def cell_check(r: dict[str, Any], place: Place, need: set[str], table: AliasTable) -> str | None:
    """A table cell in either orientation: one axis (the row, or the column header) names the fact's subject (an
    effect: also its object), the other axis fits the quantity. None = fits, else the reason (logged)."""
    row_names = {m.eid for m in find_mentions(" | ".join(place.row), table)}
    head_names = {m.eid for m in find_mentions(place.header, table)}
    label = place.row[0] if place.row else ""
    if need <= row_names and quantity_fits(r, place.header):
        return None
    if need <= (head_names | row_names) and need & head_names and quantity_fits(r, label + " " + place.header):
        return None
    if not need <= (row_names | head_names):
        return "cell: subject not on an axis"
    return f"cell: quantity does not fit header {place.header[:60]!r} / row {label[:40]!r}"


def pooled_check(r: dict[str, Any], text: str, names: set[str], ekind: dict[str, str]) -> str | None:
    """A pooled value's place says pooled AND names 2+ studies or states a number of studies; a stated number must
    equal the fact's number of studies. None = fits."""
    if not POOLED.search(text):
        return "pool: place does not say pooled"
    studies = r.get("studies") or []
    kind = ekind.get(studies[0]) if studies else None
    n_named = sum(ekind.get(e) == kind for e in names) if kind else 0
    counts = {int(m.group(1)) if m.group(1).isdigit() else WORDNUM[m.group(1).lower()]
              for m in n_studies(kind).finditer(text)}
    if counts and len(studies) not in counts:
        return f"pool: states {sorted(counts)} studies, the fact pools {len(studies)}"
    if not counts and n_named < 2:
        return "pool: names fewer than 2 studies and states no number of studies"
    return None


def span_paper(md: str, rows: list[dict[str, Any]], table: AliasTable, rules: SplitRules,
               verifier: Verifier, pid: str, ekind: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """One record per number row: spans (normalized offsets), found_by per span, and notes."""
    P = places(md, rules)
    ekind = ekind or {}
    ments = [{m.eid for m in find_mentions(p.names_text, table)} for p in P]
    seen_names = [{m.eid for m in find_mentions(p.context + " " + p.passage, table)} for p in P]
    in_passage_names = [{m.eid for m in find_mentions(p.passage, table)} for p in P]   # the gate: passage only
    coll = colliding(rows)
    res, ask, back = [], [], []          # ask: ask3 items; back: (record index, place index, control?)
    for r in rows:
        rec = {"id": r["id"], "kind": r["what"], "spans": [], "found_by": [], "notes": [], "cell_log": []}
        res.append(rec)
        v = canon(r.get("value"))
        if v is None:
            rec["notes"].append("no value")
            continue
        hits = [value_hits(p.text, v, r) for p in P]
        occ = [i for i, h in enumerate(hits) for _ in h]
        where = sorted(set(occ))
        if not where:
            rec["notes"].append("value_not_found")
            continue
        ents = set(entities_of(r))
        need = {r["subject"]} | ({r["object"]} if r["what"] == "effect" and r.get("object") else set())

        def fits(i: int, strict: bool) -> bool:
            """Rules 1-3: cells in either orientation, pooled places, names (once: any entity; several: subject)."""
            if P[i].row:
                if sum(bool(value_hits(c, v, r)) for c in P[i].row) != 1:
                    rec["cell_log"].append(f"place {i}: value in 2+ cells of the row")
                    return False
                if why := cell_check(r, P[i], need, table):
                    rec["cell_log"].append(f"place {i}: {why}")
                    return False
            elif not (need <= ments[i] if strict else ments[i] & ents):
                return False
            if r["what"] == "pool" and (why := pooled_check(r, P[i].passage, in_passage_names[i], ekind)):
                rec["cell_log"].append(f"place {i}: {why}")
                return False
            if r["what"] in ("effect", "pool") and r.get("stat") in CI:       # estimate vs CI bound, as in the gate
                roles = value_roles(P[i].text, v, P[i].header)
                if roles and r["stat"] not in roles:
                    rec["cell_log"].append(f"place {i}: stat: value is {'/'.join(sorted(roles))} here")
                    return False
            return True
        keep, how = [], ""
        if len(occ) == 1 and fits(where[0], strict=False):
            keep, how = [where[0]], "once+name"
        elif len(occ) == 1 and not P[where[0]].row and r["what"] != "pool" and \
                unit_in(r.get("unit", ""), P[where[0]].unit_text):
            keep, how = [where[0]], "once+unit"
        elif len(occ) > 1:
            keep, how = [i for i in where if fits(i, strict=True)], "several+name"
        # rule 4: a fact whose value another fact of the paper shares, or with 2+ places left, goes to the verifier
        if keep and (r["id"] in coll or len(keep) > 1):
            rec["notes"].append("rule 4: " + ("same value as " + ", ".join(coll[r["id"]][:3]) if r["id"] in coll
                                              else f"{len(keep)} places") + " -> verifier")
            where, keep = keep, []
        if keep:
            rec["spans"], rec["found_by"] = keep, [how] * len(keep)
        else:
            if len(where) > MAX_PLACES:
                rec["notes"].append(f"{len(where)} places, first {MAX_PLACES} sent to the verifier")
            for i in where[:MAX_PLACES]:
                if why := gate(r, P[i], in_passage_names[i]):
                    rec["notes"].append(f"place {i}: gate ({why})")
                    continue
                in_passage = {x for _, _, x in numbers_in(P[i].passage)}
                ask.append((P[i].context, P[i].passage, *fact_fields(r, r["value"]), P[i].marked))
                back.append((len(res) - 1, i, False))
                ask.append((P[i].context, P[i].passage, *fact_fields(r, wrong_value(v, in_passage)), P[i].marked))
                back.append((len(res) - 1, i, True))
    got = verifier.ask3(ask, f"numverify3:{pid}") if ask else []
    for (k, i, control), g in zip(back, got):
        rec = res[k]
        ok = None if g is None else all(g.values())
        if control:
            rec.setdefault("controls", []).append(ok)
        elif g is None:
            rec["notes"].append(f"place {i}: no verifier answer")
        elif ok:
            rec["spans"].append(i)
            rec["found_by"].append("verifier")
        else:
            rec["notes"].append(f"place {i}: verifier no on {[f for f, y in g.items() if not y]}")
    for rec in res:
        rec["places"] = rec.pop("spans")
        rec["spans"] = [[P[i].start, P[i].end] for i in rec["places"]]
        rec["groups"] = [list(P[i].group) for i in rec["places"]]
    # hard controls: names anywhere in what the verifier sees (a sideways table names subjects in its headers)
    by_row = {r["id"]: r for r in rows}
    audit = [{"fact": rec["id"], "found_by": how, "context": P[i].context, "passage": P[i].passage,
              "marked": P[i].marked, "text": fact_text(by_row[rec["id"]], by_row[rec["id"]]["value"]),
              "span_text": P[i].marked or P[i].text, "fields": fact_fields(by_row[rec["id"]], by_row[rec["id"]]["value"]),
              "gate": gate(by_row[rec["id"]], P[i], in_passage_names[i])}
             for rec in res for i, how in zip(rec["places"], rec["found_by"]) if how != "verifier"]
    return res, {"v2_seen": hard_candidates(res, rows, P, seen_names, in_passage_names),
                 "v1_row": hard_candidates(res, rows, P, ments, in_passage_names),
                 "audit": audit, "P": P, "seen": in_passage_names}


def hard_candidates(res: list[dict[str, Any]], rows: list[dict[str, Any]], P: list[Place],
                    ments: list[set[str]], seen_names: list[set[str]]) -> list[dict[str, Any]]:
    """Hard controls: fact F with a place of another fact G that has the SAME value but another subject, where the
    place (passage or context, as the verifier sees it) does not name F's subject (e.g. another study's row). The
    right answer is "no"."""
    by = {r["id"]: r for r in rows}
    out, seen = [], set()
    for F in res:
        rf = by[F["id"]]
        v = canon(rf.get("value"))
        for G in res:
            rg = by[G["id"]]
            if G is F or v is None or canon(rg.get("value")) != v or rg.get("subject") == rf.get("subject"):
                continue
            for i in G["places"]:
                if i in F["places"] or rf.get("subject") in ments[i] or (F["id"], i) in seen:
                    continue
                seen.add((F["id"], i))
                out.append({"fact": F["id"], "other": G["id"], "context": P[i].context, "passage": P[i].passage,
                            "text": fact_text(rf, rf["value"]), "fields": fact_fields(rf, rf["value"]),
                            "marked": P[i].marked, "gate": gate(rf, P[i], seen_names[i])})
    return out


def frozen_cell(h: dict[str, Any], rule: str, P: list[Place], by: dict[str, dict[str, Any]], table: AliasTable,
                used: dict[tuple, int]) -> tuple[int | None, str | None]:
    """The place of a frozen hard-control item. A sentence is found by (context, passage). Every cell of a table row
    has the same pair, so in a row pick the cell that holds the value and is the other fact's (G's): the cell whose
    header names G's subject, else a value cell whose header does not name F's subject. A repeated item (same row,
    another cell of G) takes the next such cell. Returns (place index, how the cell was picked)."""
    key = (h["context"], h["passage"])
    idx = [i for i, q in enumerate(P) if (q.context, q.passage) == key]
    if not idx:
        return None, None
    if not P[idx[0]].row:
        return idx[-1], None
    r, g = by[h["fact"]], by.get(h.get("other", ""), {})
    v = canon(r.get("value")) or ""
    hdr = lambda j: {m.eid for m in find_mentions(P[j].header, table)}       # noqa: E731
    vc = [j for j in idx if value_hits(P[j].text, v, r)]
    pick = [j for j in vc if g.get("subject") in hdr(j)] or [j for j in vc if r.get("subject") not in hdr(j)]
    if not pick:
        return idx[-1], "no value cell: last cell kept"
    n = used.get((rule, h["fact"], h.get("other")) + key, 0)
    used[(rule, h["fact"], h.get("other")) + key] = n + 1
    i = pick[min(n, len(pick) - 1)]
    return i, "value cell of G" if g.get("subject") in hdr(i) else "value cell"


# ------------------------------------------------------------------ one domain
def writer_of(p: dict[str, Any]) -> str:
    run = str((p.get("source") or {}).get("run", ""))
    return next((w for key, w in WRITERS if key in run), "unknown")


def ci_mismatches(rows: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Effects or pools (one paper) whose estimate and CI bounds do not share a sentence or table row (`groups`), or
    where only some of the three have a span."""
    groups: dict[tuple, dict[str, str]] = {}
    for r in rows:
        if r["what"] in ("effect", "pool") and r.get("stat") in CI:
            key = (r["paper"], r["what"], r.get("subject"), r.get("relation"), r.get("object"),
                   tuple(r.get("studies") or []))
            groups.setdefault(key, {})[r["stat"]] = r["id"]
    out = []
    for key, g in groups.items():
        sets = {st: {tuple(sp) for sp in by_id[i]["groups"]} for st, i in g.items()}
        have = {st: s for st, s in sets.items() if s}
        if len(have) < 2:
            continue
        if not set.intersection(*have.values()) or len(have) < len(g):
            out.append({"paper": key[0], "ids": g, "with_span": sorted(have),
                        "shared": bool(set.intersection(*have.values()))})
    return out


def run(seed_dir: Path, ds_dir: Path, caller, cache_only: bool = False) -> dict[str, Any]:
    man = json.loads((seed_dir / "paperfactory.json").read_text(encoding="utf-8"))
    ents = json.loads((seed_dir / "entities.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (seed_dir / "facts.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r["what"] in ("attr", "effect", "pool")]
    f = packs.field(man["field"])
    w = {"entities": ents}
    rules, base = split_rules(f, w), alias_table(w, f)
    verifier = Verifier(caller, cache_only)
    by_id: dict[str, dict[str, Any]] = {}
    vm = caller.s.model("verifier")
    hard_all: dict[str, list[dict[str, Any]]] = {k: [] for k in HARD_RULES}
    collisions = {"values": 0, "facts": 0, "pairs": 0, "different_subject_pairs": 0}
    audit_all: list[dict[str, Any]] = []
    meta_all = {r["id"]: r for r in rows}
    fz = seed_dir / "numspans_hard_frozen.json"
    frozen = {k: v for k, v in json.loads(fz.read_text(encoding="utf-8")).items() if k in HARD_RULES} \
        if fz.is_file() else {}
    af = seed_dir / "numspans_audit_frozen.json"
    audit_frozen = json.loads(af.read_text(encoding="utf-8"))["items"] if af.is_file() else []
    ekinds = {k: e["kind"] for k, e in ents.items()}

    def one(p: dict[str, Any]) -> tuple:
        md = (ds_dir / p["id"] / "paper.md").read_text(encoding="utf-8")
        norm = normalize_paper(md)
        table = alias_table(w, f, defined_terms(norm.text, base, f))
        mine = [r for r in rows if r["paper"] == p["id"]]
        recs, hard = span_paper(md, mine, table, rules, verifier, p["id"], ekinds)
        return p, md, norm, mine, recs, hard, table
    for p, md, norm, mine, recs, hard, table in pmap(one, man["papers"], PARALLEL):     # merged in paper order
        for k in HARD_RULES:
            hard_all[k] += [h | {"paper": p["id"], "same_model": writer_of(p) == vm} for h in hard[k]]
        for a, b in value_collisions(mine).items():
            collisions[a] += b
        coll = colliding(mine)
        audit_all += [h | {"paper": p["id"], "collision": coll.get(h["fact"])} for h in hard["audit"]]
        at = {(q.context, q.passage): i for i, q in enumerate(hard["P"])}
        used: dict[tuple, int] = {}
        for k in frozen:                 # the frozen hard-control items of this paper: gate and fields from today's code
            for h in frozen[k]:
                if h["paper"] == p["id"]:
                    i, h["cell_pick"] = frozen_cell(h, k, hard["P"], meta_all, table, used)
                    r = meta_all[h["fact"]]
                    h["fields"], h["gate"] = fact_fields(r, r["value"]), "place not found" if i is None else \
                        gate(r, hard["P"][i], hard["seen"][i])
                    h["marked"] = None if i is None else hard["P"][i].marked
        for h in audit_frozen:           # the frozen audit items of this paper: is the span still placed, and how?
            if h["paper"] == p["id"]:
                h["place_now"] = at.get((h["context"], h["passage"]))
        meta = {r["id"]: r for r in mine}
        p["numbers"] = []
        for rec in recs:
            r = meta[rec["id"]]
            raw = [[norm.back[a], norm.back[b - 1] + 1] for a, b in rec["spans"]]
            item = {"id": r["id"], "kind": r["what"], "label": r["label"], "value": r["value"], "unit": r["unit"],
                    "stat": r.get("stat"), "spans": raw, "found_by": rec["found_by"]}
            p["numbers"].append(item)
            by_id[r["id"]] = item | {"notes": rec["notes"], "controls": rec.get("controls", []), "paper": p["id"],
                                     "groups": rec["groups"], "same_model": writer_of(p) == vm,
                                     "places": rec["places"], "cell_log": rec["cell_log"]}
        p["writer"] = writer_of(p)
    rep = report(rows, by_id, verifier)
    rep["hard_controls"] = hard_controls(hard_all["v2_seen"], verifier, frozen.get("v2_seen")) | {
        "rule": HARD_RULES["v2_seen"]}
    # the first run's rule (its 'before' answers from the frozen file or the cache only), kept to show why it changed
    rep["hard_controls_v1"] = hard_controls(hard_all["v1_row"], verifier, frozen.get("v1_row"), before_cache_only=True) \
        | {"rule": HARD_RULES["v1_row"], "why_replaced": HARD_WHY}
    rep["error_rates"] = {name: rate(c["yes"], c["items"] - c["no_answer"])
                          for name, c in (("wrong_number_control", rep["controls"]),
                                          ("hard_control", rep["hard_controls"]),
                                          ("hard_control_v1", rep["hard_controls_v1"]))}
    for name, key in (("hard_control_after_fix", "hard_controls"), ("hard_control_v1_after_fix", "hard_controls_v1")):
        a = rep[key]["after_fix"]["gate_and_verifier3"]
        rep["error_rates"][name] = rate(a["yes"], a["items"] - a["no_answer"])
    rep["code_audit"], audit_pick = code_audit(audit_all, verifier)
    if audit_frozen:
        rep["audit_before_after"] = audit_before_after(audit_frozen, by_id)
    if not af.is_file():                 # the audit sample as drawn today, for before/after code-rule changes
        keep = ("paper", "fact", "found_by", "collision", "context", "passage", "marked", "span_text", "answer",
                "answer3", "gate")
        af.write_text(json.dumps({"source": "numspans.code_audit (code rules of commit 426664f)",
                                  "items": [{k: h.get(k) for k in keep} for h in audit_pick]}, indent=1,
                                 ensure_ascii=False), encoding="utf-8")
    rep["verifier_error_quote"] = VERIFIER_ERROR_QUOTE
    rep["value_collisions"] = collisions | {"note": "pairs of different number facts in one paper with the same "
                                                    "value (after canon); a value matched by code can be the wrong one"}
    man["number_spans"] = {k: rep[k] for k in ("verifier_model", "independence")}
    (seed_dir / "paperfactory.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    (seed_dir / "numspans.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    return rep


def rate(k: int, n: int) -> dict[str, Any]:
    """k errors of n: share with an exact (Clopper-Pearson) 95% interval."""
    return {"errors": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95_exact": clopper_pearson(k, n)}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float] | None:
    if n == 0:
        return None

    def cdf(x: int, p: float) -> float:          # P(X <= x), X ~ Binomial(n, p)
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1))

    def solve(f, target: float) -> float:        # f falls as p rises; bisection on [0, 1]
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
        return (lo + hi) / 2
    low = 0.0 if k == 0 else solve(lambda p: cdf(k - 1, p), 1 - alpha / 2)
    high = 1.0 if k == n else solve(lambda p: cdf(k, p), alpha / 2)
    return [round(low, 4), round(high, 4)]


def colliding(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Fact id -> the other facts of the same paper with the same canonical value (only facts in such a pair)."""
    by: dict[str, list[str]] = {}
    for r in rows:
        if (v := canon(r.get("value"))) is not None:
            by.setdefault(v, []).append(r["id"])
    return {i: [j for j in g if j != i] for g in by.values() if len(g) > 1 for i in g}


def code_audit(items: list[dict[str, Any]], verifier: Verifier) -> dict[str, Any]:
    """The verifier as a yes/no check on code-made spans: every span of a fact in a same-value pair ("collision"),
    plus AUDIT_RANDOM other several+name spans (fixed seed). A "no" counts as a code-rule error."""
    coll = [h for h in items if h["collision"]]
    rest = [h for h in items if not h["collision"] and h["found_by"] == "several+name"]
    pick = coll + random.Random(AUDIT_SEED).sample(rest, min(AUDIT_RANDOM, len(rest)))
    pick.sort(key=lambda h: (h["paper"], h["fact"], h["passage"]))
    def judge(pid: str) -> None:
        mine = [h for h in pick if h["paper"] == pid]
        got = verifier.ask([(h["context"], h["passage"], h["text"], h["marked"]) for h in mine], f"numaudit:{pid}",
                           AUDIT_SYS)
        for h, g in zip(mine, got):
            h["answer"] = g
        got3 = verifier.ask3([(h["context"], h["passage"], *h["fields"], h["marked"]) for h in mine],
                             f"numaudit3:{pid}")
        for h, g in zip(mine, got3):
            h["answer3"] = g
            h["answer_fixed"] = False if h["gate"] else (None if g is None else all(g.values()))
    pmap(judge, list(dict.fromkeys(h["paper"] for h in pick)), PARALLEL)

    def rate_of(xs: list[dict[str, Any]], key: str = "answer") -> dict[str, Any]:
        done = [h for h in xs if h[key] is not None]
        return rate(sum(h[key] is False for h in done), len(done)) | {"no_answer": len(xs) - len(done)}

    def fixed(xs: list[dict[str, Any]]) -> dict[str, Any]:
        return rate_of(xs, "answer_fixed") | {
            "by_gate": sum(bool(h["gate"]) for h in xs),
            "verifier3_no": {f: sum(bool(h["answer3"]) and not h["answer3"][f] for h in xs if not h["gate"])
                             for f in ("subject", "stat", "value")}}
    methods = sorted({h["found_by"] for h in pick})
    rnd = [h for h in pick if not h["collision"]]
    return {"note": "judge = the verifier (its own error on hard controls: see verifier_error_quote); a table cell is "
                    "judged as the marked cell, not the whole row. after_fix: judge = code gate (subject named, "
                    "stat role fits) + three-field verifier (subject, stat, value all yes)",
            "after_fix": {"collision": fixed(coll), "several_name_random": fixed(rnd),
                          "gate_reasons": dict(sorted(Counter(h["gate"].split(":")[0] for h in pick if h["gate"])
                                                      .items())),
                          "rejected": [{k: h[k] for k in ("paper", "fact", "found_by", "collision", "text",
                                                          "span_text", "gate", "answer3")}
                                       for h in pick if h["answer_fixed"] is False]},
            "collision": rate_of(coll) | {"by_method": {m: rate_of([h for h in coll if h["found_by"] == m])
                                                        for m in methods}},
            "several_name_random": rate_of([h for h in pick if not h["collision"]]) | {
                "of": len(rest)},
            "collision_spans": [{k: h[k] for k in ("paper", "fact", "found_by", "span_text", "answer")}
                                | {"collides_with": h["collision"]} for h in coll],
            "rejected": [{k: h[k] for k in ("paper", "fact", "found_by", "collision", "text", "span_text", "passage")}
                         for h in pick if h["answer"] is False]}, pick


def audit_before_after(items: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The frozen audit sample (code rules of commit 426664f, judged by the fixed judge: gate + three-field verifier)
    against today's spans: a wrong span is fixed when it is no longer placed; a right span is lost when it is not."""
    def judged_wrong(h: dict[str, Any]) -> bool | None:
        if h.get("gate"):
            return True
        a3 = h.get("answer3")
        return None if a3 is None else not all(a3.values())
    out: dict[str, Any] = {}
    for stratum in ("collision", "several_name_random"):
        xs = [h for h in items if bool(h["collision"]) == (stratum == "collision")]
        rows = []
        for h in xs:
            it = by_id.get(h["fact"], {})
            idx = it.get("places", [])
            now = h.get("place_now")
            how = it["found_by"][idx.index(now)] if now is not None and now in idx else None
            rows.append((judged_wrong(h), how))
        wrong = [hw for w, hw in rows if w]
        right = [hw for w, hw in rows if w is False]
        out[stratum] = {"items": len(xs), "judged_wrong": len(wrong), "judged_right": len(right),
                        "before": rate(len(wrong), len(wrong) + len(right)),
                        "after": rate(sum(hw is not None for hw in wrong), len(wrong) + len(right)),
                        "wrong_still_placed_by": dict(Counter(hw for hw in wrong if hw)),
                        "right_kept": sum(hw is not None for hw in right), "right_lost": sum(hw is None for hw in right),
                        "right_kept_by": dict(Counter(hw for hw in right if hw))}
    out["note"] = ("before: share of the frozen spans the fixed judge calls wrong; after: share of the same spans that "
                   "are wrong and still placed today (by code or by the verifier). right_lost: right spans no longer "
                   "placed (the price in coverage)")
    return out


def value_collisions(rows: list[dict[str, Any]]) -> dict[str, int]:
    """One paper's number facts grouped by canonical value: groups with 2+ facts, facts in them, pairs."""
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if (v := canon(r.get("value"))) is not None:
            by.setdefault(v, []).append(r)
    out = {"values": 0, "facts": 0, "pairs": 0, "different_subject_pairs": 0}
    for g in by.values():
        if len(g) > 1:
            out["values"] += 1
            out["facts"] += len(g)
            out["pairs"] += len(g) * (len(g) - 1) // 2
            out["different_subject_pairs"] += sum(a.get("subject") != b.get("subject")
                                                  for j, a in enumerate(g) for b in g[j + 1:])
    return out


def hard_controls(cands: list[dict[str, Any]], verifier: Verifier, frozen: list[dict[str, Any]] | None = None,
                  before_cache_only: bool = False) -> dict[str, Any]:
    """Hard controls; every right answer is "no". frozen: the items as first drawn (numspans_hard_frozen.json, with
    their first answer) so before and after are judged on the same items; else up to HARD_MAX drawn (fixed seed) and
    asked with the first verifier (v1 prompt). after_fix: the code gate, then the three-field verifier (v2)."""
    if frozen is not None:
        pick = frozen
    else:
        pick = sorted(random.Random(HARD_SEED).sample(cands, min(HARD_MAX, len(cands))),
                      key=lambda h: (h["paper"], h["fact"], h["passage"]))
        old = Verifier(verifier.call, cache_only=True) if before_cache_only else verifier
        def first(pid: str) -> None:
            mine = [h for h in pick if h["paper"] == pid]
            for h, g in zip(mine, old.ask([(h["context"], h["passage"], h["text"]) for h in mine], f"numhard:{pid}")):
                h["answer"] = g
        pmap(first, list(dict.fromkeys(h["paper"] for h in pick)), PARALLEL)

    def fixed_judge(pid: str) -> None:
        mine = [h for h in pick if h["paper"] == pid]
        got = verifier.ask3([(h["context"], h["passage"], *h["fields"], h.get("marked")) for h in mine],
                            f"numhard3:{pid}")
        for h, g in zip(mine, got):
            h["answer3"] = None if g is None else all(g.values())
            h["fields3"] = g
            h["answer_fixed"] = False if h.get("gate") else h["answer3"]
    pmap(fixed_judge, list(dict.fromkeys(h["paper"] for h in pick)), PARALLEL)

    def count(xs: list[dict[str, Any]], key: str = "answer") -> dict[str, int]:
        return {"items": len(xs), "no": sum(h[key] is False for h in xs),
                "yes": sum(h[key] is True for h in xs), "no_answer": sum(h[key] is None for h in xs)}
    return {"candidates": len(cands), "frozen": frozen is not None, **count(pick),
            "by_writer": {"same_model": count([h for h in pick if h["same_model"]]),
                          "other_model": count([h for h in pick if not h["same_model"]])},
            "note": "right value, wrong thing (another subject's place with the same value); 'yes' is a verifier "
                    "error. Tests the verifier only: code-made spans never see it",
            "accepted": [{k: h[k] for k in ("paper", "fact", "other", "text", "passage")} for h in pick
                         if h["answer"]],
            "after_fix": {"gate_alone": {"items": len(pick), "rejected": sum(bool(h.get("gate")) for h in pick),
                                         "passed": sum(not h.get("gate") for h in pick)},
                          "verifier3_alone": count(pick, "answer3"),
                          "gate_and_verifier3": count(pick, "answer_fixed"),
                          "note": "the items name no subject by construction (v2) or only in a header (v1), so the "
                                  "subject gate rejects most of them; verifier3_alone shows the new verifier "
                                  "without the gate",
                          "accepted": [{k: h.get(k) for k in ("paper", "fact", "text", "passage", "gate", "fields3")}
                                       for h in pick if h["answer_fixed"]]}}


def report(rows: list[dict[str, Any]], by_id: dict[str, dict[str, Any]], verifier: Verifier) -> dict[str, Any]:
    cov: dict[str, dict[str, int]] = {}
    for r in rows:
        it = by_id.get(r["id"])
        c = cov.setdefault(r["what"], {"facts": 0, "with_span": 0, "once+name": 0, "once+unit": 0,
                                        "several+name": 0, "verifier": 0})
        c["facts"] += 1
        if it and it["spans"]:
            c["with_span"] += 1
            c[it["found_by"][0]] += 1
    for c in cov.values():
        c["share"] = round(c["with_span"] / max(1, c["facts"]), 3)
    def controls(same: bool | None) -> dict[str, int]:
        c = [g for it in by_id.values() if same is None or it["same_model"] == same for g in it["controls"]]
        return {"items": len(c), "no": sum(g is False for g in c), "yes": sum(g is True for g in c),
                "no_answer": sum(g is None for g in c)}
    vm = verifier.call.s.model("verifier")
    papers = {it["paper"]: it["same_model"] for it in by_id.values()}
    ver = [it for it in by_id.values() if "verifier" in it["found_by"]]
    return {"verifier_model": vm, "cache_only": verifier.cache_only,
            "independence": {"same_model_as_a_writer": any(papers.values()),
                             "papers_with_numbers": len(papers), "papers_same_model": sum(papers.values()),
                             "verifier_spans_same_model": sum(it["same_model"] for it in ver),
                             "verifier_spans_other_model": sum(not it["same_model"] for it in ver),
                             "note": f"the verifier ({vm}) is the same model as the writer of papers_same_model "
                                     "papers; it may favour its own wording. See controls_by_writer: a 'yes' on "
                                     "a wrong value means it just agrees"},
            "verifier_calls": verifier.calls, "not_cached": verifier.not_cached,
            "coverage": cov,
            "controls": controls(None) | {"note": "wrong value made by code; 'yes' is a verifier error"},
            "controls_by_writer": {"same_model": controls(True), "other_model": controls(False)},
            "ci_mismatch": ci_mismatches(rows, by_id),
            "no_span": [{"id": i, "paper": it["paper"], "kind": it["kind"], "value": it["value"], "notes": it["notes"]}
                        for i, it in by_id.items() if not it["spans"]]}


def caller_for(domain: str, field_name: str, genre: str):
    from paperfactory.config import Settings
    from paperfactory.llm import Caller
    s = Settings(field=field_name, genre=genre or "argument",
                 models={"verifier": {"model": VERIFIER_MODEL, "max_parallel": PARALLEL}},
                 out=f"runs/paperfactory/numspans/{domain}")
    return Caller(s)
