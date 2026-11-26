"""Realism distribution check (data-quality metric 3): do synthetic papers read like real review papers?

Code only, no model calls. Per body paragraph it measures four properties, on text prepared the same way on both
sides:

    sent_len   mean words per sentence
    hedges     hedging words per 100 words (fixed list HEDGES)
    cites      citations per 100 words (CITATION patterns: [12], ^(4-5)^, (Smith et al., 2019), Smith (2019),
               "the Braliram 2023 study", "A v. B (Court, 2017)")
    numbers    numbers per 100 words (paperfactory.numspans.numbers_in, after names and citations are masked)

The fifth property, the share of each fact kind, comes from the blind reader (Part C); `chi2_cramers_v` takes two
count tables. `compare` runs the two-sample Kolmogorov-Smirnov test per property and the chi-square test, applies
the Holm correction over the 5 tests, and lists effect sizes (D, Cramer's V) first.

Preparation (same on both sides):
  1. Body paragraphs only: Markdown paragraphs (no headings, list items, table cells, captions), outside the
     sections in SKIP_SECTION (abstract, references, acknowledgements, ...), with >= MIN_WORDS words. At most
     MAX_PARAS paragraphs per paper, spread evenly over the body, so long papers do not dominate.
  2. Sentences: text.split_rules of the field pack with an empty world (no world names), so both sides split alike.
  3. "Invented world" sentences (`invented_world`) are dropped before measuring.
  4. Citations -> [CIT], then names -> [NAME] (synthetic: world names via text.alias_table / find_mentions;
     real: `generic_names`), then numbers -> [NUM]. Each placeholder counts as one word.

Run on the synthetic papers of a data folder (writes seed/<domain>/realism_synth.json and
invented_world_sentences.json):

    python -m paperfactory.realism synth --data ../../data --domains bench_law bench_llm bench_materials bench_medicine

Real papers (Part E): open-licence review papers downloaded to data/cache/realism/<field>/<id>.md with
manifest.jsonl (selection: data/seed/realism_selection.json, law licences: realism_law_licences.json). `real`
measures them the same way (generic name masking), compares per field with seed/bench_<field>/realism_synth.json
(plus realism_kinds.json when complete) and writes seed/bench_<field>/realism_real.json, realism_compare.json and
seed/realism_report.md. A field with fewer than MIN_REAL real papers is marked pending.

    python -m paperfactory.realism real --data ../../data --realism-dir ../../data/cache/realism

`kinds` (model calls): fact kinds of KIND_SAMPLE real and KIND_SAMPLE synthetic papers per field (synthetic sample
stratified by writer, readback.sample_papers), read section by section by readback.Reader with Haiku only (JUDGE);
cached under data/cache/realism/haiku/<domain>; no new section is started once all Haiku calls there cost --budget.
Every count carries "judge"; synthetic papers written by Haiku carry "same model as writer".

    python -m paperfactory.realism kinds --data ../../data --realism-dir ../../data/cache/realism \
        --fields medicine law materials llm --budget 14.5 [--max-sections 10] [--plan-only]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperfactory import packs
from paperfactory.numspans import numbers_in
from paperfactory.text import (_split_unit, alias_table, defined_terms, find_mentions, norm_text, normalize_paper,
                              split_rules)

PROPS = ("sent_len", "hedges", "cites", "numbers")
MIN_WORDS = 30
MAX_PARAS = 40
RESAMPLES, SEED = 2000, 0
METHOD = (f"body paragraphs only (>= {MIN_WORDS} words, <= {MAX_PARAS} per paper, evenly spaced); headings, lists, "
          "tables, captions and front/back matter removed; invented-world sentences dropped; citations, names and "
          "numbers masked as one-word placeholders. Mean intervals: percentile bootstrap over papers "
          f"({RESAMPLES} resamples, seed {SEED}, 95%).")

# ------------------------------------------------------------------ sections and paragraphs
SKIP_SECTION = re.compile(
    r"^\W*(?:\d+(?:\.\d+)*\.?\s+)?(?:abstract|summary|keywords?|references?|bibliography|works cited|literature cited|"
    r"acknowledge?ments?|funding|financial support|conflicts? of interest|competing interests?|declarations?|"
    r"disclosures?|author(?:s'?|'s)? contributions?|data availability|code availability|ethics|ethical approval|"
    r"supplementary|supporting information|appendix|appendices|abbreviations|notes|endnotes|footnotes|"
    r"highlights|graphical abstract|consent|citation|copyright|licen[cs]e)\b", re.I)
CAPTION = re.compile(r"^\W*(?:table|figure|fig\.|chart|box|panel)\s*[A-Z]?\d", re.I)


def _words(s: str) -> list[str]:
    return WORD.findall(s)


def body_paragraphs(md: str, max_paras: int = MAX_PARAS) -> list[tuple[str, str]]:
    """(section heading, paragraph text) of the paper's body paragraphs, in normalized text (text.normalize_paper).
    A heading at the top level (the title, "#") does not open a skipped section; any other heading sets the
    section, and paragraphs under a SKIP_SECTION heading are left out. At most max_paras, evenly spaced."""
    p = normalize_paper(md)
    out: list[tuple[str, str]] = []
    section, skip = "", False
    for u in p.units:
        t = p.text[u.start:u.end]
        if u.kind == "heading":
            section, skip = t, bool(SKIP_SECTION.match(t))
            continue
        if u.kind != "paragraph" or skip or CAPTION.match(t) or t.lstrip().startswith(("$$", "\\[", "<")):
            continue
        if len(_words(t)) >= MIN_WORDS:
            out.append((section, t))
    if len(out) > max_paras:
        idx = sorted({round(i * (len(out) - 1) / (max_paras - 1)) for i in range(max_paras)})
        out = [out[i] for i in idx]
    return out


# ------------------------------------------------------------------ invented-world sentences
_WORLD_NOUN = (r"(?:countr(?:y|ies)|jurisdictions?|(?:legal\s+)?systems?|settings?|worlds?|names?|trials?|"
               r"stud(?:y|ies)|drugs?|datasets?|benchmarks?|scenarios?|case\s+stud(?:y|ies)|labs?|laboratories|"
               r"entities|constructs?|contexts?|framework|exercise|analysis|materials?|alloys?|"
               r"models?|methods?|data)")
INVENTED_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("invented_noun", re.compile(r"\b(?:invented|fictional|fictitious|imaginary|made-up)(?!\s+(?:by|in|at|during)\b)"
                                 rf"\s+(?:\w+[\s-]+){{0,3}}?{_WORLD_NOUN}\b", re.I)),
    ("is_invented", re.compile(r"\b(?:is|are|were|was|been)\s+(?:all\s+|entirely\s+|wholly\s+)?"
                               r"(?:invented|fictional|fictitious|imaginary|made\s+up)\b(?!\s+(?:by|in\s+\d|at|during))", re.I)),
    ("hypothetical_world", re.compile(r"\bhypothetical\s+(?:\w+\s+)?(?:jurisdictions?|countr(?:y|ies)|legal\s+system|"
                                      r"framework|case\s+stud(?:y|ies)|trials|evaluation\s+studies|constructs?|"
                                      r"doctrinal\s+analysis|world)\b", re.I)),
    ("synthetic_world", re.compile(r"\b(?:this|a|the)\s+synthetic\s+(?:analysis|context|world|setting|dataset)\b|"
                                   r"\bin\s+a\s+synthetic\s+context\b", re.I)),
    ("no_real", re.compile(r"\bno\s+real\s+(?:legal\s+)?(?:system|country|jurisdiction|world|trial|stud)", re.I)),
    ("not_real", re.compile(r"\b(?:do|does|did)\s+not\s+(?:correspond|refer|relate)\s+to\s+any\s+real\b|"
                            r"\bnot\s+(?:derived\s+from|based\s+on|drawn\s+from)\s+real(?:-world)?\s+"
                            r"(?:clinical\s+)?(?:trials|data|cases|studies)\b|"
                            r"\b(?:carry|carries|extend|extends)\s+(?:over\s+)?to\s+(?:any\s+)?real\s+(?:legal\s+)?"
                            r"systems?\b|\bnon-standard\s+or\s+fictitious\b", re.I)),
]


def invented_world(sentence: str) -> str | None:
    """Name of the first rule a sentence matches when it says the world, its names or its studies are invented
    (a giveaway for synthetic papers), else None. Plain uses such as "real-world settings" or "synthetic data
    for training" do not match."""
    for name, pat in INVENTED_RULES:
        if pat.search(sentence):
            return name
    return None


# ------------------------------------------------------------------ citations
_YEAR = r"(?:1[6-9]|20)\d{2}[a-z]?"
_NOT_AUTHOR = (r"(?!(?:In|Since|By|Until|From|Before|After|During|Between|January|February|March|April|May|June|July|"
               r"August|September|October|November|December|Table|Figure|Section|Version|Phase|Spring|Summer|Autumn|"
               r"Fall|Winter|The|This|That|These|Those|Of|At|To|And|Or)\b)")
CITATION = re.compile("|".join([
    r"\[\s*\d+(?:\s*[-,;]\s*\d+)*\s*\]",                                        # [12], [3-5], [16, 17]
    r"\[\^[\w-]+\]",                                                            # footnote [^12]
    r"\^\(?[\d,\s-]+\)?\^",                                                     # ^12^, ^(4-5,9)^
    r"<sup>[\d,\s-]+</sup>",
    rf"\(\s*(?:see\s+(?:also\s+)?|e\.g\.,?\s+|cf\.\s+|i\.e\.,?\s+)?[A-Z][^()]{{0,150}}?\b{_YEAR}\s*\)",   # (Smith et al., 2019; ...)
    # case name, optional reporter citation: Laper v. Dravun Media, Smith v. Jones, 123 F.3d 456
    r"\b[A-Z][\w'.&-]*(?:\s+(?:[A-Z][\w'.&-]*|of|the|and|for))*\s+v\.?\s+[A-Z][\w'.&-]*"
    r"(?:\s+(?:[A-Z&][\w'.&-]*|of|the|and|for))*(?:,\s+\d+\s+[A-Z][\w.\s]{0,12}?\s\d+)?",
    # Smith et al. (2019) / Smith and Lee (2019) / Smith (2019)
    rf"\b{_NOT_AUTHOR}[A-Z][\w'-]+(?:\s+(?:et\s+al\.?|(?:and|&)\s+[A-Z][\w'-]+))?\s*\(\s*{_YEAR}\s*\)",
    # Smith et al. 2019 / the Braliram 2023 study / the Brudes Lab 2024 study
    rf"\b{_NOT_AUTHOR}[A-Z][a-z][\w'-]*(?:\s+(?:et\s+al\.?|Lab|Group|and\s+[A-Z][\w'-]+))?,?\s+{_YEAR}\b(?!\s*[-/]\s*\d)",
]))


def _merge(spans: list[tuple[int, int]], text: str, glue: str = r"[\s,;]*") -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and (a <= out[-1][1] or re.fullmatch(glue, text[out[-1][1]:a])):
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def find_citations(text: str) -> list[tuple[int, int]]:
    """Citation spans; neighbouring citations joined only by spaces, commas or semicolons count as one (a case name
    and its "(Court, 2017)" tail, "[3], [4]")."""
    return _merge([m.span() for m in CITATION.finditer(text)], text)


# ------------------------------------------------------------------ names
_CAP = r"[A-Z][\w'&.-]*[A-Za-z0-9]"
_CONNECT = r"(?:of|the|for|de|van|von|der|du|la)"   # not "and": "A and B" is two names
PROVISION = re.compile(rf"\b(?:[Ss]ections?|[Aa]rticles?|[Aa]rt\.|[Rr]ules?|[Rr]egulations?|[Cc]lauses?|s\.|ss\.|§§?)"
                       rf"\s*\d+[\w()]*(?:\(\w+\))*\s+of\s+(?:the\s+)?{_CAP}(?:\s+(?:{_CONNECT}\s+)*{_CAP})*")
CAP_SEQ = re.compile(rf"{_CAP}(?:\s+(?:{_CONNECT}\s+)*{_CAP})*|[A-Z]\b")
ACRONYM_OK = {"CI", "HR", "OR", "RR", "SD", "SE", "SEM", "SMD", "MD", "IQR", "RCT", "RCTs", "I", "II", "III", "IV",
              "V", "VI", "AND", "NOT"}
COMMON_CAP = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "Its", "In", "On", "At", "By", "For", "From", "To", "Of",
    "With", "Without", "As", "And", "But", "Or", "Nor", "If", "When", "While", "Where", "Whether", "Because", "Although",
    "Though", "However", "Thus", "Therefore", "Hence", "Moreover", "Furthermore", "Also", "First", "Second", "Third",
    "Finally", "Second", "Both", "Each", "Every", "All", "No", "Not", "Neither", "Either", "One", "Two", "Three",
    "Such", "Some", "Many", "Most", "Other", "Another", "We", "Our", "They", "Their", "He", "She", "His", "Her", "There",
    "Here", "What", "Which", "Who", "Whose", "How", "Why", "Table", "Tables", "Figure", "Figures", "Fig", "Section",
    "Sections", "Appendix", "Part", "Chapter", "Step", "Phase", "Type", "Group", "Model", "Study", "Studies",
    "Overall", "Taken", "Together", "Across", "Among", "After", "Before", "Under", "Over", "Between", "Within",
    "Despite", "Given", "Instead", "Only", "Yet", "Still", "Even", "Similarly", "Likewise", "Conversely", "Notably",
    "Importantly", "In addition", "Accordingly", "Consequently", "Nevertheless", "Nonetheless", "Rather", "Since",
    "Until", "Unlike", "Like", "Once", "Then", "So", "Do", "Does", "Is", "Are", "Was", "Were", "Can", "Could", "May",
    "Might", "Should", "Would", "Will", "Must", "Note", "See", "Let", "Consider", "Using", "Based", "Following",
}


def generic_names(text: str, starts: Sequence[int] = ()) -> list[tuple[int, int]]:
    """Name spans found without a world: provisions ("section 1 of the Data Privacy Act", "Article 17 of the
    GDPR"), runs of capitalised words (joined by "of", "the", ...), acronyms and name-number tokens (BRUMON-6,
    Thipom-7B, GPT-4). A run's leading common word (The, In, However, ...) is not
    part of the name. A single capitalised word at a sentence start (`starts`: offsets in text) is not a name;
    neither are common words (COMMON_CAP) and statistics acronyms (ACRONYM_OK)."""
    out = [m.span() for m in PROVISION.finditer(text)]
    taken = list(out)
    at_start = set(starts) | {0}
    for m in CAP_SEQ.finditer(text):
        a, b = m.span()
        if any(a < y and x < b for x, y in taken):
            continue
        toks = list(re.finditer(r"\S+", text[a:b]))
        while toks and toks[0].group() in COMMON_CAP:          # "The BRUMON-6", "However Smith"
            toks.pop(0)
        while toks and re.fullmatch(_CONNECT, toks[-1].group()):
            toks.pop()
        if not toks:
            continue
        a2, b2 = a + toks[0].start(), a + toks[-1].end()
        word = text[a2:b2]
        if len(toks) == 1:
            first_of_sentence = a2 in at_start or re.search(r"(?:^|[.!?:]\s|\n)[\"'(\[]?$", text[max(0, a2 - 3):a2])
            if word in ACRONYM_OK or word in COMMON_CAP or len(word) == 1:
                continue
            if first_of_sentence and not re.search(r"[A-Z].*[A-Z]|\d", word):   # "Laper relies" vs "BRUMON-6 shows"
                continue
        out.append((a2, b2))
    return out


def world_names(table) -> Callable[[str, Sequence[int]], list[tuple[int, int]]]:
    def names(text: str, starts: Sequence[int] = ()) -> list[tuple[int, int]]:
        return [(m.start, m.end) for m in find_mentions(text, table)]
    return names


# ------------------------------------------------------------------ masking and measuring
WORD = re.compile(r"\[(?:NAME|CIT|NUM)\]|[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
HEDGES = ("may", "might", "could", "perhaps", "possibly", "possible", "probably", "probable", "likely", "unlikely",
          "suggest", "suggests", "suggested", "suggesting", "appear", "appears", "appeared", "seem", "seems", "seemed",
          "apparently", "presumably", "plausibly", "plausible", "potentially", "indicate", "indicates", "indicated",
          "tend", "tends", "tended", "approximately", "roughly", "largely", "generally", "somewhat", "relatively",
          "arguably", "uncertain", "unclear", "to some extent", "in part")
HEDGE = re.compile(r"\b(?:" + "|".join(re.escape(h).replace(r"\ ", r"\s+") for h in HEDGES) + r")\b", re.I)


def _replace(text: str, spans: list[tuple[int, int]], token: str) -> str:
    out, cur = [], 0
    for a, b in spans:
        if a < cur:
            continue
        out += [text[cur:a], token]
        cur = b
    return "".join(out + [text[cur:]])


def _in_placeholder(t: str, a: int, b: int) -> bool:
    return "[CIT]" in t[a:b] or (a > 0 and t[a - 1] == "[" and t[b:b + 1] == "]")


@dataclass
class Masked:
    text: str
    words: int
    hedges: int
    cites: int
    numbers: int


def mask_sentence(s: str, namer: Callable[[str, Sequence[int]], list[tuple[int, int]]]) -> Masked:
    """Citations -> [CIT], names -> [NAME] (namer), numbers -> [NUM]; counts on the masked text."""
    cits = find_citations(s)
    t = norm_text(_replace(s, cits, " [CIT] "))           # find_mentions offsets are in norm_text
    names = [(a, b) for a, b in _merge(namer(t, ()), t, glue=r"") if not _in_placeholder(t, a, b)]
    t = _replace(t, names, "[NAME]")
    nums = [(a, b) for a, b, _ in numbers_in(t)]
    t = _replace(t, nums, "[NUM]")
    t = re.sub(r"\s+", " ", t).strip()
    return Masked(t, len(WORD.findall(t)), len(HEDGE.findall(t)), len(cits), len(nums))


def measure_paragraph(text: str, rules, namer) -> tuple[dict[str, Any] | None, list[tuple[str, str]]]:
    """(property row or None, dropped invented-world sentences as (rule, sentence))."""
    dropped: list[tuple[str, str]] = []
    kept: list[Masked] = []
    for a, b in _split_unit(text, rules):
        s = text[a:b]
        if (rule := invented_world(s)):
            dropped.append((rule, s))
            continue
        m = mask_sentence(s, namer)
        if m.words:
            kept.append(m)
    w = sum(m.words for m in kept)
    if not kept or w < 10:
        return None, dropped
    row = {"sent_len": w / len(kept),
           "hedges": 100 * sum(m.hedges for m in kept) / w,
           "cites": 100 * sum(m.cites for m in kept) / w,
           "numbers": 100 * sum(m.numbers for m in kept) / w,
           "words": w, "sentences": len(kept)}
    return row, dropped


def measure_paper(md: str, rules, namer, paper: str = "", max_paras: int = MAX_PARAS) -> list[dict[str, Any]]:
    rows = []
    for i, (sec, t) in enumerate(body_paragraphs(md, max_paras)):
        row, _ = measure_paragraph(t, rules, namer)
        if row:
            rows.append(row | {"paper": paper, "para": i, "section": sec})
    return rows


def invented_sentences(md: str, rules, paper: str = "") -> list[dict[str, Any]]:
    """Every invented-world sentence of the paper (all sections, paragraphs and list items, also the abstract)."""
    p = normalize_paper(md)
    out, section = [], ""
    for u in p.units:
        t = p.text[u.start:u.end]
        if u.kind == "heading":
            section = t
            continue
        if u.kind == "cell":
            continue
        for a, b in _split_unit(t, rules):
            if (rule := invented_world(t[a:b])):
                out.append({"paper": paper, "section": section, "rule": rule, "sentence": t[a:b]})
    return out


# ------------------------------------------------------------------ statistics (pure Python; scipy is not installed)
def _kolmogorov_sf(lam: float) -> float:
    """P(K > lam) for the Kolmogorov distribution: 2 sum (-1)^(j-1) exp(-2 j^2 lam^2)."""
    if lam < 0.2:
        return 1.0
    s = sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return min(1.0, max(0.0, 2 * s))


def _ks_exact_p(n: int, m: int, d: float) -> float:
    """Exact two-sided P(D >= d) for two samples of sizes n, m without ties: share of the C(n+m, n) lattice paths
    from (0,0) to (n,m) that leave the band |i/n - j/m| < d (path counting, as in Hodges 1957)."""
    eps = 1e-9
    total = math.comb(n + m, n)
    inside = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if abs(i / n - j / m) >= d - eps:
                continue
            if i == 0 and j == 0:
                inside[i][j] = 1
            else:
                inside[i][j] = (inside[i - 1][j] if i else 0) + (inside[i][j - 1] if j else 0)
    return min(1.0, max(0.0, 1 - inside[n][m] / total))


def ks_2samp(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Two-sample two-sided Kolmogorov-Smirnov test. D = max |F_x - F_y|. p: exact path count when n*m <= 10,000
    (ties are ignored, as in scipy), else the Kolmogorov limit with Stephens' small-sample correction
    lam = (sqrt(ne) + 0.12 + 0.11 / sqrt(ne)) D, ne = n m / (n + m)."""
    xs, ys = sorted(x), sorted(y)
    n, m = len(xs), len(ys)
    if not n or not m:
        raise ValueError("both samples need at least one value")
    d, i, j = 0.0, 0, 0
    while i < n and j < m:
        v = min(xs[i], ys[j])
        while i < n and xs[i] == v:
            i += 1
        while j < m and ys[j] == v:
            j += 1
        d = max(d, abs(i / n - j / m))
    if n * m <= 10_000:
        return {"D": d, "p": _ks_exact_p(n, m, d), "method": "exact", "n": n, "m": m}
    ne = n * m / (n + m)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    return {"D": d, "p": _kolmogorov_sf(lam), "method": "asymptotic", "n": n, "m": m}


def _gammainc_upper(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x): series for x < a + 1, continued fraction otherwise
    (Numerical Recipes 6.2)."""
    if x <= 0:
        return 1.0
    lg = math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(10_000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - lg))
    tiny = 1e-300
    b, c, dd = x + 1 - a, 1 / tiny, 1 / (x + 1 - a)
    h = dd
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        dd = an * dd + b
        dd = tiny if abs(dd) < tiny else dd
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        dd = 1 / dd
        h *= dd * c
        if abs(dd * c - 1) < 1e-15:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - lg) * h)


def chi2_sf(x: float, df: int) -> float:
    """P(X > x) for a chi-square variable with df degrees of freedom."""
    return _gammainc_upper(df / 2, x / 2)


def chi2_cramers_v(a: Mapping[str, int], b: Mapping[str, int]) -> dict[str, Any]:
    """Chi-square test of independence on the 2 x k table (rows: the two count tables, e.g. real and synthetic fact
    kinds; columns: the union of kinds, kinds with zero in both are left out). Cramer's V = sqrt(chi2 / (N (min(r,
    c) - 1))) = sqrt(chi2 / N) for two rows. Also returns the smallest expected count (chi-square is rough when it
    is below 5)."""
    kinds = sorted(k for k in set(a) | set(b) if a.get(k, 0) + b.get(k, 0) > 0)
    if len(kinds) < 2:
        raise ValueError("need at least two kinds with counts")
    if any(v < 0 for v in list(a.values()) + list(b.values())):
        raise ValueError("counts must not be negative")
    rows = [[a.get(k, 0) for k in kinds], [b.get(k, 0) for k in kinds]]
    rs = [sum(r) for r in rows]
    if not all(rs):
        raise ValueError("each table needs a positive total")
    cs = [rows[0][i] + rows[1][i] for i in range(len(kinds))]
    n = sum(rs)
    chi2, min_exp = 0.0, math.inf
    for r in range(2):
        for i in range(len(kinds)):
            e = rs[r] * cs[i] / n
            min_exp = min(min_exp, e)
            chi2 += (rows[r][i] - e) ** 2 / e
    df = len(kinds) - 1
    return {"chi2": chi2, "df": df, "V": math.sqrt(chi2 / n), "p": chi2_sf(chi2, df), "n": n,
            "kinds": kinds, "min_expected": min_exp,
            "share_a": {k: rows[0][i] / rs[0] for i, k in enumerate(kinds)},
            "share_b": {k: rows[1][i] / rs[1] for i, k in enumerate(kinds)}}


def holm(ps: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def paper_bootstrap(rows: list[dict[str, Any]], key: str, resamples: int = RESAMPLES, seed: int = SEED) -> list[float]:
    """95% percentile interval of the paragraph mean, resampling whole papers (paragraphs of one paper are not
    independent)."""
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["paper"], []).append(r[key])
    groups = list(by.values())
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        pick = rng.choices(groups, k=len(groups))
        vals = [v for g in pick for v in g]
        means.append(sum(vals) / len(vals))
    means.sort()
    return [round(means[int(0.025 * resamples)], 4), round(means[min(resamples - 1, math.ceil(0.975 * resamples) - 1)], 4)]


def _q(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"paragraphs": len(rows), "papers": len({r["paper"] for r in rows})}
    for k in PROPS:
        xs = [r[k] for r in rows]
        mu = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1))
        out[k] = {"mean": round(mu, 3), "mean_ci": paper_bootstrap(rows, k), "sd": round(sd, 3),
                  "median": round(_q(xs, 0.5), 3), "p10": round(_q(xs, 0.1), 3), "p90": round(_q(xs, 0.9), 3),
                  "share_zero": round(sum(x == 0 for x in xs) / len(xs), 3)}
    return out


def compare(real: list[dict[str, Any]], synth: list[dict[str, Any]], real_kinds: Mapping[str, int] | None = None,
            synth_kinds: Mapping[str, int] | None = None) -> list[dict[str, Any]]:
    """One row per test: property, effect size (D or Cramer's V) first, then real and synthetic means, raw p and
    Holm-corrected p. Holm runs over all tests present (5 with fact kinds, else 4; `holm_over` says which)."""
    res = []
    for k in PROPS:
        x, y = [r[k] for r in real], [r[k] for r in synth]
        t = ks_2samp(x, y)
        res.append({"property": k, "test": f"KS ({t['method']})", "effect": "D", "effect_size": round(t["D"], 4),
                    "real_mean": round(sum(x) / len(x), 3), "synth_mean": round(sum(y) / len(y), 3),
                    "p": t["p"], "n_real": len(x), "n_synth": len(y)})
    if real_kinds is not None and synth_kinds is not None:
        c = chi2_cramers_v(real_kinds, synth_kinds)
        res.append({"property": "fact_kind_share", "test": "chi-square", "effect": "Cramer's V",
                    "effect_size": round(c["V"], 4), "real_mean": c["share_a"], "synth_mean": c["share_b"],
                    "p": c["p"], "n_real": sum(real_kinds.values()), "n_synth": sum(synth_kinds.values()),
                    "min_expected": c["min_expected"]})
    for r, p in zip(res, holm([r["p"] for r in res]), strict=True):
        r["p_holm"] = p
        r["holm_over"] = len(res)
    return res


# ------------------------------------------------------------------ synthetic run
def _token_set(spans: list[tuple[int, int]], text: str) -> set[int]:
    return {a + m.start() for a, b in spans for m in re.finditer(r"\S+", text[a:b])}


def run_synth(seed_dir: Path, ds_dir: Path, max_paras: int = MAX_PARAS) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure the synthetic papers of one domain under world masking and generic masking. Returns
    (realism_synth, invented_world_sentences)."""
    man = json.loads((seed_dir / "paperfactory.json").read_text(encoding="utf-8"))
    ents = json.loads((seed_dir / "entities.json").read_text(encoding="utf-8"))
    f = packs.field(man["field"])
    w = {"entities": ents}
    rules, base = split_rules(f, {"entities": {}}), alias_table(w, f)
    rows_w, rows_g, inv = [], [], []
    tp = fp = fn = 0
    for p in man["papers"]:
        pid = p["id"]
        md = (ds_dir / pid / "paper.md").read_text(encoding="utf-8")
        table = alias_table(w, f, defined_terms(normalize_paper(md).text, base, f))
        rows_w += measure_paper(md, rules, world_names(table), pid, max_paras)
        rows_g += measure_paper(md, rules, generic_names, pid, max_paras)
        inv += invented_sentences(md, rules, pid)
        for _, t in body_paragraphs(md, max_paras):
            for a, b in _split_unit(t, rules):
                s = t[a:b]
                s2 = norm_text(_replace(s, find_citations(s), " [CIT] "))
                ws = _token_set([x for x in _merge(world_names(table)(s2), s2, r"") if not _in_placeholder(s2, *x)], s2)
                gs = _token_set([x for x in _merge(generic_names(s2), s2, r"") if not _in_placeholder(s2, *x)], s2)
                tp, fp, fn = tp + len(ws & gs), fp + len(gs - ws), fn + len(ws - gs)
    check = {"rule_world": "text.alias_table + defined_terms + find_mentions (world entity names)",
             "rule_generic": "generic_names: capitalised word runs, acronyms, name-number tokens; no world",
             "token_agreement": {"generic_precision": round(tp / max(1, tp + fp), 3),
                                 "generic_recall": round(tp / max(1, tp + fn), 3),
                                 "tokens_world": tp + fn, "tokens_generic": tp + fp}, "properties": {}}
    for k in PROPS:
        a, b = [r[k] for r in rows_w], [r[k] for r in rows_g]
        t = ks_2samp(a, b)
        diffs = [y - x for x, y in zip(a, b, strict=True)]
        check["properties"][k] = {"mean_world": round(sum(a) / len(a), 3), "mean_generic": round(sum(b) / len(b), 3),
                                  "mean_diff": round(sum(diffs) / len(diffs), 3),
                                  "rel_diff": round(sum(diffs) / max(1e-9, abs(sum(a))), 4),
                                  "mean_abs_paragraph_diff": round(sum(abs(d) for d in diffs) / len(diffs), 3),
                                  "ks_D_world_vs_generic": round(t["D"], 4)}
    by_rule: dict[str, int] = {}
    for x in inv:
        by_rule[x["rule"]] = by_rule.get(x["rule"], 0) + 1
    inv_out = {"domain": seed_dir.name, "detector": "paperfactory.realism.invented_world",
               "sentences": len(inv), "papers_with_any": len({x["paper"] for x in inv}),
               "papers": len(man["papers"]), "by_rule": by_rule, "items": inv}
    out = {"domain": seed_dir.name, "field": man["field"], "method": METHOD, "side": "synthetic only",
           "masking": "world", "summary": summarize(rows_w), "summary_generic_masking": summarize(rows_g),
           "masking_check": check, "invented_world_sentences": len(inv),
           "note": "No real-vs-synthetic tests yet (real papers not chosen). Paragraph-level tests treat paragraphs "
                   "as independent; paragraphs of one paper are not, so KS p-values will be optimistic.",
           "rows": rows_w}
    return out, inv_out


# ------------------------------------------------------------------ real papers (Part E)
MIN_REAL = 25                       # fewer kept real papers in a domain -> its comparison is marked pending
JUDGE = "claude-haiku-5-5"          # the only reader allowed for fact kinds (Aditya, 26 Nov 2026: no Sonnet)
KIND_SAMPLE, KIND_SEED = 15, 2026   # papers per side per domain read for fact kinds


def real_papers(realism_dir: Path, field_name: str) -> list[dict[str, Any]]:
    """Manifest rows of the downloaded real papers of one field (realism_dir/manifest.jsonl, Markdown in
    realism_dir/<field>/<id>.md), in manifest order."""
    man = realism_dir / "manifest.jsonl"
    rows = [json.loads(x) for x in man.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r["domain"] == field_name]


def measure_real(realism_dir: Path, field_name: str, max_paras: int = MAX_PARAS
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """(paragraph rows, invented-world sentences, papers) of the real papers of one field: the same preparation as
    the synthetic side, with generic name masking (a real paper has no world)."""
    f = packs.field(field_name)
    rules = split_rules(f, {"entities": {}})
    rows, inv = [], []
    papers = real_papers(realism_dir, field_name)
    for p in papers:
        md = (realism_dir / field_name / f"{p['id']}.md").read_text(encoding="utf-8")
        rows += measure_paper(md, rules, generic_names, p["id"], max_paras)
        inv += invented_sentences(md, rules, p["id"])
    return rows, inv, len(papers)


def synth_generic_rows(seed_dir: Path, ds_dir: Path, max_paras: int = MAX_PARAS) -> list[dict[str, Any]]:
    """Synthetic paragraph rows under generic masking (the masking the real side gets), for the second comparison."""
    man = json.loads((seed_dir / "paperfactory.json").read_text(encoding="utf-8"))
    rules = split_rules(packs.field(man["field"]), {"entities": {}})
    rows = []
    for p in man["papers"]:
        md = (ds_dir / p["id"] / "paper.md").read_text(encoding="utf-8")
        rows += measure_paper(md, rules, generic_names, p["id"], max_paras)
    return rows


def kind_counts(papers: list[dict[str, Any]]) -> dict[str, int]:
    """Fact-kind counts (reader tool names) summed over the papers of a realism_kinds.json side."""
    out: dict[str, int] = {}
    for p in papers:
        for k, v in p["kinds"].items():
            out[k] = out.get(k, 0) + v
    return dict(sorted(out.items()))


def kinds_tables(kinds: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, int]] | None:
    """(real counts, synthetic counts) from a complete realism_kinds.json, else None."""
    if not kinds or not kinds.get("complete"):
        return None
    return kind_counts(kinds["real"]["papers"]), kind_counts(kinds["synthetic"]["papers"])


def compare_domain(field_name: str, real_rows: list[dict[str, Any]], n_real: int, synth: Mapping[str, Any],
                   synth_generic: list[dict[str, Any]] | None = None, kinds: Mapping[str, Any] | None = None,
                   real_invented: int = 0) -> dict[str, Any]:
    """realism_compare.json of one domain: the main test table (real generic masking vs synthetic world masking, as
    stored in realism_synth.json), the same with generic masking on both sides, and summaries. A domain with fewer
    than MIN_REAL real papers is marked pending (numbers still given). Fact kinds join the table (5 tests under
    Holm) only when a complete realism_kinds.json is given."""
    kt = kinds_tables(kinds or {})
    table = compare(real_rows, synth["rows"], *(kt or (None, None)))
    for r in table:
        if r["property"] == "fact_kind_share":
            r["judge"] = JUDGE
            r["note"] = kinds["synthetic"].get("note", "")
    out: dict[str, Any] = {
        "domain": synth["domain"], "field": field_name,
        "status": "ok" if n_real >= MIN_REAL else f"pending: only {n_real} real papers (< {MIN_REAL})",
        "method": METHOD, "real_masking": "generic_names", "synth_masking": "world (realism_synth.json rows)",
        "note": "Paragraph-level KS tests treat paragraphs as independent; paragraphs of one paper are not, so raw "
                "p-values are optimistic. Effect sizes first.",
        "real_papers": n_real, "synth_papers": synth["summary"]["papers"],
        "tests": table,
        "real_summary": summarize(real_rows), "synth_summary": synth["summary"],
        "invented_world_sentences": {"real": real_invented, "synthetic": synth.get("invented_world_sentences")}}
    if synth_generic:
        out["tests_generic_both"] = compare(real_rows, synth_generic)
        out["synth_summary_generic"] = summarize(synth_generic)
    if kinds:
        out["fact_kinds"] = {k: kinds[k] for k in ("judge", "complete", "cost", "sample") if k in kinds} | {
            "real": kind_counts(kinds["real"]["papers"]), "synthetic": kind_counts(kinds["synthetic"]["papers"]),
            "synthetic_by_writer": kinds["synthetic"].get("by_writer", {})}
    return out


# ------------------------------------------------------------------ fact kinds with the blind reader (Haiku)
def sample_real(papers: list[dict[str, Any]], n: int = KIND_SAMPLE, seed: int = KIND_SEED) -> list[dict[str, Any]]:
    """n real papers, fixed seed, in input order; all if there are n or fewer."""
    if len(papers) <= n:
        return list(papers)
    keep = {p["id"] for p in random.Random(seed).sample(papers, n)}
    return [p for p in papers if p["id"] in keep]


def read_kinds(reader, items: list[tuple[str, str, str, str]], spent: Callable[[], float], budget: float,
               workers: int = 8) -> tuple[dict[str, dict[str, int]], dict[str, int], bool]:
    """Read (side, paper id, section key, section text) items with readback.Reader-like `reader.read(pid, sec, md)`
    and count valid calls per tool. Stops starting new sections once spent() >= budget. Returns ({paper: {tool:
    count}}, {"sections", "read", "skipped_budget", "rejected", "repaired"}, complete)."""
    from concurrent.futures import ThreadPoolExecutor

    from paperfactory.readback import Section
    counts: dict[str, dict[str, int]] = {}
    stats = {"sections": len(items), "read": 0, "unread": 0, "skipped_budget": 0, "rejected": 0, "repaired": 0}

    def one(it: tuple[str, str, str, str]) -> tuple[str, dict[str, Any] | None]:
        side, pid, key, text = it
        if spent() >= budget:
            return f"{side}:{pid}", None
        res = reader.read(pid, Section(0, len(text), key), text)
        return f"{side}:{pid}", res
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for pid, res in ex.map(one, items):
            c = counts.setdefault(pid, {})
            if res is None:
                stats["skipped_budget"] += 1
                continue
            stats["read" if res["read"] else "unread"] += 1
            stats["rejected"] += res.get("dropped", 0)
            stats["repaired"] += res.get("repaired", 0)
            for call in res["calls"]:
                c[call["tool"]] = c.get(call["tool"], 0) + 1
    return counts, stats, stats["skipped_budget"] == 0 and stats["unread"] == 0


def kinds_plan(field_name: str, seed_dir: Path, ds_dir: Path, realism_dir: Path, n: int = KIND_SAMPLE
               ) -> dict[str, Any]:
    """The fixed-seed sample of both sides and their sections (readback.sections), no model calls."""
    from paperfactory.numspans import writer_of
    from paperfactory.readback import sample_papers, sections
    man = json.loads((seed_dir / "paperfactory.json").read_text(encoding="utf-8"))
    syn = sample_papers(man["papers"], n, KIND_SEED)
    real = sample_real(real_papers(realism_dir, field_name), n, KIND_SEED)
    items: list[tuple[str, str, str, str]] = []
    writers = {}
    for p in syn:
        md = (ds_dir / p["id"] / "paper.md").read_text(encoding="utf-8")
        writers[p["id"]] = p.get("writer") or writer_of(p)
        items += [("synthetic", p["id"], f"s{i}", s.text(md)) for i, s in enumerate(sections(md))]
    for p in real:
        md = (realism_dir / field_name / f"{p['id']}.md").read_text(encoding="utf-8")
        items += [("real", p["id"], f"s{i}", s.text(md)) for i, s in enumerate(sections(md))]
    return {"synthetic": [p["id"] for p in syn], "real": [p["id"] for p in real], "writers": writers,
            "items": items}


def kinds_result(field_name: str, plan: Mapping[str, Any], counts: Mapping[str, Mapping[str, int]],
                 stats: Mapping[str, int], complete: bool, cost: Any) -> dict[str, Any]:
    """realism_kinds.json: per-paper tool counts on both sides, marked with the judge and, for synthetic papers
    written by the judge model, "same model as writer"."""
    from paperfactory.numspans import SAME_MODEL_NOTE
    syn = []
    for pid in plan["synthetic"]:
        w = plan["writers"][pid]
        row = {"paper": pid, "writer": w, "kinds": dict(sorted(counts.get(f"synthetic:{pid}", {}).items())),
               "judge": JUDGE}
        if w == JUDGE:
            row["note"] = SAME_MODEL_NOTE
        syn.append(row)
    real = [{"paper": pid, "kinds": dict(sorted(counts.get(f"real:{pid}", {}).items())), "judge": JUDGE}
            for pid in plan["real"]]
    by_writer: dict[str, Any] = {}
    for w in sorted({r["writer"] for r in syn}):
        rows = [r for r in syn if r["writer"] == w]
        by_writer[w] = {"papers": len(rows), "kinds": kind_counts(rows), "judge": JUDGE}
        if w == JUDGE:
            by_writer[w]["note"] = SAME_MODEL_NOTE
    n_same = sum(r["writer"] == JUDGE for r in syn)
    return {"field": field_name, "judge": JUDGE, "complete": complete, "cost": cost, "stats": dict(stats),
            "sample": {"n": KIND_SAMPLE, "seed": KIND_SEED, "synthetic": "readback.sample_papers (stratified by "
                       "writer)", "real": "random.sample of the downloaded real papers"},
            "kinds_are": "valid reader tool calls (thesis_scorer.facttools) per section, readback.Reader, prompt "
                         "readback_v1; sections from readback.sections on both sides",
            "real": {"papers": real, "kinds": kind_counts(real), "judge": JUDGE},
            "synthetic": {"papers": syn, "kinds": kind_counts(syn), "judge": JUDGE, "by_writer": by_writer,
                          "note": f"{n_same} of {len(syn)} synthetic papers written by {JUDGE}: {SAME_MODEL_NOTE}"
                          if n_same else ""}}


def kinds_caller(field_name: str, out_dir: Path, genre: str = "argument"):
    from paperfactory.config import Settings
    from paperfactory.llm import Caller
    from paperfactory.readback import JUDGE_ROLE, READER_ROLE
    s = Settings(field=field_name, genre=genre, models={READER_ROLE: {"model": JUDGE, "max_parallel": 8},
                                                        JUDGE_ROLE: {"model": JUDGE, "max_parallel": 8}},
                 out=str(out_dir))
    return Caller(s)


def spent_total(haiku_dir: Path) -> float:
    """USD of every Haiku call logged under haiku_dir/<domain>/calls.jsonl (the budget is for all domains)."""
    usd = 0.0
    for log in sorted(haiku_dir.glob("*/calls.jsonl")):
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                usd += json.loads(line).get("cost_usd") or 0.0
    return usd


def run_kinds(field_name: str, seed_dir: Path, ds_dir: Path, realism_dir: Path, budget: float,
              max_sections: int | None = None, cache_dir: Path | None = None) -> dict[str, Any]:
    """Read the fact-kind sample of one domain with Haiku (readback.Reader, cached under cache_dir). max_sections:
    cost probe, the first sections of each side only (half each)."""
    from paperfactory.readback import READER_MODEL, Reader
    if READER_MODEL != JUDGE:
        raise SystemExit(f"readback.READER_MODEL is {READER_MODEL}, expected {JUDGE}; stop and report")
    man = json.loads((seed_dir / "paperfactory.json").read_text(encoding="utf-8"))
    caller = kinds_caller(field_name, cache_dir or realism_dir / "haiku" / seed_dir.name, man.get("genre", "argument"))
    plan = kinds_plan(field_name, seed_dir, ds_dir, realism_dir)
    items = plan["items"]
    if max_sections is not None:
        half = max_sections // 2
        items = [x for x in items if x[0] == "synthetic"][:half] + [x for x in items if x[0] == "real"][:max_sections - half]
    reader = Reader(caller, packs.field(field_name))
    counts, stats, complete = read_kinds(reader, items, lambda: spent_total(realism_dir / "haiku"), budget)
    out = kinds_result(field_name, plan, counts, stats, complete and max_sections is None, caller.cost())
    out["spent_all_domains_usd"] = round(spent_total(realism_dir / "haiku"), 4)
    out["sections_by_side"] = {s: sum(x[0] == s for x in plan["items"]) for s in ("real", "synthetic")}
    out["chars_by_side"] = {s: sum(len(x[3]) for x in plan["items"] if x[0] == s) for s in ("real", "synthetic")}
    return out


def report_md(compares: list[dict[str, Any]], pending: Mapping[str, str]) -> str:
    """Short Markdown report: per domain, effect size first, then means, raw p and Holm p."""
    lines = ["# Realism check: real vs synthetic review papers", "",
             "Made by `python -m paperfactory.realism real`. Per body paragraph; effect size first (KS D; Cramer's V "
             "for fact kinds), then the means, raw p and Holm-corrected p. Real side: generic name masking; synthetic "
             "side: world masking (as in realism_synth.json). Paragraph-level p-values are optimistic (paragraphs of "
             "one paper are not independent).", ""]
    for c in compares:
        lines += [f"## {c['field']} ({c['domain']}): {c['real_papers']} real, {c['synth_papers']} synthetic papers"
                  + ("" if c["status"] == "ok" else f" — {c['status']}"), "",
                  "| property | effect | real mean | synthetic mean | p | p (Holm) | n paragraphs real / synth |",
                  "|---|---|---|---|---|---|---|"]
        for t in c["tests"]:
            if t["property"] == "fact_kind_share":
                rm = ", ".join(f"{k} {v:.2f}" for k, v in t["real_mean"].items())
                sm = ", ".join(f"{k} {v:.2f}" for k, v in t["synth_mean"].items())
                lines.append(f"| fact kinds (judge {t['judge']}) | V={t['effect_size']:.3f} | {rm} | {sm} | "
                             f"{t['p']:.2g} | {t['p_holm']:.2g} | {t['n_real']} / {t['n_synth']} facts |")
            else:
                lines.append(f"| {t['property']} | D={t['effect_size']:.3f} | {t['real_mean']:.2f} | "
                             f"{t['synth_mean']:.2f} | {t['p']:.2g} | {t['p_holm']:.2g} | {t['n_real']} / {t['n_synth']} |")
        if c.get("fact_kinds") and not any(t["property"] == "fact_kind_share" for t in c["tests"]):
            lines.append(f"\nFact kinds: incomplete read, not tested ({c['fact_kinds'].get('cost')}).")
        if (n := c.get("fact_kinds", {}).get("synthetic_by_writer")):
            notes = [f"{w}: {v['papers']} papers" + (f" ({v['note']})" if v.get("note") else "") for w, v in n.items()]
            lines.append("\nFact-kind sample by writer: " + "; ".join(notes) + ".")
        g = c.get("tests_generic_both")
        if g:
            lines.append("\nSame with generic masking on both sides: " + "; ".join(
                f"{t['property']} D={t['effect_size']:.3f} (Holm p {t['p_holm']:.2g})" for t in g) + ".")
        iv = c["invented_world_sentences"]
        lines += [f"\nInvented-world sentences (detector hits): real {iv['real']}, synthetic {iv['synthetic']}.", ""]
    for d, why in pending.items():
        lines += [f"## {d}: {why}", ""]
    return "\n".join(lines).rstrip() + "\n"


def run_real(data: Path, realism_dir: Path, fields: Sequence[str], max_paras: int = MAX_PARAS,
             write: bool = True) -> list[dict[str, Any]]:
    """`real` command: measure the real papers per field, compare with seed/bench_<field>/realism_synth.json (and
    realism_kinds.json when present), write seed/bench_<field>/realism_real.json, realism_compare.json and
    seed/realism_report.md."""
    compares, pending = [], {}
    for fld in fields:
        seed_dir, ds_dir = data / "seed" / f"bench_{fld}", data / "dataset" / f"bench_{fld}"
        rows, inv, n = measure_real(realism_dir, fld, max_paras)
        if not n:
            pending[fld] = "pending: no real papers downloaded"
            continue
        synth = json.loads((seed_dir / "realism_synth.json").read_text(encoding="utf-8"))
        kf = seed_dir / "realism_kinds.json"
        kinds = json.loads(kf.read_text(encoding="utf-8")) if kf.is_file() else None
        cmp_ = compare_domain(fld, rows, n, synth, synth_generic_rows(seed_dir, ds_dir, max_paras), kinds, len(inv))
        compares.append(cmp_)
        if write:
            real = {"domain": seed_dir.name, "field": fld, "method": METHOD, "side": "real", "masking": "generic",
                    "papers": n, "summary": summarize(rows), "invented_world_sentences": len(inv),
                    "invented_items": inv, "rows": rows}
            for name, obj in (("realism_real.json", real), ("realism_compare.json", cmp_)):
                (seed_dir / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if write:
        (data / "seed" / "realism_report.md").write_text(report_md(compares, pending), encoding="utf-8")
    return compares


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m paperfactory.realism")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="measure the synthetic papers of each domain")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--domains", nargs="+", required=True)
    s.add_argument("--max-paras", type=int, default=MAX_PARAS)
    r = sub.add_parser("real", help="measure the real papers and compare them with realism_synth.json")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--realism-dir", type=Path, required=True, help="data/cache/realism (manifest.jsonl, <field>/*.md)")
    r.add_argument("--fields", nargs="+", default=["medicine", "law", "materials", "llm"])
    r.add_argument("--max-paras", type=int, default=MAX_PARAS)
    k = sub.add_parser("kinds", help="fact kinds of a fixed sample on both sides, read by Haiku (model calls)")
    k.add_argument("--data", type=Path, required=True)
    k.add_argument("--realism-dir", type=Path, required=True)
    k.add_argument("--fields", nargs="+", required=True)
    k.add_argument("--budget", type=float, required=True, help="USD; no new section is started past it")
    k.add_argument("--max-sections", type=int, default=None, help="cost probe: read only this many sections")
    k.add_argument("--plan-only", action="store_true", help="print section counts and characters, no model calls")
    a = ap.parse_args(argv)
    if a.cmd == "real":
        for c in run_real(a.data, a.realism_dir, a.fields, a.max_paras):
            print(c["field"], c["status"], " ".join(f"{t['property']}:{t['effect']}={t['effect_size']}"
                                                    f"/p_holm={t['p_holm']:.2g}" for t in c["tests"]))
        return
    if a.cmd == "kinds":
        for fld in a.fields:
            seed_dir, ds_dir = a.data / "seed" / f"bench_{fld}", a.data / "dataset" / f"bench_{fld}"
            if a.plan_only:
                plan = kinds_plan(fld, seed_dir, ds_dir, a.realism_dir)
                print(fld, {s: (sum(x[0] == s for x in plan["items"]), sum(len(x[3]) for x in plan["items"]
                                                                           if x[0] == s)) for s in ("real", "synthetic")})
                continue
            out = run_kinds(fld, seed_dir, ds_dir, a.realism_dir, a.budget, a.max_sections)
            print(fld, json.dumps({"complete": out["complete"], "stats": out["stats"], "cost": out["cost"]}))
            if a.max_sections is None:
                (seed_dir / "realism_kinds.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                                                             encoding="utf-8")
        return
    for d in a.domains:
        out, inv = run_synth(a.data / "seed" / d, a.data / "dataset" / d, a.max_paras)
        (a.data / "seed" / d / "realism_synth.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                                                                encoding="utf-8")
        (a.data / "seed" / d / "invented_world_sentences.json").write_text(
            json.dumps(inv, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        sm = out["summary"]
        print(d, sm["papers"], "papers", sm["paragraphs"], "paragraphs |",
              " ".join(f"{k}={sm[k]['mean']}" for k in PROPS), "| invented:", inv["sentences"])


if __name__ == "__main__":
    main()
