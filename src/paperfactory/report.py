"""Part F: the data-quality report for the supervisor (one PDF), made only from the outputs of Parts B-E.

Every number in the PDF is read from a file under data/seed/ (named under each table); nothing is typed by hand.
Wording rules (Aditya, 27 Nov 2026): a "decoy" is called a distractor (true near-miss fact); writer precision is given
as a lower bound (data/seed/bench_readback_notes.json); every Claude-made label and every Haiku-judged number is
marked (Haiku also wrote the bench20h papers: "same model as writer"); the 20-span eye check of Part B is "not done"
unless its labels exist (EYE_CHECK).

Intervals: the exact (Clopper-Pearson) or Wilson intervals stored in the output files, as computed in Parts B-E (the
report names the kind under each table); no interval is recomputed here.

    cd services/paper-factory
    uv run --with reportlab==4.2.5 python -m paperfactory.report --data ../../data --out ../../data/seed/report_partF.pdf

`gather` (no reportlab) collects the numbers; `build_pdf` lays them out.
"""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

DOMAINS = ("bench_law", "bench_llm", "bench_materials", "bench_medicine")
FIELD = {"bench_law": "law", "bench_llm": "llm", "bench_materials": "materials", "bench_medicine": "medicine"}
DISTRACTOR = "distractor (true near-miss fact)"
HAIKU = "claude-haiku-5-5"
SAME = "same model as writer"
EYE_CHECK = "seed/numspans_eyecheck.csv"          # Part B: 20 spans checked by eye (labels file, if it exists)
SAMPLES_PER_DOMAIN = 2
COLORS = {"claim": "#bfe8bf", "decoy": "#f6c1c1", "conflict": "#ffd49a", "number": "#c3d8ff", "invented": "#dcc6f2"}
ASCII = str.maketrans({"\u2019": "'", "\u2014": "-", "\u2013": "-"})
MARK_NAME = {"invented": "possible invented", "claim": "claim", "decoy": "distractor", "conflict": "conflict claim",
             "number": "number"}
PRIORITY = ("invented", "claim", "decoy", "conflict", "number")   # later wins where marks overlap


def _j(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _rows(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def pct(r: float | None, nd: int = 1) -> str:
    return "n/a" if r is None else f"{100 * r:.{nd}f}%"


def ci(c: list[float] | None, nd: int = 1) -> str:
    return "" if not c else f"[{100 * c[0]:.{nd}f}, {100 * c[1]:.{nd}f}]"


# ------------------------------------------------------------------ gather (numbers + their files)
def counts(data: Path, d: str) -> dict[str, Any]:
    seed = data / "seed" / d
    rows = _rows(seed / "facts.jsonl")
    c: Counter = Counter()
    for r in rows:
        if r["what"] == "relation":
            c["relation claims" if r.get("role") == "claim" else "distractors"] += 1
            c["conflict facts"] += bool(r.get("conflict"))
        else:
            c[r["what"]] += 1
    qt = Counter(_j(g)["type"] for g in sorted((seed / "gold").glob("*.json")))
    man = _j(seed / "paperfactory.json")
    return {"papers": len(man["papers"]), "facts": dict(c), "facts_total": len(rows), "questions": dict(qt),
            "writers": dict(Counter(p.get("writer", "?") for p in man["papers"])),
            "source": [f"seed/{d}/facts.jsonl", f"seed/{d}/gold/*.json", f"seed/{d}/paperfactory.json"]}


def gather(data: Path) -> dict[str, Any]:
    from paperfactory.readback import pooled
    out: dict[str, Any] = {"domains": {}, "notes": _j(data / "seed" / "bench_readback_notes.json")}
    rbs = []
    for d in DOMAINS:
        seed = data / "seed" / d
        ns, rb, sc = _j(seed / "numspans.json"), _j(seed / "readback.json"), _j(seed / "shortcut.json")
        rc, syn = _j(seed / "realism_compare.json"), _j(seed / "realism_synth.json")
        inv = _j(seed / "invented_world_sentences.json")
        rbs.append(rb)
        mm = rb.get("matching_misses", {})
        out["domains"][d] = {
            "counts": counts(data, d),
            "fact_recall": ns["fact_recall"], "coverage": ns["coverage"],
            "code_error": ns.get("code_error_rate_quote"),
            "writer_precision": rb["summary"]["writer_precision"],
            "writer_precision_lb": mm.get("writer_precision", {}).get("with_misses_planned"),
            "matching_misses": mm.get("count"), "reader_recall": rb["summary"]["reader_recall"],
            "reader_model": rb["reader_model"], "add_custom_share": rb["summary"]["add_custom_share"],
            "rb_papers": len(rb["papers"]),
            "shortcut": [r for r in sc["summary"] if r["model"] == HAIKU], "shortcut_models": sc.get("models"),
            "shortcut_note": sc.get("note"),
            "realism": rc, "invented_world": syn.get("invented_world_sentences"),
            "invented_papers": inv.get("papers_with_any"), "invented_of": inv.get("papers"),
        }
    out["pooled_rb"] = pooled(rbs)
    out["selection"] = _j(data / "seed" / "realism_selection.json")["domains"]
    lic = _j(data / "seed" / "realism_law_licences.json")
    out["law_licences"] = {"confirmed": lic.get("confirmed"), "candidates": len(lic.get("papers") or [])
                           or lic.get("candidates")}
    out["eye_check_done"] = (data / EYE_CHECK).is_file()
    import csv
    with (data / "seed" / "bench_readback_handcheck_seedcheck.csv").open(encoding="utf-8") as fh:
        hrows = list(csv.DictReader(fh))
    out["hand_n"] = len(hrows)
    out["hand_labels"] = dict(Counter(r.get("your_seed_label", "").strip() for r in hrows))
    return out


# ------------------------------------------------------------------ sample papers: marks from spans
def marks_for(data: Path, d: str, pid: str) -> tuple[str, list[str], dict[str, int], list[str]]:
    """(paper text, per-character mark or "", mark counts, planted facts with no position)."""
    seed = data / "seed" / d
    md = (data / "dataset" / d / pid / "paper.md").read_text(encoding="utf-8")
    man = _j(seed / "paperfactory.json")
    p = next(x for x in man["papers"] if x["id"] == pid)
    conflict = {r["id"].split(":", 1)[1] for r in _rows(seed / "facts.jsonl")
                if r["paper"] == pid and r.get("conflict")}
    spans: dict[str, list[tuple[int, int]]] = {k: [] for k in PRIORITY}
    nopos = []
    for f in p["facts"]:
        k = "conflict" if f["id"] in conflict else f["kind"]
        if f["spans"]:
            spans[k] += [tuple(s) for s in f["spans"]]
        else:
            role = DISTRACTOR if f["kind"] == "decoy" else ("conflict claim" if k == "conflict" else "claim")
            nopos.append(f"{role}: {f['subject']} {f['relation']} {f['object']}")
    for n in p.get("numbers", []):
        if n.get("spans"):
            spans["number"] += [tuple(s) for s in n["spans"]]
        else:
            nopos.append(f"number: {n['label']} = {n['value']}{(' ' + n['unit']) if n.get('unit') else ''}")
    rb = _j(seed / "readback.json")
    rp = next((x for x in rb["papers"] if x["paper"] == pid), None)
    for c in (rp or {}).get("claims", []):
        if c["sort"] == "possible_invented" and c.get("quote"):
            i = md.find(c["quote"], c.get("section_start", 0))
            if i < 0:
                i = md.find(c["quote"])
            if i >= 0:
                spans["invented"].append((i, i + len(c["quote"])))
    mark = [""] * len(md)
    for k in PRIORITY:
        for a, b in spans[k]:
            for j in range(max(0, a), min(len(md), b)):
                mark[j] = k
    return md, mark, {k: len(v) for k, v in spans.items()}, nopos


def pick_samples(data: Path, d: str, n: int = SAMPLES_PER_DOMAIN) -> list[str]:
    """Shortest papers of the read-back sample, one per writer model first (so purple marks exist)."""
    rb = _j(data / "seed" / d / "readback.json")
    ln = {p["paper"]: len((data / "dataset" / d / p["paper"] / "paper.md").read_text(encoding="utf-8"))
          for p in rb["papers"]}
    by_w: dict[str, list[str]] = {}
    for p in sorted(rb["papers"], key=lambda p: ln[p["paper"]]):
        by_w.setdefault(p["writer"], []).append(p["paper"])
    out = [v[0] for _, v in sorted(by_w.items())][:n]
    rest = [p for p in sorted(ln, key=ln.get) if p not in out]
    return out + rest[: n - len(out)]


# ------------------------------------------------------------------ PDF
def build_pdf(g: dict[str, Any], data: Path, out: Path) -> None:
    import reportlab
    from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    fdir = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("Vera", str(fdir / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("Vera-Bold", str(fdir / "VeraBd.ttf")))
    from reportlab.lib.fonts import addMapping
    addMapping("Vera", 0, 0, "Vera")
    addMapping("Vera", 1, 0, "Vera-Bold")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontName="Vera", fontSize=9.5, leading=13, alignment=TA_LEFT)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#444444"))
    cell = ParagraphStyle("c", parent=body, fontSize=7.5, leading=9)
    h1 = ParagraphStyle("h1", parent=body, fontName="Vera-Bold", fontSize=16, leading=20, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=body, fontName="Vera-Bold", fontSize=12, leading=15, spaceBefore=10,
                        spaceAfter=4)
    h3 = ParagraphStyle("h3", parent=body, fontName="Vera-Bold", fontSize=10, leading=13, spaceBefore=6)
    paper_txt = ParagraphStyle("pt", parent=body, fontSize=8, leading=11)
    paper_tab = ParagraphStyle("ptab", parent=body, fontName="Courier", fontSize=6.3, leading=8.5)
    paper_h = ParagraphStyle("ph", parent=body, fontName="Vera-Bold", fontSize=9, leading=12, spaceBefore=4)
    E = html.escape
    st: list[Any] = []

    def P(t: str, s: ParagraphStyle = body) -> Paragraph:
        return Paragraph(t, s)

    def src(*files: str) -> Paragraph:
        return P("Source: " + "; ".join(f"<font name='Courier'>{E(f)}</font>" for f in files), small)

    def table(rows: list[list[Any]], widths: list[float] | None = None, head: int = 1) -> Table:
        rows = [[c if isinstance(c, Paragraph) else P(E(str(c)), cell) for c in r] for r in rows]
        t = Table(rows, colWidths=widths, repeatRows=head)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#999999")),
                               ("BACKGROUND", (0, 0), (-1, head - 1), colors.HexColor("#eeeeee")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
        return t

    D = g["domains"]
    pr = g["pooled_rb"]
    notes = g["notes"]
    hc = notes["hand_check"]

    # ---- title and summary
    st += [P("Benchmark data quality: report on the synthetic papers", h1),
           P("For Tim. Made on 27 Nov 2026 by <font name='Courier'>python -m paperfactory.report</font> from the "
             "outputs of Parts B-E (data/seed/). Four domains: law, llm, materials, medicine. Every number names its "
             "file under the table. Intervals are value [low, high], 95%.", body), Spacer(1, 6),
           P("In short", h2)]
    wp = pr["writer_precision_with_misses_planned"]
    tests = [t for d in DOMAINS for t in D[d]["realism"]["tests"]]
    es_lo, es_hi = min(t["effect_size"] for t in tests), max(t["effect_size"] for t in tests)
    n_sig, n_tests = sum(t["p_holm"] < 0.05 for t in tests), len(tests)
    ans = [r for d in DOMAINS for r in D[d]["shortcut"] if r["type"] not in ("absent", "near_miss", "out_of_scope")]
    sc_right, sc_n = sum(r["right"] for r in ans), sum(r["n"] for r in ans)
    sc_rows = [r for d in DOMAINS for r in D[d]["shortcut"]]
    sc_above, sc_types = sum(bool(r.get("above_guess")) for r in sc_rows), len(sc_rows)
    st.append(table([
        ["Metric", "Result (all four domains)", "Meaning"],
        ["1 Fact recall", "; ".join(f"{FIELD[d]} {pct(D[d]['fact_recall']['recall'])} "
                                    f"{ci(D[d]['fact_recall']['ci95'])}" for d in DOMAINS),
         "Planted facts that we can find at a place in their paper."],
        ["2 Writer precision", f"at least {pct(wp['rate'])} {ci(wp['ci95_exact'])} (lower bound); reader recall "
         f"{pct(pr['reader_recall']['rate'])} {ci(pr['reader_recall']['ci95_exact'])}",
         "Facts a blind reader finds in the papers that we planted, out of all it finds about named things."],
        ["3 Realism", f"real and synthetic papers differ on every property (effect sizes {es_lo:.2f}-{es_hi:.2f}; "
         f"{n_sig} of {n_tests} tests with Holm p < 0.05)",
         "Do the synthetic papers read like real review papers?"],
        ["4 Shortcut rate", f"closed-book Claude Haiku: {sc_right} of {sc_n} answerable questions right; "
         f"{sc_above} of {sc_types} question-type rows above guessing",
         "Questions answered right with no papers at all."],
    ], [3 * cm, 7 * cm, 7 * cm]))
    st.append(src("seed/bench_*/numspans.json (fact_recall)", "seed/bench_*/readback.json (matching_misses)",
                  "seed/bench_*/realism_compare.json", "seed/bench_*/shortcut.json"))

    # ---- who judged what
    st += [P("Who judged what", h2), table([
        ["Numbers or labels", "Made by", "Note"],
        [f"Hand check of {g['hand_n']} possible invented facts ({len(hc['planned_misses'])} planned misses; seed "
         f"labels: " + ", ".join(f"{v} '{k}'" for k, v in g["hand_labels"].items()) + ")",
         "Claude (not a human)", "data/seed/bench_readback_notes.json"],
        ["Haiku first pass on the same items", HAIKU, f"{SAME}; missed every planned {DISTRACTOR}; not used"],
        ["Number spans placed by the verifier (part of fact recall); code error rate 3.9% [2.0%, 6.8%]", HAIKU,
         SAME],
        ["Writer precision and reader recall (blind reader, 26 Nov 2026)", D["bench_law"]["reader_model"],
         "run before the 'no Sonnet' rule; not the writer of any paper"],
        ["Shortcut answers and their scoring", HAIKU, f"answerer and the scorer's claim reader; {SAME}"],
        ["Fact-kind shares (realism property 5)", HAIKU, "real papers: Haiku-judged, no seed to check; "
         f"synthetic papers written by Haiku: {SAME}"],
        ["Part B: 20 number spans checked by eye", "done" if g["eye_check_done"] else "NOT DONE",
         "no labels file" if not g["eye_check_done"] else EYE_CHECK],
    ], [7.5 * cm, 3.2 * cm, 6.3 * cm])]
    st.append(P(f"Haiku wrote the bench20h papers ({sum(D[d]['counts']['writers'].get(HAIKU, 0) for d in DOMAINS)} "
                f"of {sum(D[d]['counts']['papers'] for d in DOMAINS)}); Qwen wrote the rest. A Haiku judgement "
                f"of a Haiku-written paper may favour its own wording, so every Haiku-judged number is marked "
                f"'{SAME}'.", small))

    # ---- how the data is made
    st += [PageBreak(), P("How the data is made", h2)]
    dr = Drawing(17 * cm, 5.2 * cm)
    boxes = [("World", "invented entities:\nstudies, trials,\nlegal cases, methods,\nmaterials"),
             ("Planted facts", "claims, distractors\n(true near-miss\nfacts), conflicts;\nnumbers by code\n(quant.py)"),
             ("Papers", "written by Haiku\nor Qwen from a\nbrief; checked\nand repaired"),
             ("Tag step", "places each fact\nin its paper\n(spans); numbers:\ncode + verifier"),
             ("Questions", "and gold answers,\nmade by code\nfrom the fact rows")]
    w, gap = 2.8 * cm, 0.7 * cm
    for i, (t, sub) in enumerate(boxes):
        x = i * (w + gap)
        dr.add(Rect(x, 0.6 * cm, w, 4.2 * cm, fillColor=colors.HexColor("#f2f6fb"), strokeColor=colors.HexColor(
            "#4a6a8a"), strokeWidth=0.8, rx=4, ry=4))
        dr.add(String(x + w / 2, 4.2 * cm, t, fontName="Vera-Bold", fontSize=9, textAnchor="middle"))
        for k, line in enumerate(sub.split("\n")):
            dr.add(String(x + w / 2, 3.5 * cm - k * 0.4 * cm, line, fontName="Vera", fontSize=6.8,
                          textAnchor="middle"))
        if i < len(boxes) - 1:
            y, x1, x2 = 2.7 * cm, x + w + 0.05 * cm, x + w + gap - 0.08 * cm
            dr.add(Line(x1, y, x2, y, strokeWidth=1))
            dr.add(Polygon([x2, y, x2 - 4, y + 3, x2 - 4, y - 3], fillColor=colors.black, strokeWidth=0))
    st += [dr, P(f"A {DISTRACTOR} is a true fact of the world that a paper states beside one of its claims: the same "
                 "relation with one entity different. Stating it is planned. A conflict is two claims of one paper "
                 "with different subjects, the same object and opposite relations. The writer model never makes a "
                 "number: every number comes from quant.py.", body)]

    # ---- counts
    st += [P("Counts per domain", h2)]
    kinds = ["relation claims", "distractors", "conflict facts", "attr", "effect", "pool"]
    rows = [["domain", "papers", "relation claims", "distractors (true near-miss facts)", "of which conflict", "attr numbers",
             "effect numbers", "pool numbers", "all fact rows"]]
    for d in DOMAINS:
        c = D[d]["counts"]
        rows.append([FIELD[d], c["papers"]] + [c["facts"].get(k, 0) for k in kinds] + [c["facts_total"]])
    st.append(table(rows))
    st.append(P("'Of which conflict': relation facts (claims) flagged as part of a planted conflict.", small))
    qtypes = sorted({t for d in DOMAINS for t in D[d]["counts"]["questions"]})
    rows = [["domain"] + qtypes + ["all"]]
    for d in DOMAINS:
        q = D[d]["counts"]["questions"]
        rows.append([FIELD[d]] + [q.get(t, 0) for t in qtypes] + [sum(q.values())])
    st += [Spacer(1, 4), P("Questions per type", h3), table(rows), src("seed/bench_*/facts.jsonl",
                                                                     "seed/bench_*/gold/*.json",
                                                                     "seed/bench_*/paperfactory.json")]

    # ---- metric 1
    st += [PageBreak(), P("Metric 1: fact recall", h2),
           P("Planted facts that have a place (span) in their paper, out of planted facts. Relation facts were placed "
             "by the tag step; number facts by code rules, and by a Haiku verifier where code could not decide "
             f"({SAME}). Facts whose number is in the paper but whose place could not be proven are left out of "
             "the denominator ('position unknown').", body)]
    rows = [["domain", "fact recall", "relation claims", "distractors (true near-miss facts)", "attr numbers", "effect numbers",
             "pool numbers", "position unknown"]]
    for d in DOMAINS:
        fr = D[d]["fact_recall"]
        bk = fr["by_kind"]

        def kc(k: str) -> str:
            x = bk.get(k)
            return "-" if not x else f"{pct(x['recall'])} {ci(x['ci95'])} ({x['placed']}/{x['planted'] - x['position_unknown']})"
        rows.append([FIELD[d], f"{pct(fr['recall'])} {ci(fr['ci95'])}", kc("claim"), kc("decoy"), kc("attr"),
                     kc("effect"), kc("pool"), fr["position_unknown"]])
    st += [table(rows), P("Wilson 95% intervals. All facts still missing a place are relation facts. Code error "
                          f"rate of the number spans: {D['bench_law']['code_error']['rate']} "
                          f"({D['bench_law']['code_error']['errors']} of {D['bench_law']['code_error']['n']}, fresh "
                          f"random sample, judged by {HAIKU}: {SAME}).", small),
           src("seed/bench_*/numspans.json (fact_recall, code_error_rate_quote)")]

    # ---- metric 2
    st += [P("Metric 2: writer precision (with reader recall)", h2),
           P("A blind reader that never sees the planted facts reads every section of 50 sampled papers per domain "
             "and writes down each statement about named things. Code sorts each one: planned (a planted claim, "
             f"{DISTRACTOR}, conflict or seed number of that paper) or possible invented. Writer precision = planned "
             "/ (planned + possible invented). Reader recall: planted facts with a place that the reader found.",
             body)]
    rows = [["domain", "papers read", "writer precision, lower bound", "as matched by code", "matching misses",
             "reader recall"]]
    for d in DOMAINS:
        x = D[d]
        rows.append([FIELD[d], x["rb_papers"], f"{pct(x['writer_precision_lb']['rate'])} "
                     f"{ci(x['writer_precision_lb']['ci95_exact'])}",
                     f"{pct(x['writer_precision']['rate'])} {ci(x['writer_precision']['ci95_exact'])}",
                     x["matching_misses"], f"{pct(x['reader_recall']['rate'])} {ci(x['reader_recall']['ci95_exact'])}"])
    rows.append(["all", sum(D[d]["rb_papers"] for d in DOMAINS), f"{pct(wp['rate'])} {ci(wp['ci95_exact'])}",
                 f"{pct(pr['writer_precision']['rate'])} {ci(pr['writer_precision']['ci95_exact'])}",
                 pr["matching_misses"], f"{pct(pr['reader_recall']['rate'])} {ci(pr['reader_recall']['ci95_exact'])}"])
    st.append(table(rows))
    bw = pr["by_writer"]
    st += [P("Exact (Clopper-Pearson) 95% intervals. Reader: " + E(D["bench_law"]["reader_model"]) + ". "
             "<b>Lower bound:</b> a matching miss is a planted fact the reader wrote down in other words that code "
             "did not match (for example 'Satho Lab 2018 study reports_improvement thermal conductivity' for the "
             f"{DISTRACTOR} 'Satho Lab 2018 study reports_improvement coating Kogal-8'). Code finds such half "
             "matches and counts them as planned; free-text misses are found only by hand, so the true precision "
             "is at least this.", small),
           P(f"Hand check ({E(hc['labeller'])}): {E(hc['planned_misses_count'])} possible invented facts were planned "
             f"misses; the code rule found {E(hc['rule_found'])}. None contradicts the seed. Main cause: "
             f"{E(notes['main_cause_of_matching_misses'])}.", small),
           P("By writer model (as matched by code): " + "; ".join(
               f"{E(w)}: precision {pct(v['writer_precision']['rate'])} {ci(v['writer_precision']['ci95_exact'])}, "
               f"reader recall {pct(v['reader_recall']['rate'])} {ci(v['reader_recall']['ci95_exact'])}"
               for w, v in bw.items()) + ". Law is lowest: its writers add their own arguments (ranks of the "
             "legal bodies, readings of the cases).", small),
           src("seed/bench_*/readback.json (summary, matching_misses)", "seed/bench_readback_notes.json",
               "seed/bench_readback_handcheck_seedcheck.csv")]

    # ---- metric 3
    st += [PageBreak(), P("Metric 3: realism (real vs synthetic review papers)", h2),
           P("Per body paragraph, after names, citations and numbers are masked and 'invented world' sentences are "
             "dropped: sentence length, hedging words, citations and numbers per 100 words (two-sample "
             "Kolmogorov-Smirnov D), and the share of each fact kind from the blind reader (chi-square, Cramér's V). "
             "Effect size first; Holm correction over the 5 tests. Paragraph-level p-values are optimistic "
             "(paragraphs of one paper are not independent).", body)]
    names = {"sent_len": "sentence length (words)", "hedges": "hedges / 100 words", "cites": "citations / 100 words",
             "numbers": "numbers / 100 words", "fact_kind_share": "fact kinds (Haiku-judged)"}
    for d in DOMAINS:
        rc = D[d]["realism"]
        small_s = "" if rc["status"] == "ok" else " — small sample"
        rows = [["property", "real mean", "synthetic mean", "KS D / Cramér's V", "p (Holm)", "n papers real / synth",
                 "n paragraphs or facts"]]
        kn = rc.get("fact_kinds", {}).get("sample", {}).get("n", "?")
        for t in rc["tests"]:
            if t["property"] == "fact_kind_share":
                rm = ", ".join(f"{k.replace('add_', '')} {v:.2f}" for k, v in t["real_mean"].items())
                sm = ", ".join(f"{k.replace('add_', '')} {v:.2f}" for k, v in t["synth_mean"].items())
                rows.append([names[t["property"]], rm, sm, f"V = {t['effect_size']:.2f}", f"{t['p_holm']:.1e}",
                             f"{kn} / {kn}", f"{t['n_real']} / {t['n_synth']} facts"])
            else:
                rows.append([names[t["property"]], f"{t['real_mean']:.2f}", f"{t['synth_mean']:.2f}",
                             f"D = {t['effect_size']:.2f}", f"{t['p_holm']:.1e}",
                             f"{rc['real_papers']} / {rc['synth_papers']}", f"{t['n_real']} / {t['n_synth']}"])
        st.append(KeepTogether([P(f"{FIELD[d]}: n = {rc['real_papers']} real, {rc['synth_papers']} synthetic "
                                  f"papers{small_s}", h3),
                                table(rows, [3.3 * cm, 3.3 * cm, 3.3 * cm, 2 * cm, 1.6 * cm, 1.8 * cm, 2 * cm])]))
    big_d, big = max(((FIELD[d], t) for d in DOMAINS for t in D[d]["realism"]["tests"]),
                     key=lambda x: x[1]["effect_size"])
    law_n = D["bench_law"]["realism"]["real_papers"]
    st += [P("Fact kinds: a fixed sample of real and synthetic papers per domain (n in the tables) read by Haiku; real papers: Haiku-judged, no seed "
             f"to check; synthetic papers written by Haiku: {SAME}. Largest effect: "
             f"{names[big['property']]} in {big_d} (D = {big['effect_size']:.2f}). Law: {law_n} real papers "
             "(small sample).", small),
           src("seed/bench_*/realism_compare.json", "seed/realism_report.md", "seed/realism_selection.json",
               "seed/realism_law_licences.json")]
    inv_ex = _j(data / "seed" / "bench_law" / "invented_world_sentences.json")["items"][0]["sentence"]
    rows = [["domain", "'invented world' sentences in synthetic papers", "papers with one or more"]]
    for d in DOMAINS:
        rows.append([FIELD[d], D[d]["invented_world"], f"{D[d]['invented_papers']} of {D[d]['invented_of']}"])
    st += [P("Giveaway: papers saying their world is invented", h3),
           P(f"For example (law): '{E(inv_ex)}'. Such sentences are dropped before the realism check; the writer "
             "prompts still need a fix.", body), table(rows),
           src("seed/bench_*/realism_synth.json (invented_world_sentences)", "seed/bench_*/invented_world_sentences.json")]

    # ---- metric 4
    st += [PageBreak(), P("Metric 4: shortcut rate (closed book)", h2),
           P("Each gold question asked with no papers. The world is invented, so the right-answer rate should be near "
             "the guessing rate. For trap questions (the answer is 'not reported') the guessing rate is the model's "
             "own 'not reported' rate on answerable questions.", body)]
    for d in DOMAINS:
        rows = [["type", "n", "right", "rate", "guessing rate", "above guessing?"]]
        for r in D[d]["shortcut"]:
            g_ = r.get("guess")
            gs = "-" if g_ is None or g_ != g_ else f"{g_:.2f}"
            rows.append([r["type"], r["n"], r["right"], f"{pct(r['rate'])} {ci(r['ci95'])}", gs,
                         "yes" if r.get("above_guess") else "no"])
        st.append(KeepTogether([P(f"{FIELD[d]} (answerer {HAIKU}; {SAME})", h3),
                                table(rows, [3 * cm, 1.2 * cm, 1.2 * cm, 4.5 * cm, 2.5 * cm, 2.5 * cm])]))
    law_lk = next(r for r in D["bench_law"]["shortcut"] if r["type"] == "lookup")
    st += [P("Exact 95% intervals. Qwen (orch-model) could not be tested: its gateway was down. Law questions now name "
             "the deciding body the same way in trap and normal questions (before, that wording gave traps away). Law lookups "
             f"right: {law_lk['right']} of {law_lk['n']} (from general knowledge of legal ranks).", small),
           src("seed/bench_*/shortcut.json (summary)")]

    # ---- sample papers
    st += [PageBreak(), P("Sample papers with marks", h2)]
    legend = [["mark", "meaning"]] + [[P(f"<font backColor='{COLORS[k]}'>&nbsp;example text&nbsp;</font>", cell), m]
                                      for k, m in (("claim", "planted claim (true)"),
                                                   ("decoy", f"{DISTRACTOR} (true; shown red as the spec asks)"),
                                                   ("conflict", "claim in a planted conflict"),
                                                   ("number", "planted number (made by code)"),
                                                   ("invented", "possible invented fact (blind reader, Part C)"))]
    st += [table(legend, [4 * cm, 11 * cm]),
           P("Marks come from the spans in paperfactory.json and the read-back claims; where marks overlap the later "
             "row in the list wins. Purple marks only exist for papers in the read-back sample (two per domain "
             "here, the shortest, one per writer model where possible). Under each paper: its planted facts with "
             "no position.", small)]
    first = True
    for d in DOMAINS:
        for pid in pick_samples(data, d):
            md, mark, nk, nopos = marks_for(data, d, pid)
            man = _j(data / "seed" / d / "paperfactory.json")
            p = next(x for x in man["papers"] if x["id"] == pid)
            st += [PageBreak() if not first else Spacer(1, 8), P(f"{FIELD[d]}: {E(pid)} — {E(p['title'])}", h3),
                   P(f"Writer: {E(p.get('writer', '?'))}. Marks: " + ", ".join(
                       f"{MARK_NAME[k]} {v}" for k, v in nk.items()) +
                     f". Source: data/dataset/{d}/{E(pid)}/paper.md, seed/{d}/paperfactory.json, "
                     f"seed/{d}/readback.json", small)]
            pos = 0
            for block in md.split("\n\n"):
                a, b = pos, pos + len(block)
                pos = b + 2
                if not block.strip():
                    continue
                lines = block.split("\n")
                is_tab = all(x.lstrip().startswith("|") for x in lines if x.strip())
                strip = len(block) - len(block.lstrip("#")) if block.startswith("#") else 0
                strip += len(block[strip:]) - len(block[strip:].lstrip()) if strip else 0
                out_s, cur, buf = [], None, []

                def flush() -> None:
                    if buf:
                        t = "".join(buf)
                        if is_tab:                       # Courier (built in) has no curly quotes or dashes
                            t = t.translate(ASCII)
                        t = E(t).replace("\n", "<br/>")
                        out_s.append(f"<font backColor='{COLORS[cur]}'>{t}</font>" if cur else t)
                        buf.clear()
                for j in range(a + strip, b):
                    if mark[j] != cur:
                        flush()
                        cur = mark[j]
                    buf.append(md[j])
                flush()
                style = paper_tab if is_tab else (paper_h if strip else paper_txt)
                st.append(P("".join(out_s), style))
            first = False
            st.append(P("Planted facts with no position: " + ("; ".join(E(x) for x in nopos) if nopos else "none") +
                        ".", small))

    # ---- limits
    st += [PageBreak(), P("Limits and what is not finished", h2)]
    lim = [
        "Writer precision is a lower bound: free-text matching misses are found only by hand, and the hand check "
        f"covers {g['hand_n']} items. The hand-check labels were made by Claude, not a human.",
        f"Reader errors: reader recall is {pct(pr['reader_recall']['rate'])}; a planted fact the reader missed cannot "
        "count for or against the writer. Reader misses are mostly setup numbers in wide tables.",
        f"Same model: Haiku judged the shortcut answers, the fact kinds and part of the number spans, and wrote "
        f"{sum(D[d]['counts']['writers'].get(HAIKU, 0) for d in DOMAINS)} of the papers ({SAME}).",
        "Real-paper sample (candidates / kept / measured): " + "; ".join(
            f"{f} {v['candidates']} / {v['kept_n']} / {D['bench_' + f]['realism']['real_papers']}"
            for f, v in g["selection"].items()) + f". Law licences confirmed on publisher pages: "
        f"{g['law_licences']['confirmed']}; law is a small sample. Measured < kept where no full text could be "
        "fetched or converted. Real-paper fact kinds have no seed to check the reader against.",
        "Realism p-values treat paragraphs as independent, so they are optimistic; the effect sizes are the result.",
        "Shortcut test: only Claude Haiku; Qwen was not reachable.",
        "Part B: the 20 number spans to be checked by eye are " + ("checked." if g["eye_check_done"] else
                                                                   "NOT checked yet."),
        "Marks on the sample papers have not yet been spot-checked by eye (one paper per domain), as the spec asks.",
        "Intervals are the exact (Clopper-Pearson) or Wilson intervals stored in the output files, not paper-level "
        "bootstrap intervals.",
        "Not measured here: question difficulty, the style-leak test (it belongs to distractor difficulty), the QA "
        "systems.",
    ]
    for x in lim:
        st.append(P("• " + E(x), body))
    st.append(Spacer(1, 8))
    st.append(src("seed/bench_readback_notes.json", "seed/realism_selection.json", "seed/bench_*/shortcut.json"))

    def footer(c, doc):
        c.saveState()
        c.setFont("Vera", 7)
        c.drawString(2 * cm, 1.2 * cm, "Benchmark data quality — Parts B-E — 27 Nov 2026")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
        c.restoreState()
    out.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                      bottomMargin=1.8 * cm, title="Benchmark data quality", author="thesis-research-agent"
                      ).build(st, onFirstPage=footer, onLaterPages=footer)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m paperfactory.report", description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path("../../data"))
    ap.add_argument("--out", type=Path, default=Path("../../data/seed/report_partF.pdf"))
    a = ap.parse_args(argv)
    g = gather(a.data)
    build_pdf(g, a.data, a.out)
    print(a.out)


if __name__ == "__main__":
    main()
