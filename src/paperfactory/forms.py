"""Form cards: the layout and measured style of real papers, without their words (23 Nov 2026).

Why: with three hand-written templates per field, 40 papers still share three layouts. The 500 donor papers
(data/cache/donors/<field>/<id>.md + manifest.jsonl, see AGENTS.md) have 125 real layouts per field. A form card
keeps a donor's FORM only, measured by code, no model:

  outline    its level-2 and level-3 headings in order. A heading is kept as text only when every word of it is in
             the generic list (packs/forms/generic_headings.txt); any other heading is "topical": the card keeps its
             place and level, never its words. So a card cannot leak the donor's subject by construction.
  share      each level-2 section's share of the paper's words
  tables     how many, and in which section
  sentences  mean length in words; share of sentences with "we" / "our"
  words      length of the paper

`paperfactory profile-corpus` writes packs/forms/<field>.json (checked in: numbers and generic headings only).
With that file, every paper of a run gets one donor's outline as its template (style.cards), each donor used once
before any is used twice. Topical headings are written by code from the paper's own entities. Without the file the
genre's hand-written templates are used, as before. Donor texts are never shown to a writer model.
"""
from __future__ import annotations

import json
import random
import re
import statistics
from pathlib import Path
from typing import Any

from paperfactory.packs import PACKS, Field, Genre

FORMS = PACKS / "forms"
# every word a kept heading may contain (data: packs/forms/generic_headings.txt)
GENERIC = {w for line in (FORMS / "generic_headings.txt").read_text(encoding="utf-8").splitlines()
           if not line.startswith("#") for w in line.split()}
DROP = re.compile(r"^(references?|bibliography|acknowledge?ments?|funding|author(s'?)? contributions?|"
                  r"supplementa\w+.*|declarations?.*|conflicts? of interests?.*|competing interests?.*|"
                  r"data (and code )?availability.*|code availability.*|appendi(x|ces).*|abbreviations|"
                  r"ethic\w*.*|footnotes?|notes|keywords?|highlights|graphical abstract|credit .*|"
                  r"author information|additional information|publisher'?s note|orcid.*|disclosures?)$", re.I)
NUMBERING = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.|[A-Z]\.|§+\s*\d+\.?)\s+")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _clean(h: str) -> str:
    h = re.sub(r"[*_`#]", "", h).strip()
    return NUMBERING.sub("", h).strip().rstrip(".:").strip()


def is_generic(heading: str) -> bool:
    words = re.findall(r"[A-Za-z]+", heading.lower())
    return bool(words) and all(w in GENERIC for w in words)


def card(md: str, meta: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The form card of one donor paper (Markdown with # headings and | tables), or None with nothing usable."""
    meta = meta or {}
    lines = md.splitlines()
    heads = [(i, len(m.group(1)), _clean(m.group(2))) for i, line in enumerate(lines)
             if (m := re.match(r"^(#{1,4})\s+(.*\S)\s*$", line))]
    if not heads:
        return None
    top = min(lv for _, lv, _ in heads)
    if sum(1 for _, lv, _ in heads if lv == top) == 1 and any(lv > top for _, lv, _ in heads):
        top += 1                                   # one top heading = the title; sections are one level below
    outline: list[dict[str, Any]] = []
    spans: list[tuple[int, int, int]] = []         # (outline index of the level-2 entry, first line, last line)
    cut = False                                    # inside a dropped section (references ...): skip its children
    for k, (i, lv, text) in enumerate(heads):
        if lv < top or lv > top + 1:
            continue
        end = next((j for j, lv2, _ in heads[k + 1:] if lv2 <= lv), len(lines))
        if lv == top:
            cut = bool(DROP.match(text)) or not text
            if cut:
                continue
            outline.append({"level": 2, "text": text if is_generic(text) else None})
            spans.append((len(outline) - 1, i + 1, end))
        elif not cut and outline and is_generic(text) and not DROP.match(text):
            outline.append({"level": 3, "text": text})      # topical subsections are left to the writer
    if not spans:
        return None
    body = [" ".join(x for x in lines[a:b] if x.strip() and not x.lstrip().startswith(("#", "|"))) for _, a, b in spans]
    counts = [len(t.split()) for t in body]
    total = sum(counts) or 1
    for (idx, a, b), n in zip(spans, counts):
        outline[idx]["share"] = round(n / total, 3)
        tables = sum(1 for j in range(a, b) if lines[j].lstrip().startswith("|")
                     and not (j > a and lines[j - 1].lstrip().startswith("|")))
        if tables:
            outline[idx]["tables"] = tables
    sents = [s for t in body for s in SENT.split(t) if 3 <= len(s.split()) <= 120]
    if len(sents) < 20:
        return None
    we = sum(1 for s in sents if re.search(r"\b(we|our)\b", s, re.I)) / len(sents)
    return {"id": meta.get("id"), "paper_type": meta.get("paper_type"), "licence": meta.get("licence"),
            "outline": outline, "words": total, "tables": sum(x.get("tables", 0) for x in outline),
            "sentence_words": round(statistics.mean(len(s.split()) for s in sents), 1), "we_share": round(we, 3)}


def usable(c: dict[str, Any]) -> str | None:
    """Why a card cannot be a template (None = it can). A layout needs an opening, a body and an end."""
    l2 = [x for x in c["outline"] if x["level"] == 2 and (x["text"] or "").lower() != "abstract"]
    if len(l2) < 3:
        return f"{len(l2)} sections besides the abstract"
    if len(l2) > 12:
        return f"{len(l2)} top-level sections (heading levels lost in conversion?)"
    if sum(1 for x in l2 if x["text"] is None) > len(l2) * 0.75 and len(l2) > 4:
        return "almost every heading is topical"
    return None


def profile(folder: Path, manifest: Path | None = None) -> dict[str, Any]:
    """Cards of every .md file in a field's donor folder. manifest: manifest.jsonl with id, paper_type, licence."""
    meta: dict[str, dict[str, Any]] = {}
    if manifest and manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                meta[str(m.get("id"))] = m
    cards, dropped = [], {}
    for path in sorted(folder.glob("*.md")):
        c = card(path.read_text(encoding="utf-8", errors="replace"), meta.get(path.stem, {"id": path.stem}))
        why = "no headings or under 20 sentences" if c is None else usable(c)
        if why:
            dropped[path.stem] = why
        else:
            cards.append(c | {"id": path.stem})
    types: dict[str, int] = {}
    for c in cards:
        types[str(c.get("paper_type"))] = types.get(str(c.get("paper_type")), 0) + 1
    return {"cards": cards, "dropped": dropped, "paper_types": dict(sorted(types.items(), key=lambda kv: -kv[1])),
            "layouts": len({json.dumps([(x["level"], x["text"]) for x in c["outline"]]) for c in cards})}


def write(f_name: str, prof: dict[str, Any]) -> Path:
    FORMS.mkdir(parents=True, exist_ok=True)
    path = FORMS / f"{f_name}.json"
    path.write_text(json.dumps(prof, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load(f: Field, g: Genre) -> list[dict[str, Any]]:
    """The field's cards that fit the genre: genre.json "donor_types" maps a field to words, one of which must be
    in the card's paper_type (any case); no entry = every card. Empty list = no forms: use the genre's templates."""
    path = FORMS / f"{f.name}.json"
    if not path.is_file() or g.raw.get("donor_layouts") is False:
        return []
    cards = json.loads(path.read_text(encoding="utf-8"))["cards"]
    want = [x.lower() for x in g.raw.get("donor_types", {}).get(f.name, [])]
    return [c for c in cards if not want or any(x in str(c.get("paper_type", "")).lower() for x in want)]


TOPICAL = ["{O}", "Evidence on {O}", "{SP_CAP} on {O}", "The case of {O}", "{O}: what the {SP} show",
           "{O} in the {SP}", "Reading the {SP} on {O}"]
TOPICAL_NO_OBJ = ["The {SP} compared", "Where the {SP} part", "What the {SP} show"]


def template_text(c: dict[str, Any], w: dict[str, Any], f: Field, pid: str, seed: Any) -> str:
    """A paper's template.md from a donor outline. Generic headings as the donor has them; each topical level-2
    heading is written here from one of the paper's own objects (never from the donor). An abstract comes first
    whether or not the donor's Markdown had one as a section."""
    rng = random.Random(f"{seed}:outline:{f.name}:{pid}")
    p = w["papers"][pid]
    objs = list(dict.fromkeys(o for _, _, o in p["claims"] + p["decoys"]))
    rng.shuffle(objs)
    words = {"SP": f.word("subject_plural"), "SP_CAP": f.word("subject_plural").capitalize()}
    lines, used = ["# {TITLE}", "", "## Abstract", ""], {"abstract"}
    for x in c["outline"]:
        text = x["text"]
        if text is None:
            for _ in range(8):
                o = w["entities"][objs.pop(0)]["label"] if objs else None
                text = rng.choice(TOPICAL if o else TOPICAL_NO_OBJ).format(O=o, **words)
                text = text[:1].upper() + text[1:]
                if text.lower() not in used:
                    break
        if text.lower() in used:
            continue
        used.add(text.lower())
        lines += [("## " if x["level"] == 2 else "### ") + text, ""]
    return "\n".join(lines)


def style_of(c: dict[str, Any], lo: int, hi: int) -> dict[str, Any]:
    """The card's measured style as style-card values: voice, sentence length and pages (the donor's length in
    pages of the genre, kept between 4 and 8 so one paper stays one writer job)."""
    n = c["sentence_words"]
    return {"voice": 'first person plural ("we")' if c["we_share"] >= 0.08 else 'impersonal (no "we" and no "I")',
            "rhythm": f"about {round(n)} words per sentence on average, some much shorter and some longer",
            "pages": max(4, min(8, round(c["words"] / ((lo + hi) / 2))))}
