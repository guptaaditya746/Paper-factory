"""Seed check of the read-back hand check: is each possible invented fact a contradiction of the seed, or an extra
statement that agrees with it?

Input: the hand-check key (labels.export_invented, e.g. data/seed/bench_readback_handcheck.key.json). For every item
of kind possible_invented, code builds the paper's seed: all fact rows of that paper (facts.jsonl) and the entities
they name or the claim names, with their attributes, written with entity names. One Haiku call per item (prompt
SEEDCHECK_SYS, version SEEDCHECK_VERSION) answers:

    label      contradicts_seed | extra_consistent | cannot_tell
    facts      ids of the seed lines it contradicts (required for contradicts_seed, else empty)
    reason     one short sentence

Code checks: label in the enum; for contradicts_seed at least one fact id, every id one of the given seed ids. A
failed answer is sent back once with its error. The judge is Haiku (Aditya, 26 Nov 2026: no Sonnet); every label
carries "judge" and, as Haiku also wrote the bench20h papers, "judge_note": "same model as writer".

These labels are a FIRST PASS, not the hand check: the reported writer precision stays the read-back figure
(readback.json) until Aditya labels the items himself. Writes the labels into the key file (field
"haiku_first_pass") and a review sheet <csv stem>_seedcheck.csv, one row per possible invented item, with the Haiku
label in its own column and an empty column "your_seed_label" (contradicts seed | extra, consistent) for Aditya.
The hand-check CSV itself stays blind (a Haiku column there would show which rows are controls).

    python -m paperfactory.seedcheck --key ../../data/seed/bench_readback_handcheck.key.json --data-dir ../../data
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from paperfactory.numspans import SAME_MODEL_NOTE, WRITERS

SEEDCHECK_VERSION = "seedcheck_v1"
JUDGE_MODEL = "claude-haiku-5-5"
ROLE = "verifier"
LABELS = ("contradicts_seed", "extra_consistent", "cannot_tell")
LABEL_TEXT = {"contradicts_seed": "contradicts seed", "extra_consistent": "extra, consistent",
              "cannot_tell": "cannot tell"}

SEEDCHECK_SYS = """Task: compare one statement from a paper with that paper's seed (the facts the paper was built
from). Say whether the statement contradicts the seed or is an extra statement that agrees with it.

Input fields:
- quote: the sentence from the paper.
- claim: the statement as a reader wrote it down from the quote.
- seed: lines "id: fact". Facts use names; "rank"/"level" style attributes order things (higher = above).

Output fields:
- label: "contradicts_seed" when the claim and a seed line cannot both be true (other relation on the same pair,
  other value for the same attribute, other direction of an effect, other order of ranks).
  "extra_consistent" when no seed line says otherwise: the claim adds a detail, sums up several seed lines, or
  states the same thing in other words.
  "cannot_tell" when the claim is too vague to compare (for example it names no thing from the seed).
- facts: for contradicts_seed, the ids of the seed lines it contradicts; else [].
- reason: one short sentence.

Examples (made-up names):
1. Normal. quote "The Velor-2 trial found that bramacet raised HbA1c." claim "Velor-2 finds_harm bramacet".
   seed "F1: Velor-2 finds_benefit bramacet" -> {"label": "contradicts_seed", "facts": ["F1"],
   "reason": "The seed says Velor-2 found a benefit of bramacet, the claim says harm."}
2. Confusing. quote "Both Velor-2 and Asto favoured bramacet." claim "the two trials finds_benefit bramacet".
   seed "F1: Velor-2 finds_benefit bramacet", "F2: Asto finds_benefit bramacet" -> {"label": "extra_consistent",
   "facts": [], "reason": "It sums up F1 and F2."}
3. Missing. quote "Velor-2 was run at 51 sites." claim "Velor-2 sites = 51". seed has Velor-2 lines but no site
   count -> {"label": "extra_consistent", "facts": [], "reason": "The seed gives no site count, so nothing disagrees."}
"""

SEEDCHECK_OUT = {"type": "object", "properties": {
    "label": {"type": "string", "enum": list(LABELS)},
    "facts": {"type": "array", "items": {"type": "string"}},
    "reason": {"type": "string"}}, "required": ["label", "facts", "reason"], "additionalProperties": False}


def _name(ents: dict[str, Any], eid: str) -> str:
    e = ents.get(eid)
    return e["label"] if e else eid


def fact_line(r: dict[str, Any], ents: dict[str, Any]) -> str:
    """One fact row in words (entity ids -> names)."""
    n = lambda k: _name(ents, r[k]) if r.get(k) else ""                     # noqa: E731
    unit = f" {r['unit']}" if r.get("unit") else ""
    if r["what"] == "relation":
        extra = (", decoy" if r.get("role") == "decoy" else "") + (", conflicting finding" if r.get("conflict") else "")
        return f"{n('subject')} {r['relation']} {n('object')}{extra}"
    if r["what"] == "attr":
        return f"{n('subject')} {r['attr']} = {r['value']}{unit}"
    if r["what"] == "effect":
        return f"{n('subject')}, {n('object')} ({r.get('relation', '')}): {r['stat']} = {r['value']}{unit}"
    if r["what"] == "pool":
        studies = ", ".join(_name(ents, s) for s in r.get("studies") or [])
        return f"pooled {n('subject')} over [{studies}]: {r['stat']} = {r['value']}{unit}"
    return json.dumps({k: v for k, v in r.items() if k not in ("id", "paper")}, ensure_ascii=False)


def seed_lines(rows: list[dict[str, Any]], ents: dict[str, Any], pid: str, ids: list[str]) -> dict[str, str]:
    """{line id: text}: every fact row of paper pid, then the attributes of each entity those rows or ids name."""
    own = [r for r in rows if r["paper"] == pid]
    out = {r["id"].split(":", 1)[-1]: fact_line(r, ents) for r in own}
    named = list(dict.fromkeys([r[k] for r in own for k in ("subject", "object") if r.get(k) in ents] +
                               [s for r in own for s in r.get("studies") or [] if s in ents] +
                               [i for i in ids if i in ents]))
    for eid in named:
        e = ents[eid]
        attrs = "; ".join(f"{k} = {v}" for k, v in (e.get("attrs") or {}).items())
        out[f"E:{eid}"] = f"{e['label']} is a {e.get('kind', 'thing')}" + (f" ({attrs})" if attrs else "")
    return out


def check_answer(out: dict[str, Any], allowed: set[str]) -> str | None:
    if out.get("label") not in LABELS:
        return f"label must be one of {list(LABELS)}"
    if out["label"] == "contradicts_seed" and not out.get("facts"):
        return "contradicts_seed needs the ids of the seed lines it contradicts"
    bad = [f for f in out.get("facts") or [] if f not in allowed]
    return f"facts must be seed line ids; not given: {bad}" if bad else None


def label_item(caller, rid: str, item: dict[str, Any], seed: dict[str, str]) -> dict[str, Any]:
    user = json.dumps({"quote": item.get("quote"), "claim": item.get("claim"),
                       "seed": [f"{k}: {v}" for k, v in seed.items()]}, ensure_ascii=False, indent=1)
    got = caller(ROLE, f"seedcheck:{rid}", SEEDCHECK_SYS, user, SEEDCHECK_OUT,
                 lambda o: check_answer(o, set(seed)))
    model = caller.s.model(ROLE)
    res = {"version": SEEDCHECK_VERSION, "status": "first pass, not the hand check", "judge": model}
    if model in {m for _, m in WRITERS}:
        res["judge_note"] = SAME_MODEL_NOTE
    if got is None:
        return res | {"label": None, "error": "no valid answer"}
    return res | {"label": got["label"], "label_text": LABEL_TEXT[got["label"]], "facts": got["facts"],
                  "facts_text": [seed[f] for f in got["facts"]], "reason": got["reason"]}


def run(key_path: Path, data_dir: Path, caller) -> dict[str, Any]:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    cache: dict[str, tuple[list, dict]] = {}
    counts: dict[str, int] = {}
    for rid, it in key.items():
        if it["kind"] != "possible_invented":
            continue
        d = it["domain"]
        if d not in cache:
            seed = data_dir / "seed" / d
            rows = [json.loads(x) for x in (seed / "facts.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            cache[d] = (rows, json.loads((seed / "entities.json").read_text(encoding="utf-8")))
        rows, ents = cache[d]
        ids = [i for v in (it.get("ids") or {}).values() for i in (v or []) if i]
        it.pop("seed_check", None)
        it["haiku_first_pass"] = label_item(caller, rid, it, seed_lines(rows, ents, it["paper"], ids))
        lab = it["haiku_first_pass"].get("label") or "failed"
        counts[lab] = counts.get(lab, 0) + 1
    key_path.write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    sheet = key_path.with_name(key_path.name.replace(".key.json", "_seedcheck.csv"))
    with sheet.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["id", "field", "paper", "writer", "sentence", "claim", "haiku_first_pass",
                                            "contradicted_seed_lines", "reason", "judge", "judge_note",
                                            "your_seed_label"])
        wr.writeheader()
        for rid, it in key.items():
            sc = it.get("haiku_first_pass")
            if not sc:
                continue
            wr.writerow({"id": rid, "field": it["field"], "paper": it["paper"], "writer": it["writer"],
                         "sentence": it.get("quote"), "claim": json.dumps(it.get("claim"), ensure_ascii=False),
                         "haiku_first_pass": sc.get("label_text") or "", "contradicted_seed_lines":
                         " | ".join(sc.get("facts_text") or []), "reason": sc.get("reason") or "",
                         "judge": sc["judge"], "judge_note": sc.get("judge_note", ""), "your_seed_label": ""})
    return {"counts": counts, "key": str(key_path), "sheet": str(sheet), "cost": caller.cost("seedcheck")}


def caller_for(out: str = "runs/paperfactory/seedcheck"):
    from paperfactory.config import Settings
    from paperfactory.llm import Caller
    return Caller(Settings(field="medicine", genre="argument", models={ROLE: {"model": JUDGE_MODEL, "max_parallel": 8}}, out=out))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m paperfactory.seedcheck", description=__doc__.split("\n")[0])
    ap.add_argument("--key", required=True)
    ap.add_argument("--data-dir", default="../../data")
    a = ap.parse_args(argv)
    print(json.dumps(run(Path(a.key), Path(a.data_dir), caller_for()), indent=1))


if __name__ == "__main__":
    main()
