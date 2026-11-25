"""Copy a paperfactory run's checked papers into the Workbench's folders (22 Nov 2026).

The Workbench (parent repo thesis-research-agent) reads a domain from <data>/seed/<domain>/ (truth) and
<data>/dataset/<domain>/<paper>/paper.md (the paper). A paperfactory domain carries `paperfactory.json` next to its
seed.ttl; the Workbench's evqa/seed/pfview.py reads that file
instead of the evqa schema (paperfactory facts are relations between entities, not numbers, so the numbers check does
not apply). Only papers that passed the check (the ones in tagged.json) are copied.

Each fact's spans are the sentences the tagger + verifier accepted for it, moved from the normalized text the tagger
read to offsets in the original paper.md (text.Paper.back).

  uv run paperfactory export-workbench --config src/paperfactory/configs/qwen/pilot40-law.json --domain pf_law \
      --data-dir ../../data
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from repo_paths import DATA

from paperfactory.text import normalize_paper

BRIEFS = ("idea_sparse.md", "experimental_log.md")


def _title(md: str, fallback: str) -> str:
    first = next((ln for ln in md.splitlines() if ln.strip()), "")
    return first.lstrip("# ").strip() if first.startswith("#") else fallback


def _raw_spans(md: str, spans: list[list[int]]) -> list[list[int]]:
    back = normalize_paper(md).back
    return [[back[a], back[b - 1] + 1] for a, b in spans]


def vocab(field: str) -> dict[str, Any]:
    """The field's words the question builder needs: number labels and units per kind, relation words, the effect
    measure (with which side is better) and the field words."""
    from paperfactory import packs
    f = packs.field(field)
    return {"field": field, "words": f.raw["words"], "effect": f.raw.get("effect"),
            "relations": {r: {"words": d["words"], "effect": d.get("effect")} for r, d in f.relations.items()},
            "conflict_pairs": [list(x) for x in f.conflict_pairs],
            "kinds": {k: {"about": d.get("about", ""), "numbers": {a: {"label": n.get("label", a), "unit": n.get("unit", "")}
                                                                  for a, n in (d.get("numbers") or {}).items()}}
                      for k, d in f.kinds.items()}}


def _brief_rows(run: Path, pid: str) -> dict[str, tuple[str, str]]:
    """Row id -> (label, value) of the Numbers table in the writer's brief (experimental_log.md)."""
    path = run / "papers" / pid / "experimental_log.md"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[0].startswith("R-"):
            out[cells[0]] = (cells[1], cells[2])
    return out


def planned_rows(run: Path, pid: str) -> list[dict[str, Any]]:
    """The number rows the paper's writer was given, with what each one is (world ids). From numbers.json when the
    run wrote one; else rebuilt by the same code that wrote the brief (v3: plan.number_rows from the saved
    content_plan.json; v2: materials.number_rows), kept only where the rebuilt id, label and value equal the brief's
    Numbers table (runs before 25 Nov 2026 wrote no numbers.json, so their numbers never reached the benchmark)."""
    path = run / "papers" / pid / "numbers.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    w = json.loads((run / "world.json").read_text(encoding="utf-8"))
    if not w.get("numbers") or pid not in w["papers"]:
        return []
    from paperfactory import materials, packs
    from paperfactory import plan as plan_
    f = packs.field(w["field"])
    cp_path = run / "papers" / pid / "content_plan.json"
    if cp_path.is_file():
        cp = json.loads(cp_path.read_text(encoding="utf-8"))
        rows = plan_.number_rows(w, f, packs.genre(w["genre"]), pid, cp, materials._row_prefix(f, pid))[0]
    else:
        rows = materials.number_rows(w, f, pid)[0]
    brief = _brief_rows(run, pid)
    return [r for r in rows if brief.get(r["id"]) == (r["label"], r["value"]) and r.get("about")]


def number_rows(run: Path, pid: str) -> list[dict[str, Any]]:
    """A paper's number rows (planned_rows) as shared fact rows. The paper passed the number check, so each value
    is in it as written."""
    out = []
    for r in planned_rows(run, pid):
        a = r.get("about") or {}
        base = {"id": r["id"], "paper": pid, "value": r["value"], "unit": r["unit"], "label": r["label"]}
        if "entity" in a:
            out.append(base | {"what": "attr", "subject": a["entity"], "attr": a["attr"]})
        elif "fact" in a:
            s, rel, o = a["fact"]
            out.append(base | {"what": "effect", "subject": s, "relation": rel, "object": o, "stat": a["stat"]})
        elif "pool" in a:
            out.append(base | {"what": "pool", "subject": a["pool"], "stat": a["stat"], "studies": a["studies"]})
    return out


def export(run: Path, domain: str, data: Path = DATA) -> dict[str, Any]:
    if not domain.isidentifier():
        raise ValueError(f"bad domain name {domain!r} (letters, digits and _ only)")
    w = json.loads((run / "world.json").read_text(encoding="utf-8"))
    tagged = json.loads((run / "tagged.json").read_text(encoding="utf-8"))["papers"]
    ents = w["entities"]
    seed_dir, ds_dir = data / "seed" / domain, data / "dataset" / domain
    if seed_dir.exists() and not (seed_dir / "paperfactory.json").is_file():
        raise ValueError(f"{seed_dir} exists and is not a paperfactory export; pick another domain name")
    for d in (seed_dir, ds_dir):
        if d.exists():
            shutil.rmtree(d)
    papers, rows = [], []
    for pid in sorted(tagged):
        md = (run / "work" / pid / "paper.md").read_text(encoding="utf-8")
        plan, t = w["papers"][pid], tagged[pid]
        facts = []
        for kind in ("claims", "decoys"):
            for s, r, o in plan.get(kind, []):
                fid = f"{s}-{r}-{o}"
                facts.append({"id": fid, "kind": kind[:-1], "subject": ents[s]["label"], "relation": r,
                              "object": ents[o]["label"], "spans": _raw_spans(md, t["spans"].get(fid, []))})
                rows.append({"id": f"{pid}:{fid}", "paper": pid, "what": "relation", "subject": s, "relation": r,
                             "object": o, "role": kind[:-1], "conflict": [s, r, o] in (plan.get("conflict") or [])})
        rows += number_rows(run, pid)
        papers.append({"id": pid, "title": _title(md, plan.get("title_hint", pid)), "issue": plan.get("issue", ""),
                       "thesis": plan.get("thesis", ""), "facts": facts})
        (ds_dir / pid).mkdir(parents=True)
        (ds_dir / pid / "paper.md").write_text(md, encoding="utf-8")
        (seed_dir / "papers" / pid).mkdir(parents=True)
        for b in BRIEFS:
            src = run / "papers" / pid / b
            if src.is_file():
                shutil.copy2(src, seed_dir / "papers" / pid / b)
    shutil.copy2(run / "seed.ttl", seed_dir / "seed.ttl")
    # the shared fact rows (one per planted relation and per reported number) and the entities: what the benchmark's
    # questions and gold answers are computed from (evqa/seed/pfbench.py in the parent repo)
    (seed_dir / "facts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                          encoding="utf-8")
    (seed_dir / "entities.json").write_text(json.dumps(ents, indent=1, ensure_ascii=False), encoding="utf-8")
    (seed_dir / "vocab.json").write_text(json.dumps(vocab(w["field"]), indent=1, ensure_ascii=False), encoding="utf-8")
    manifest = {"source": "paperfactory", "run": str(run), "field": w["field"], "genre": w["genre"],
                "planned": len(w["papers"]), "papers": papers}
    (seed_dir / "paperfactory.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"domain": domain, "papers": len(papers), "planned": len(w["papers"])}
