"""One Workbench domain per field from every paperfactory run of that field (24 Nov 2026).

Sources (all kept where they are; nothing is deleted):
  h   the benchmark run (Claude Haiku, joined world worlds/bench20-<field>.json)   paper ids as in the world
  v1  the 40-paper Qwen pilot (runs/paperfactory/qwen40-<field>)                   "v1-<id>"
  v2  the 3-paper v2 pilot (runs/paperfactory/v2p3-<field>)                        "v2-<joined id>"
  v3  the v3 baseline rerun (runs/paperfactory/v3baseline/v2p3-<field>)           "v3-<joined id>"

The joined world (worlds/bench-<field>.json) was made from v1 + v2 by join.py, which recorded what it changed
(w["joins"]): v1 keeps its ids and names; v2 entities that are the same thing take v1's id, and v2's accidental
name clashes got fresh words. v3 is a world of its own, so it is joined here the same way (join.join). Every v2 and v3
paper, brief and fact label gets those word swaps, and the fact spans move with the text.

Papers may disagree (v2's numbers were not remade, several plans are written twice): kept on purpose, an evidence
system has to deal with disagreement.

  uv run python ops/merge_bench.py law [llm materials medicine] [--data-dir ../../data] [--prefix bench_]
"""
from __future__ import annotations

import argparse
import bisect
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from paperfactory import join, packs, workbench

PF = Path(__file__).resolve().parents[1]
RUNS = {"v1": PF / "../../runs/paperfactory/qwen40-{f}", "v2": PF / "../../runs/paperfactory/v2p3-{f}",
        "v3": PF / "runs/paperfactory/v3baseline/v2p3-{f}", "h": PF / "runs/paperfactory/bench20h-{f}"}
BRIEFS = workbench.BRIEFS


def renamer(swap: dict[str, str]) -> Callable[[str], tuple[str, Callable[[int], int]]]:
    """text -> (renamed text, old offset -> new offset)."""
    def run(text: str) -> tuple[str, Callable[[int], int]]:
        out, ends, deltas, last, delta = [], [], [], 0, 0
        for m in join.WORD.finditer(text):
            w = m.group(0)
            if w.lower() in swap:
                new = join._recase(swap[w.lower()], w)
                out += [text[last:m.start()], new]
                last = m.end()
                delta += len(new) - len(w)
                ends.append(m.end())
                deltas.append(delta)
        out.append(text[last:])

        def move(o: int) -> int:
            i = bisect.bisect_right(ends, o)
            return o + (deltas[i - 1] if i else 0)
        return "".join(out), move
    return run


def mapping(tag: str, field: str, bench: dict[str, Any]) -> tuple[dict, dict, dict, dict[str, Any]]:
    """(entity id map, word swaps, paper id map, entities to add) of one source into the joined world."""
    if tag in ("h", "v1"):
        return {}, {}, {}, {}
    if tag == "v2":
        j, pre = bench["joins"], "v2p3:"
        strip = lambda d: {k[len(pre):]: v for k, v in d.items() if k.startswith(pre)}  # noqa: E731
        return strip(j["unified"]) | strip(j["ids"]), strip(j["renamed_words"]), strip(j["papers"]), {}
    world = json.loads((Path(str(RUNS["v3"]).format(f=field)) / "world.json").read_text(encoding="utf-8"))
    base = {**bench, "joins": {"sources": [], "unified": {}, "ids": {}, "papers": {}, "renamed_words": {},
                               "clashes": []}}
    w = join.join(base, world, packs.field(field), 3, "v3")
    j, pre = w["joins"], "v3:"
    strip = lambda d: {k[len(pre):]: v for k, v in d.items() if k.startswith(pre)}  # noqa: E731
    added = {k: v for k, v in w["entities"].items() if k not in bench["entities"]}
    return strip(j["unified"]) | strip(j["ids"]), strip(j["renamed_words"]), strip(j["papers"]), added


def merge(field: str, data: Path, domain: str) -> dict[str, Any]:
    bench = json.loads((PF / f"src/paperfactory/worlds/bench-{field}.json").read_text(encoding="utf-8"))
    ents = dict(bench["entities"])
    seed_dir, ds_dir = data / "seed" / domain, data / "dataset" / domain
    if seed_dir.exists() and not (seed_dir / "paperfactory.json").is_file():
        raise SystemExit(f"{seed_dir} exists and is not a paperfactory export; pick another --prefix")
    for d in (seed_dir, ds_dir):
        if d.exists():
            shutil.rmtree(d)
    rows, papers, counts, vocab = [], [], {}, None
    with tempfile.TemporaryDirectory() as tmp:
        for tag in ("h", "v1", "v2", "v3"):
            run = Path(str(RUNS[tag]).format(f=field)).resolve()
            if not (run / "tagged.json").is_file():
                counts[tag] = "no run"
                continue
            workbench.export(run, f"{tag}_{field}", Path(tmp))
            src_seed, src_ds = Path(tmp) / "seed" / f"{tag}_{field}", Path(tmp) / "dataset" / f"{tag}_{field}"
            idmap, swap, pmap, added = mapping(tag, field, bench)
            ents |= added
            rename = renamer(swap)
            eid = lambda x: idmap.get(x, x)  # noqa: E731
            newpid = lambda p: p if tag == "h" else f"{tag}-{pmap.get(p, p)}"  # noqa: E731
            man = json.loads((src_seed / "paperfactory.json").read_text(encoding="utf-8"))
            vocab = vocab or (src_seed / "vocab.json").read_text(encoding="utf-8")
            for p in man["papers"]:
                pid = newpid(p["id"])
                md, move = rename((src_ds / p["id"] / "paper.md").read_text(encoding="utf-8"))
                (ds_dir / pid).mkdir(parents=True)
                (ds_dir / pid / "paper.md").write_text(md, encoding="utf-8")
                (seed_dir / "papers" / pid).mkdir(parents=True)
                for b in BRIEFS:
                    if (src_seed / "papers" / p["id"] / b).is_file():
                        txt = (src_seed / "papers" / p["id"] / b).read_text(encoding="utf-8")
                        (seed_dir / "papers" / pid / b).write_text(rename(txt)[0], encoding="utf-8")
                facts = []
                for fct in p["facts"]:
                    s, r, o = fct["id"].split("-", 2) if fct["id"].count("-") == 2 else (None, None, None)
                    s, o = eid(s), eid(o)
                    facts.append({**fct, "id": f"{s}-{r}-{o}" if s else fct["id"],
                                  "subject": ents[s]["label"] if s in ents else rename(fct["subject"])[0],
                                  "object": ents[o]["label"] if o in ents else rename(fct["object"])[0],
                                  "spans": [[move(a), move(b)] for a, b in fct["spans"]]})
                papers.append({**p, "id": pid, "title": rename(p["title"])[0], "issue": rename(p["issue"])[0],
                               "thesis": rename(p["thesis"])[0], "facts": facts,
                               "source": {"run": str(run), "paper": p["id"]}})
            for line in (src_seed / "facts.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                r["paper"] = newpid(r["paper"])
                for k in ("subject", "object"):
                    if k in r:
                        r[k] = eid(r[k])
                if "studies" in r:
                    r["studies"] = [eid(x) for x in r["studies"]]
                if "label" in r:
                    r["label"] = rename(r["label"])[0]
                r["id"] = (f"{r['paper']}:{r['subject']}-{r['relation']}-{r['object']}" if r["what"] == "relation"
                           else r["id"] if tag == "h" else f"{tag}-{r['id']}")
                rows.append(r)
            counts[tag] = len(man["papers"])
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{field}: duplicate fact row ids after the merge")
    missing = {x for r in rows for x in (r.get("subject"), r.get("object")) if x and x not in ents
               and r["what"] != "pool"}
    if missing:
        raise SystemExit(f"{field}: fact rows name unknown entities {sorted(missing)[:10]}")
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "facts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                          encoding="utf-8")
    (seed_dir / "entities.json").write_text(json.dumps(ents, indent=1, ensure_ascii=False), encoding="utf-8")
    (seed_dir / "vocab.json").write_text(vocab or "{}", encoding="utf-8")
    (seed_dir / "seed.ttl").write_text(ttl(rows, ents), encoding="utf-8")
    manifest = {"source": "paperfactory", "run": "merged (ops/merge_bench.py)", "field": field,
                "genre": bench.get("genre", ""), "planned": len(papers), "sources": counts, "papers": papers}
    (seed_dir / "paperfactory.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"domain": domain, "papers": len(papers), "fact_rows": len(rows), "by_source": counts}


def ttl(rows: list[dict[str, Any]], ents: dict[str, Any]) -> str:
    """The relation facts and entity labels as Turtle (the Workbench lists a domain by its seed.ttl)."""
    q = lambda s: json.dumps(s, ensure_ascii=False)  # noqa: E731
    out = ["@prefix ex: <https://example.org/paperfactory/> .",
           "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .", ""]
    for k, e in sorted(ents.items()):
        out.append(f"ex:{k} a ex:Entity ; ex:kind {q(e['kind'])} ; rdfs:label {q(e['label'])} .")
    for r in rows:
        if r["what"] == "relation":
            out.append(f"<https://example.org/paperfactory/{r['id']}> a ex:Fact ; ex:inPaper ex:{r['paper']} ; "
                       f"ex:subject ex:{r['subject']} ; ex:relation ex:{r['relation']} ; ex:object ex:{r['object']} .")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("field", nargs="+")
    ap.add_argument("--data-dir", default=str(PF / "../../data"))
    ap.add_argument("--prefix", default="bench_")
    a = ap.parse_args()
    for f in a.field:
        print(json.dumps(merge(f, Path(a.data_dir).resolve(), a.prefix + f)))
