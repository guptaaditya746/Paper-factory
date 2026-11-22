"""Join two worlds of one field into one (22 Nov 2026).

Why: each field had several separately invented worlds (v1 40-paper pilot, v2 3-paper pilot). The benchmark keeps one
set per field, built from all of them; nothing is deleted (the source worlds and papers stay where they are).

Rules, all in code:
- Same thing: an entity of B with the same label, kind and attributes as an entity of A is A's entity (e.g. the
  courts). B's facts then use A's id.
- Accident: the same invented word in two different entities (the name maker is seeded, so both worlds drew e.g.
  "THUGRON" for two different trials). B's word is replaced by a fresh invented word everywhere in B, so no entity's
  name sits inside another's (world.check).
- Ids and paper ids of B that clash with A get new ones; every fact, attribute reference and conflict follows.
- Numbers are dropped: the run makes them again for the whole joined world (quant.make), so all are from one seed.
- Clashes: the same subject and object with different relations in different papers. Kept as they are (an evidence
  system wants disagreement) and listed in w["joins"]["clashes"].
Everything done is recorded in w["joins"].
"""
from __future__ import annotations

import copy
import re
from typing import Any

from paperfactory import names
from paperfactory.packs import Field

WORD = re.compile(r"[A-Za-z]+")


def _words(e: dict[str, Any]) -> set[str]:
    return {w.lower() for n in [e.get("label", "")] + e.get("alt", []) for w in WORD.findall(n)}


def _recase(new: str, like: str) -> str:
    if like.isupper():
        return new.upper()
    if like.islower():
        return new.lower()
    return new.capitalize()


def _ref_attrs(f: Field, kind: str) -> list[str]:
    return [a for a, t in f.kinds.get(kind, {}).get("attrs", {}).items() if str(t).startswith("ref:")]


def _fresh(namer: names.Namer, avoid: set[str]) -> str:
    """A fresh word that is not part of, and does not contain, a word in use (e.g. never "tukrapur" next to the
    drug "tukrapurtinib"): two near-identical names are an accident a reader can confuse."""
    while True:
        new = namer.word().lower()
        if not any(new in x or x in new for x in avoid if len(x) >= 4):
            return new


def _pid_prefix(pids: list[str]) -> str:
    m = re.match(r"(.*?)(\d+)$", pids[0]) if pids else None
    return m.group(1) if m else "P-"


def join(a: dict[str, Any], b: dict[str, Any], f: Field, seed: Any, b_name: str = "b") -> dict[str, Any]:
    """A + B -> one world (A's ids and names kept). Does not change its inputs."""
    w, b = copy.deepcopy(a), copy.deepcopy(b)
    w.pop("numbers", None)
    ents_a = w["entities"]
    rec = w.setdefault("joins", {"sources": [], "unified": {}, "ids": {}, "papers": {}, "renamed_words": {},
                                 "clashes": []})
    rec["sources"].append(b_name)

    # 1. same thing: label, kind and attrs equal (ref attrs compared after mapping)
    by_label = {e["label"].lower(): eid for eid, e in ents_a.items()}
    idmap: dict[str, str] = {}
    for _ in range(3):      # refs (e.g. a decision's court) can only match once their target is mapped
        for eid, e in b["entities"].items():
            other = by_label.get(e["label"].lower())
            if eid in idmap or other is None or ents_a[other]["kind"] != e["kind"]:
                continue
            refs = _ref_attrs(f, e["kind"])
            attrs = {k: (idmap.get(v, v) if k in refs else v) for k, v in e.get("attrs", {}).items()}
            if attrs == ents_a[other].get("attrs", {}):
                idmap[eid] = other
    rec["unified"] |= {f"{b_name}:{k}": v for k, v in idmap.items()}

    # 2. accidents: invented words of B's other entities that A already uses -> fresh words
    namer = names.Namer(seed, f)
    namer.rng.random()      # a different stream from the world's own names
    kept = set().union(*(_words(b["entities"][k]) for k in idmap)) if idmap else set()
    a_words = set().union(*(_words(e) for e in ents_a.values()))
    namer.used |= a_words | set().union(*(_words(e) for e in b["entities"].values()))
    pack_words = set(re.findall(r"[a-z]+", str(f.raw).lower()))
    swap: dict[str, str] = {}
    for eid, e in b["entities"].items():
        if eid in idmap:
            continue
        for wd in _words(e):
            if wd in a_words and wd not in kept and wd not in pack_words and wd not in swap:
                swap[wd] = _fresh(namer, a_words)
    rec["renamed_words"] |= {f"{b_name}:{k}": v for k, v in swap.items()}

    def rename(text: str) -> str:
        return WORD.sub(lambda m: _recase(swap[m.group(0).lower()], m.group(0))
                        if m.group(0).lower() in swap else m.group(0), text)

    # 3. ids: B's own entities, new id on a clash
    for eid in b["entities"]:
        if eid not in idmap:
            new = eid
            while new in ents_a or new in idmap.values():
                new = "J" + new
            idmap[eid] = new
    rec["ids"] |= {f"{b_name}:{k}": v for k, v in idmap.items() if k != v}
    for eid, e in b["entities"].items():
        if idmap[eid] in ents_a:
            continue
        refs = _ref_attrs(f, e["kind"])
        ents_a[idmap[eid]] = {**e, "label": rename(e["label"]), "alt": [rename(x) for x in e.get("alt", [])],
                              "attrs": {k: (idmap.get(v, v) if k in refs else v) for k, v in e.get("attrs", {}).items()}}

    # 4. papers: facts follow the id map; a clashing paper id gets the next free number
    prefix = _pid_prefix(list(w["papers"]))
    taken = set(w["papers"])
    nums = [int(m.group(1)) for p in taken if (m := re.search(r"(\d+)$", p))]
    nxt = max(nums, default=0) + 1
    width = max((len(re.search(r"(\d+)$", p).group(1)) for p in taken if re.search(r"(\d+)$", p)), default=2)
    fact = lambda x: [idmap.get(x[0], x[0]), x[1], idmap.get(x[2], x[2])]  # noqa: E731
    for pid, p in b["papers"].items():
        new = pid
        if new in taken:
            new, nxt = f"{prefix}{nxt:0{width}d}", nxt + 1
        taken.add(new)
        rec["papers"][f"{b_name}:{pid}"] = new
        q = {**p, "claims": [fact(x) for x in p.get("claims", [])], "decoys": [fact(x) for x in p.get("decoys", [])]}
        if p.get("conflict"):
            q["conflict"] = [fact(x) for x in p["conflict"]]
        for k in ("title_hint", "issue", "thesis"):
            if isinstance(q.get(k), str):
                q[k] = rename(q[k])
        q["source"] = f"{b_name}:{pid}"
        w["papers"][new] = q
    if b.get("clusters"):
        w.setdefault("clusters", []).extend(b["clusters"])
    rec["clashes"] = clashes(w)
    return w


def clashes(w: dict[str, Any]) -> list[dict[str, Any]]:
    """Subject-object pairs that carry different relations in different papers (claims only)."""
    seen: dict[tuple[str, str], dict[str, list[str]]] = {}
    for pid, p in w["papers"].items():
        for s, r, o in p.get("claims", []):
            seen.setdefault((s, o), {}).setdefault(r, []).append(pid)
    return [{"subject": s, "object": o, "relations": rs} for (s, o), rs in seen.items() if len(rs) > 1]
