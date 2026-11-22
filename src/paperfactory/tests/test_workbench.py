"""export-workbench (paperfactory/workbench.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperfactory import workbench
from paperfactory.text import normalize_paper

MD = "# A Study of Things\n\n## Abstract\n\n**Kor v. Bel** relied on *section 1*. Nothing else here.\n"


def _run(tmp: Path) -> Path:
    run = tmp / "run"
    (run / "work" / "P-01").mkdir(parents=True)
    (run / "work" / "P-01" / "paper.md").write_text(MD, encoding="utf-8")
    (run / "work" / "P-02").mkdir(parents=True)   # not in tagged.json: failed the check, never copied
    (run / "papers" / "P-01").mkdir(parents=True)
    for b in workbench.BRIEFS:
        (run / "papers" / "P-01" / b).write_text(f"brief {b}", encoding="utf-8")
    (run / "seed.ttl").write_text("# seed\n", encoding="utf-8")
    ents = {"D1": {"label": "Kor v. Bel"}, "D2": {"label": "Ab v. Cd"}, "P1": {"label": "section 1"}}
    papers = {"P-01": {"title_hint": "x", "claims": [["D1", "relies_on", "P1"]], "decoys": [["D2", "relies_on", "P1"]]},
              "P-02": {"title_hint": "y", "claims": [], "decoys": []}}
    (run / "world.json").write_text(json.dumps({"field": "law", "genre": "argument", "entities": ents,
                                                "papers": papers}), encoding="utf-8")
    norm = normalize_paper(MD).text
    a = norm.index("Kor v. Bel")
    b = norm.index("section 1.") + len("section 1.")
    tagged = {"papers": {"P-01": {"spans": {"D1-relies_on-P1": [[a, b]]}, "sentences": [[a, b, norm[a:b]]]}}}
    (run / "tagged.json").write_text(json.dumps(tagged), encoding="utf-8")
    return run


def test_export(tmp_path: Path) -> None:
    """The files the Workbench reads (its view of them is tested in the parent repo, backend/tests/test_pfview.py)."""
    data = tmp_path / "data"
    assert workbench.export(_run(tmp_path), "pf_law", data) == {"domain": "pf_law", "papers": 1, "planned": 2}
    m = json.loads((data / "seed" / "pf_law" / "paperfactory.json").read_text())
    [p] = m["papers"]
    assert p["title"] == "A Study of Things" and [f["kind"] for f in p["facts"]] == ["claim", "decoy"]
    md = (data / "dataset" / "pf_law" / "P-01" / "paper.md").read_text()
    a, b = p["facts"][0]["spans"][0]
    assert md[a:b] == "Kor v. Bel** relied on *section 1*."          # raw Markdown offsets
    assert p["facts"][1]["spans"] == []
    assert not (data / "dataset" / "pf_law" / "P-02").exists()       # failed the check: never copied
    assert all((data / "seed" / "pf_law" / "papers" / "P-01" / b_).is_file() for b_ in workbench.BRIEFS)


def test_never_overwrites_an_evqa_domain(tmp_path: Path) -> None:
    (tmp_path / "data" / "seed" / "materials").mkdir(parents=True)
    with pytest.raises(ValueError, match="not a paperfactory export"):
        workbench.export(_run(tmp_path), "materials", tmp_path / "data")
