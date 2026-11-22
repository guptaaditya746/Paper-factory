"""The service API (paperfactory/api.py): configs, run registry, papers. No model calls: the run process is faked."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paperfactory import api


class FakeProc:
    pid = 4242

    def __init__(self, *a, **k) -> None:
        pass

    def poll(self) -> int:
        return 0


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api, "STATE", tmp_path / "state")
    monkeypatch.setattr(api.subprocess, "Popen", FakeProc)
    api._procs.clear()
    return TestClient(api.app)


def test_configs_and_unknown_config(client: TestClient) -> None:
    assert "qwen/v2p3-law.json" in client.get("/configs").json()
    assert client.post("/runs", json={"config": "../../pyproject.toml"}).status_code == 404
    assert client.post("/runs", json={"config": "qwen/v2p3-law.json", "steps": ["bogus"]}).status_code == 422


def test_run_lifecycle_and_papers(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    (out / "work" / "LAW-01").mkdir(parents=True)
    (out / "work" / "LAW-01" / "paper.md").write_text("# A Title\n\nText.\n", encoding="utf-8")
    (out / "work" / "LAW-01" / "paper.repairs.json").write_text(json.dumps({"status": "clean", "open_flags": []}))
    w = {"entities": {"D1": {"label": "Kor v. Bel"}, "P1": {"label": "section 1"}},
         "papers": {"LAW-01": {"title_hint": "x", "claims": [["D1", "relies_on", "P1"]], "decoys": [],
                               "cluster": "A"}}}
    (out / "world.json").write_text(json.dumps(w), encoding="utf-8")
    real_load = api.config.load
    monkeypatch.setattr(api.config, "load", lambda p: type("S", (), {"out_dir": out})() if p else real_load(p))
    r = client.post("/runs", json={"config": "qwen/v2p3-law.json"}).json()
    st = client.get(f"/runs/{r['run_id']}").json()
    assert st["state"] == "finished" and st["papers_written"] == 1
    assert client.get(f"/runs/{r['run_id']}/papers").json() == [
        {"id": "LAW-01", "title": "A Title", "cluster": "A", "written": True, "check": "clean"}]
    one = client.get(f"/runs/{r['run_id']}/papers/LAW-01").json()
    assert one["facts"][0]["subject"] == "Kor v. Bel" and one["check"]["status"] == "clean"
    assert client.get("/runs/nope").status_code == 404
