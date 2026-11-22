"""The data-generation service: start a run, follow it, read its papers (22 Nov 2026).

  uv run paperfactory-api --host 127.0.0.1 --port 8090

  GET  /health                       -> {"ok": true}
  GET  /configs                      -> the run configs in src/paperfactory/configs (relative paths)
  POST /runs {"config", "steps"?}    -> starts `paperfactory run` as its own process; {"run_id", ...}
  GET  /runs                         -> every run this service started
  GET  /runs/{run_id}                -> status (running | finished | failed), exit code, log tail, paper counts
  GET  /runs/{run_id}/papers         -> the run's papers: id, title, check status
  GET  /runs/{run_id}/papers/{pid}   -> one paper: paper.md, its planted facts, its check record

A run is the same command-line run (one process per run; the model calls go through the configured backend), so
the API adds no new pipeline logic. The service keeps its registry in runs/_service/runs.json. A config is any JSON
file under src/paperfactory/configs; the request never passes arbitrary paths or commands.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from repo_paths import REPO

from paperfactory import config

CONFIGS = Path(__file__).resolve().parent / "configs"
STATE = REPO / "runs" / "_service"
STEPS = ("world", "seed", "materials", "write", "check", "eval", "deepeval")
_lock = threading.Lock()
_procs: dict[str, subprocess.Popen] = {}

app = FastAPI(title="Paper factory", description=__doc__.split("\n\n")[0])


class RunRequest(BaseModel):
    config: str                       # relative to src/paperfactory/configs, e.g. "qwen/v2p3-law.json"
    steps: list[str] | None = None


def _registry() -> dict[str, Any]:
    f = STATE / "runs.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}


def _save(reg: dict[str, Any]) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STATE / "runs.json.tmp"
    tmp.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    tmp.replace(STATE / "runs.json")


def _config_path(name: str) -> Path:
    p = (CONFIGS / name).resolve()
    if not (p.is_file() and p.suffix == ".json" and p.is_relative_to(CONFIGS)):
        raise HTTPException(404, f"no config {name!r} under src/paperfactory/configs")
    return p


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/configs")
def configs() -> list[str]:
    return sorted(p.relative_to(CONFIGS).as_posix() for p in CONFIGS.rglob("*.json"))


@app.post("/runs")
def start(req: RunRequest) -> dict[str, Any]:
    path = _config_path(req.config)
    steps = req.steps or list(STEPS[:-1])
    if bad := set(steps) - set(STEPS):
        raise HTTPException(422, f"unknown steps {sorted(bad)}; steps: {list(STEPS)}")
    s = config.load(path)
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    STATE.mkdir(parents=True, exist_ok=True)
    log = STATE / f"{run_id}.log"
    cmd = ["uv", "run", "--project", str(REPO), "paperfactory", "run", "--config", str(path),
           "--steps", ",".join(steps)]
    with _lock:
        out = log.open("w", encoding="utf-8")
        _procs[run_id] = subprocess.Popen(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT)
        reg = _registry()
        reg[run_id] = {"config": req.config, "steps": steps, "out": str(s.out_dir), "log": str(log),
                       "started": time.time(), "pid": _procs[run_id].pid}
        _save(reg)
    return {"run_id": run_id, **reg[run_id]}


def _status(run_id: str, rec: dict[str, Any]) -> dict[str, Any]:
    p = _procs.get(run_id)
    code = p.poll() if p else rec.get("exit_code")
    if p and code is not None and "exit_code" not in rec:
        with _lock:
            reg = _registry()
            reg[run_id]["exit_code"] = code
            _save(reg)
    state = "running" if p and code is None else ("unknown" if code is None else
                                                  "finished" if code == 0 else "failed")
    log = Path(rec["log"])
    tail = log.read_text(encoding="utf-8", errors="replace")[-3000:] if log.is_file() else ""
    out = Path(rec["out"])
    papers = len(list((out / "work").glob("*/paper.md"))) if (out / "work").is_dir() else 0
    return {"run_id": run_id, **rec, "state": state, "exit_code": code, "papers_written": papers, "log_tail": tail}


@app.get("/runs")
def runs() -> list[dict[str, Any]]:
    return [_status(k, v) | {"log_tail": None} for k, v in sorted(_registry().items())]


@app.get("/runs/{run_id}")
def run(run_id: str) -> dict[str, Any]:
    rec = _registry().get(run_id)
    if rec is None:
        raise HTTPException(404, f"no run {run_id!r}")
    return _status(run_id, rec)


def _out(run_id: str) -> Path:
    rec = _registry().get(run_id)
    if rec is None:
        raise HTTPException(404, f"no run {run_id!r}")
    return Path(rec["out"])


def _check(out: Path, pid: str) -> dict[str, Any] | None:
    f = out / "work" / pid / "paper.repairs.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


@app.get("/runs/{run_id}/papers")
def papers(run_id: str) -> list[dict[str, Any]]:
    out = _out(run_id)
    wf = out / "world.json"
    if not wf.is_file():
        return []
    w = json.loads(wf.read_text(encoding="utf-8"))
    res = []
    for pid, p in w["papers"].items():
        md = out / "work" / pid / "paper.md"
        title = next((ln.lstrip("# ").strip() for ln in md.read_text(encoding="utf-8").splitlines()
                      if ln.startswith("# ")), p.get("title_hint")) if md.is_file() else p.get("title_hint")
        chk = _check(out, pid)
        res.append({"id": pid, "title": title, "cluster": p.get("cluster"), "written": md.is_file(),
                    "check": chk["status"] if chk else None})
    return res


@app.get("/runs/{run_id}/papers/{pid}")
def paper(run_id: str, pid: str) -> dict[str, Any]:
    out = _out(run_id)
    w = json.loads((out / "world.json").read_text(encoding="utf-8"))
    if pid not in w["papers"]:
        raise HTTPException(404, f"no paper {pid!r} in run {run_id!r}")
    p, ents = w["papers"][pid], w["entities"]
    md = out / "work" / pid / "paper.md"
    facts = [{"kind": role[:-1], "subject": ents[s]["label"], "relation": r, "object": ents[o]["label"],
              "numbers": w.get("numbers", {}).get("facts", {}).get(f"{s}-{r}-{o}")}
             for role in ("claims", "decoys") for s, r, o in p[role]]
    chk = _check(out, pid)
    return {"id": pid, "paper": md.read_text(encoding="utf-8") if md.is_file() else None, "facts": facts,
            "check": {k: chk[k] for k in ("status", "open_flags")} if chk else None}


def main(argv: list[str] | None = None) -> None:
    import uvicorn
    ap = argparse.ArgumentParser(prog="paperfactory-api", description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args(argv)
    uvicorn.run(app, host=a.host, port=a.port)
