# For code agents: what exists, where, and how to build on it

This repo is the data-generation service of the thesis (synthetic papers with planted facts). It normally sits as a
git submodule at `services/paper-factory` inside the thesis repo (`thesis-research-agent`). Paths below written as
`../../X` are in that parent repo. Most of the local data is **not in git** (size or licence); if a path is missing,
you are in a fresh clone and must fetch it again (see each entry).

## Rules first

- Never print, log or commit secrets: `OPENALEX_API_KEY` (in `.env`), anything in `../../llm-setup/.env`
  (`LITELLM_MASTER_KEY`, `OPENROUTER_API_KEY`). Read them from the environment only.
- `../../llm-setup/` is the admin-approved LLM serving setup: read-only.
- Prompt changes: follow `../../docs/prompt-writing.md` (atomic tasks, examples, code checks, separate generate
  and verify). Version prompts; keep old versions (e.g. `system_v1`, `*_v1` genre files).
- Numbers in papers are made by code (`quant.py`), never by the writer model.

## Code map (src/)

| Path | What |
|---|---|
| `paperfactory/cli.py` | `paperfactory run --config ... --steps world,seed,materials,write,check,eval`; also `export-workbench`, `fetch-topics` |
| `paperfactory/api.py` | FastAPI service (`uv run paperfactory-api`): POST /runs, GET /runs/{id}, /runs/{id}/papers[/pid] |
| `paperfactory/world.py` | invents the world: topic clusters, entities, claims, decoys, conflicts; `check` validates it |
| `paperfactory/quant.py` | numbers per entity and per fact (effects with 95% CI, inverse-variance pooling, Q, I²) |
| `paperfactory/style.py` | style cards per paper (template, tone, voice, opening, rhythm, pages); banned phrases |
| `paperfactory/forms.py` | form cards from the donor papers (`paperfactory profile-corpus --donors ../../data/cache/donors` writes `packs/forms/<field>.json`): outlines with generic headings only, section shares, table counts, sentence length; with that file each paper gets one donor's layout as its template |
| `paperfactory/plan.py` | content plan per paper: which numbers it reports and under which headers, table designs, fact order, the section and form of each planted fact (`content_plan.json` beside the notes) |
| `paperfactory/materials.py` | the brief each writer gets (sources, findings, number rows, tables to build) |
| `paperfactory/write.py` | calls PaperOrchestra (`external/paperorchestra`, prompt set `v1nn`) |
| `paperfactory/repair.py` | checker + repair loop; final gate (PaperOrchestra `fact_check`, ≥2 tables, no leaked row ids) |
| `paperfactory/evaluate.py`, `deepeval_step.py` | tagging, leak test, QA recall |
| `paperfactory/workbench.py` | exports passed papers to `../../data/{seed,dataset}/<domain>` for the Workbench UI |
| `paperfactory/packs/` | `fields/*.json` (entity kinds, relations, number specs), `genres/` (synthesis, argument + templates), `topics/` |
| `paperfactory/configs/` | run configs; current pilot: `qwen/v2p3-{law,llm,materials,medicine}.json` |
| `modelcall/`, `jsonio/` | model-call layer (OpenAI-compatible gateway, cache, retries), JSON helpers |

More detail: `src/paperfactory/README.md`, `RUNBOOK-qwen.md`, `TODO.md`. Design notes: `docs/redesign-v3-brief.md`; old log: `../../docs/archive/codex-log.md`.

## Local data you can use

| What | Where | Notes |
|---|---|---|
| **500 real donor papers** (Markdown) | `../../data/cache/donors/<field>/<id>.md` | 125 each: law, llm, medicine, materials; 406 MB with raw files. Open licences only (389 CC-BY, 54 CC-BY-NC, 25 public domain, 19 CC-BY-NC-SA, 10 CC-BY-SA, 3 CC0). For **form only** (outlines, section lengths, table shapes, citation style, number ranges): never show them to a writer model and never turn them into benchmark items. Not used by the pipeline yet. |
| donor manifest | `../../data/cache/donors/manifest.jsonl` | one line per paper: id, field, title, year, source, source_url, licence, licence_url, paper_type, topic, words, n_tables, n_headings, n_references, format_from, doi, openalex_id, ... |
| donor provenance | `../../data/cache/donors/README.md`, `rejected.jsonl`, `_raw/`, `_cand/`, `_scripts/` | how they were picked (OpenAlex sample per topic), licence checks, converters (JATS→MD, arXiv LaTeXML HTML→MD). Re-run with `_scripts/` (own venv). |
| topic packs | `src/paperfactory/packs/topics/{medicine,llm,materials,law}.json`, `ranges.json` | OpenAlex topics (60/10/40/15) and number ranges (e.g. ClinicalTrials.gov phase-3 enrolment p25=85, p90=1011). In git. Refresh: `paperfactory fetch-topics`. |
| pilot runs (v2, 3 papers per field) | `../../runs/paperfactory/v2p3-<field>/` (+ `.log`, `.world-check.txt`) | world.json, materials, `work/<pid>/paper.md`, check records, eval. Results: law 2/3 pass, others 3/3. |
| older runs | `../../runs/paperfactory/qwen40-<field>/`, `qwen-*`, `qwen-smoke-*` | 40-paper v1 pilot and smoke runs |
| Workbench export | `../../data/seed/<domain>/`, `../../data/dataset/<domain>/<pid>/paper.md` | domains `pf_*` (v1 passed papers) and `v2_*` (v2 pilot). UI: `../../` webapp, `/workbench`. |
| human review labels | `../../data/reviews/{fidelity,population}.jsonl` | Workbench review choices (e.g. fact supported or not) for the older evqa `medical` domain |
| PaperOrchestra cache | `../../data/cache/paperorchestra/` | call cache |
| runs started by the API | `runs/` (this repo), registry `runs/_service/runs.json` | not in git |

## External APIs (all reachable from the compute nodes)

| API | Used for | Key |
|---|---|---|
| OpenAlex (`api.openalex.org`) | topics, donor candidates | `OPENALEX_API_KEY` in `.env` |
| ClinicalTrials.gov v2 (`clinicaltrials.gov/api/v2`) | enrolment ranges | none |
| NCBI efetch / PMC OA, Europe PMC | donor full texts (JATS XML) | none |
| arXiv (OAI-PMH, `arxiv.org/html`, ar5iv) | llm donors, licences | none |
| Caselaw Access Project (`static.case.law`) | law donors (U.S. opinions) | none |

## Models

- **Qwen (local)**: vLLM on a Slurm GPU job, behind the LiteLLM gateway as `orch-model` (setup: `../../llm-setup/`,
  README there). Run inside the job: `srun --jobid <job> --overlap ops/injob.sh uv run paperfactory run --config ...`.
  `ops/injob.sh` loads `LLM_ENV_FILE` (default `../../llm-setup/.env`). Check the job with `squeue -u $USER`.
- **Claude Haiku**: approved for small jobs (blind test, restyle); not wired in yet.

## Open work (Nov 2026)

1. "Primary report" variant: the paper's own study is a named world study flagged `self` (name in title, abstract,
   tables; tagger maps "we/this study" to it). Estimated 2–3 days.
2. Style leak: writers copy the log's relation words. 3. LAW thesis can contradict the world.
4. Use the donor papers for form (section plans, table shapes). 5. Haiku blind test and restyle.

## Check before you finish

    uv sync && uv run pytest -q          # 238 pass, 1 skip (no model calls)
    uv run ruff check src
