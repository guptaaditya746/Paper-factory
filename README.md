# Paper factory

Synthetic research papers with planted facts, for evaluating question-answering systems over papers.
Code invents a world (entities, facts, numbers), PaperOrchestra writes the papers, and checks compare every
paper with the world before evaluation. Used as a submodule of the thesis repo (`services/paper-factory`).

**Code agents: read `AGENTS.md` first** (local data such as the 500 donor papers, APIs, pilot runs, rules).

- `src/paperfactory/`: the pipeline (world → seed → materials → write → check → eval), packs (fields, genres,
  topics), configs, tests. Start with `src/paperfactory/README.md` and `RUNBOOK-qwen.md`.
- `src/modelcall`, `src/jsonio`: the model-call layer (OpenAI-compatible gateway, cache, retries).
- `external/paperorchestra/`: the vendored PaperOrchestra writer (Apache-2.0, see its LICENSE and VENDORED.md).

## Use

    uv sync
    uv run paperfactory run --config src/paperfactory/configs/qwen/v2p3-law.json
    uv run paperfactory-api --port 8090          # service: POST /runs, GET /runs/{id}, GET /runs/{id}/papers
    uv run paperfactory export-workbench --config ... --domain pf_law --data-dir ../../data

On the GPU job: `srun --jobid <job> --overlap ops/injob.sh uv run paperfactory run --config ...`
(`LLM_ENV_FILE` points to the gateway's env file; default: the parent repo's `llm-setup/.env`).

## How it connects

- Writes papers and fact rows into the benchmark repo (`thesis-benchmark`, at `../../data` in the thesis repo):
  `data/seed/<domain>/` (facts) and `data/dataset/<domain>/` (papers).
- The thesis repo's `evqa seed pfbench <domain>` turns those fact rows into questions and gold; `ops/bench4.sh`
  there runs the whole four-domain build.
- Model calls go to LiteLLM (served by `thesis-llm-setup`), optionally through `model-queue`.
- Design notes: `docs/redesign-v3-brief.md` (v3 plan), `src/paperfactory/TODO.md` (open work).

## Test

    uv run pytest -q          # offline, 8 workers (pyproject.toml); no GPU or model needed

Topic lists and number ranges are fetched once (`paperfactory fetch-topics`) and checked in under
`src/paperfactory/packs/topics/`; the OpenAlex key is read from `OPENALEX_API_KEY` or `.env` and never logged.
