#!/usr/bin/env bash
# Run a command on the GPU job with the model gateway's settings.
# Usage (login node): srun --jobid <job> --overlap ops/injob.sh uv run paperfactory run --config ...
# LLM_ENV_FILE: the gateway's env file (default: ../../llm-setup/.env, i.e. the parent repo when this repo is its
# submodule at services/paper-factory). Only LITELLM_MASTER_KEY is read from it; it is never printed.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${LLM_ENV_FILE:-$REPO/../../llm-setup/.env}"
if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE"; set +a; fi
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV UV_CACHE_DIR UV_PYTHON_INSTALL_DIR
# Python env on the node's own disk (/tmp), not on Ceph (24 Nov 2026): Python start-up reads thousands of small
# files, and four runs doing that at once from Ceph came right before gammaweb08's Ceph mount hung. /tmp is cleared
# when the Slurm job ends; the first use in a job installs from uv's cache, later uses only check it.
export UV_PROJECT_ENVIRONMENT="/tmp/${USER}-paperfactory-venv"
export UV_LINK_MODE=copy
(cd "$REPO" && uv sync --locked -q) || { echo "injob: uv sync failed" >&2; exit 1; }
# Model calls go through model-queue (:4100) when it runs in this job, else straight to LiteLLM (:4000).
# An explicit LITELLM_BASE_URL wins. The queue keeps evqa, paper-factory and baselines together under the GPU limit.
if [ -z "${LITELLM_BASE_URL:-}" ]; then
  if curl -sf -m 2 http://127.0.0.1:4100/health >/dev/null 2>&1; then
    export LITELLM_BASE_URL=http://127.0.0.1:4100/v1
  else
    export LITELLM_BASE_URL=http://127.0.0.1:4000/v1
    echo "injob: model-queue not running on :4100, calling LiteLLM :4000 directly" >&2
  fi
fi
cd "$REPO"
exec "$@"
