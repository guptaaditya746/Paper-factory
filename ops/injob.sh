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
export LITELLM_BASE_URL="${LITELLM_BASE_URL:-http://127.0.0.1:4000/v1}"
cd "$REPO"
exec "$@"
