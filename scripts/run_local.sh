#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
DATA_ROOT="${LEROBOT_ALLOWED_ROOTS:-/cpfs01/Embodied/datasets}"
PORT="${LEROBOT_STUDIO_PORT:-7860}"
HOST="${LEROBOT_STUDIO_HOST:-127.0.0.1}"

cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv
  else
    python3 -m venv .venv
  fi
fi
if ! .venv/bin/python -c 'import av, fastapi, numpy, pyarrow, uvicorn' >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python -e '.[dev]'
  else
    .venv/bin/python -m pip install -e '.[dev]'
  fi
fi
if [[ ! -f frontend/dist/index.html ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    printf 'frontend/dist is missing and npm is not installed. Install Node.js 20+ first.\n' >&2
    exit 1
  fi
  NPM_ARGS=(--prefix frontend install)
  if [[ -n "${NPM_REGISTRY:-}" ]]; then
    NPM_ARGS+=(--registry "$NPM_REGISTRY")
  fi
  npm "${NPM_ARGS[@]}"
  npm --prefix frontend run build
fi

export LEROBOT_ALLOWED_ROOTS="$DATA_ROOT"
export LEROBOT_STUDIO_WORKDIR="${LEROBOT_STUDIO_WORKDIR:-$ROOT/studio-data}"
exec .venv/bin/uvicorn backend.app:app --host "$HOST" --port "$PORT" --workers 1
