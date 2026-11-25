#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPLY=false
INCLUDE_DEPS=false

for argument in "$@"; do
  case "$argument" in
    --apply) APPLY=true ;;
    --include-deps) INCLUDE_DEPS=true ;;
    *)
      printf 'Unknown argument: %s\n' "$argument" >&2
      exit 2
      ;;
  esac
done

TARGETS=(
  "$ROOT/lerobot_cleaning_studio"
  "$ROOT/lerobot-cleaning-studio-20260722.tar.gz"
  "$ROOT/lerobot-cleaning-studio-20260723.tar.gz"
  "$ROOT/lerobot-cleaning-studio-20260727.tar.gz"
  "$ROOT/lerobot-cleaning-studio-20260728-cleaning-contract.tar.gz"
  "$ROOT/lerobot-cleaning-internvla-20260728-cleaning-contract.tar.gz"
  "$ROOT/lerobot_cleaning_studio.zip"
  "$ROOT/.pytest_cache"
  "$ROOT/backend/__pycache__"
  "$ROOT/tests/__pycache__"
  "$ROOT/studio-data-dev"
  "$ROOT/studio-data-test"
  "$ROOT/studio-data-test-v2"
  "$ROOT/studio-data-test-v3"
  "$ROOT/.DS_Store"
)

if [[ "$INCLUDE_DEPS" == true ]]; then
  TARGETS+=(
    "$ROOT/.venv"
    "$ROOT/frontend/node_modules"
  )
fi

printf '%s\n' 'Cleaning Studio cleanup plan'
printf 'Mode: %s\n' "$([[ "$APPLY" == true ]] && echo APPLY || echo DRY-RUN)"
for target in "${TARGETS[@]}"; do
  if [[ -e "$target" ]]; then
    size="$(du -sh "$target" 2>/dev/null | awk '{print $1}')"
    printf '  %-8s %s\n' "$size" "$target"
    if [[ "$APPLY" == true ]]; then
      rm -rf "$target"
    fi
  fi
done

find "$ROOT" -name '.DS_Store' -type f -print | while read -r target; do
  printf '  %-8s %s\n' "metadata" "$target"
  if [[ "$APPLY" == true ]]; then
    rm -f "$target"
  fi
done

if [[ "$APPLY" == false ]]; then
  printf '\nNothing was deleted. Re-run with --apply after checking the list.\n'
fi
