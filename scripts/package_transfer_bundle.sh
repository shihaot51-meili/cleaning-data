#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERNVLA_ROOT="${INTERNVLA_ROOT:-$ROOT/../external_repos/InternVLA-A-series}"
STAMP="${LEROBOT_STUDIO_PACKAGE_STAMP:-$(date +%Y%m%d)}"
OUTPUT="${LEROBOT_TRANSFER_PACKAGE_OUTPUT:-$ROOT/dist/lerobot-cleaning-internvla-${STAMP}.tar.gz}"
mkdir -p "$(dirname "$OUTPUT")"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

if [[ ! -d "$INTERNVLA_ROOT/src/lerobot" ]]; then
  printf 'InternVLA repository not found: %s\n' "$INTERNVLA_ROOT" >&2
  exit 1
fi

if [[ ! -f "$ROOT/frontend/dist/index.html" ]]; then
  npm --prefix "$ROOT/frontend" ci
  npm --prefix "$ROOT/frontend" run build
fi

mkdir -p "$STAGING/transfer/lerobot_cleaning_studio"
PACKAGE_ENTRIES=(
  ".env.example"
  ".gitignore"
  "README_CN.md"
  "backend"
  "docs"
  "examples"
  "frontend"
  "integrations"
  "output/pdf"
  "pyproject.toml"
  "scripts"
  "tests"
  "uv.lock"
)
tar \
  --exclude="frontend/node_modules" \
  --exclude="*/.DS_Store" \
  --exclude="*/__pycache__" \
  --exclude="*.pyc" \
  --exclude="*.pyo" \
  -cf - \
  -C "$ROOT" "${PACKAGE_ENTRIES[@]}" | tar -xf - -C "$STAGING/transfer/lerobot_cleaning_studio"

OVERLAY="$STAGING/transfer/internvla_overlay"
mkdir -p "$OVERLAY"
FILES=(
  "src/lerobot/configs/default.py"
  "src/lerobot/datasets/cleaning_manifest.py"
  "src/lerobot/datasets/factory.py"
  "src/lerobot/transforms/core.py"
  "src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py"
  "src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py"
  "util_scripts/compute_norm_stats_single.py"
  "util_scripts/compute_norm_stats_multi.py"
  "util_scripts/preflight_internvla_training.py"
  "util_scripts/smoke_cleaning_training_step.py"
  "tests/test_cleaning_manifest_contract.py"
  "tests/test_cleaning_training_smoke.py"
  "tests/test_data_pipeline_safety.py"
  "tests/test_training_state_contract.py"
)

for relative_path in "${FILES[@]}"; do
  if [[ ! -f "$INTERNVLA_ROOT/$relative_path" ]]; then
    printf 'Missing InternVLA overlay file: %s\n' "$relative_path" >&2
    exit 1
  fi
  mkdir -p "$OVERLAY/$(dirname "$relative_path")"
  cp "$INTERNVLA_ROOT/$relative_path" "$OVERLAY/$relative_path"
done

git -C "$INTERNVLA_ROOT" rev-parse HEAD > "$STAGING/transfer/INTERNVLA_BASE_COMMIT.txt"
(
  cd "$OVERLAY"
  find . -type f -print0 | sort -z | xargs -0 shasum -a 256
) > "$STAGING/transfer/INTERNVLA_OVERLAY_SHA256.txt"

tar -czf "$OUTPUT" -C "$STAGING" transfer
printf 'Transfer package written to %s\n' "$OUTPUT"
if command -v shasum >/dev/null 2>&1; then
  DIGEST="$(shasum -a 256 "$OUTPUT" | awk '{print $1}')"
else
  DIGEST="$(sha256sum "$OUTPUT" | awk '{print $1}')"
fi
printf 'SHA256: %s\n' "$DIGEST"

CHECKSUMS="$ROOT/dist/SHA256SUMS.txt"
RELATIVE_OUTPUT="dist/$(basename "$OUTPUT")"
TEMP_CHECKSUMS="$(mktemp)"
if [[ -f "$CHECKSUMS" ]]; then
  awk -v target="$RELATIVE_OUTPUT" '$2 != target' "$CHECKSUMS" > "$TEMP_CHECKSUMS"
fi
printf '%s  %s\n' "$DIGEST" "$RELATIVE_OUTPUT" >> "$TEMP_CHECKSUMS"
sort -k2,2 "$TEMP_CHECKSUMS" > "$CHECKSUMS"
rm -f "$TEMP_CHECKSUMS"
