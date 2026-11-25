#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_NAME="$(basename "$ROOT")"
ROOT_PARENT="$(dirname "$ROOT")"
STAMP="${LEROBOT_STUDIO_PACKAGE_STAMP:-$(date +%Y%m%d)}"
OUTPUT="${LEROBOT_STUDIO_PACKAGE_OUTPUT:-$ROOT/dist/lerobot-cleaning-studio-${STAMP}.tar.gz}"
mkdir -p "$(dirname "$OUTPUT")"

cd "$ROOT"
if [[ ! -f frontend/dist/index.html ]]; then
  npm --prefix frontend ci
  npm --prefix frontend run build
fi

tar \
  --exclude="${ROOT_NAME}/.git" \
  --exclude="${ROOT_NAME}/.venv" \
  --exclude="${ROOT_NAME}/frontend/node_modules" \
  --exclude="${ROOT_NAME}/studio-data*" \
  --exclude="${ROOT_NAME}/.e2e-*" \
  --exclude="${ROOT_NAME}/.adaptive-*" \
  --exclude="${ROOT_NAME}/.process-e2e-workdir*" \
  --exclude="${ROOT_NAME}/artifacts" \
  --exclude="${ROOT_NAME}/tmp" \
  --exclude="${ROOT_NAME}/.pytest_cache" \
  --exclude="${ROOT_NAME}/.DS_Store" \
  --exclude="*/__pycache__" \
  --exclude="*.pyc" \
  --exclude="*.pyo" \
  --exclude="${ROOT_NAME}/lerobot-cleaning-studio-*.tar.gz" \
  --exclude="${ROOT_NAME}/lerobot-cleaning-internvla-*.tar.gz" \
  --exclude="${ROOT_NAME}/dist" \
  --exclude="${ROOT_NAME}/docs" \
  --exclude="${ROOT_NAME}/tests" \
  --exclude="${ROOT_NAME}/examples" \
  --exclude="${ROOT_NAME}/*.zip" \
  --exclude="${ROOT_NAME}/${ROOT_NAME}" \
  -czf "$OUTPUT" \
  -C "$ROOT_PARENT" \
  "$ROOT_NAME"

printf 'Source package written to %s\n' "$OUTPUT"
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
