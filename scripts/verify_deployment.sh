#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${1:-}"
BASE_URL="${2:-http://127.0.0.1:7860}"

if [[ -z "${DATASET_PATH}" ]]; then
  echo "Usage: $0 /datasets/path/to/lerobot [http://127.0.0.1:7860]" >&2
  exit 2
fi

BASE_URL="${BASE_URL%/}"
CURL=(curl --fail --silent --show-error --connect-timeout 5 --max-time 60)

ROOT_STATUS="$(curl --silent --output /dev/null --write-out '%{http_code}' \
  --connect-timeout 5 --max-time 15 "${BASE_URL}/")"
if [[ "${ROOT_STATUS}" != "200" ]]; then
  echo "[FAIL] frontend returned HTTP ${ROOT_STATUS}" >&2
  exit 1
fi
echo "[OK] frontend HTTP 200"

HEALTH="$(${CURL[@]} "${BASE_URL}/api/health")"
printf '%s' "${HEALTH}" | python3 -c \
  'import json, sys; payload=json.load(sys.stdin); assert payload.get("ok") is True, payload'
echo "[OK] API health"

OPEN_BODY="$(python3 -c \
  'import json, sys; print(json.dumps({"path": sys.argv[1]}, ensure_ascii=False))' \
  "${DATASET_PATH}")"
MANIFEST="$(${CURL[@]} \
  -H 'Content-Type: application/json' \
  --data "${OPEN_BODY}" \
  "${BASE_URL}/api/datasets/open")"
DATASET_ID="$(printf '%s' "${MANIFEST}" | python3 -c \
  'import json, sys; print(json.load(sys.stdin)["dataset_id"])')"
VERSION="$(printf '%s' "${MANIFEST}" | python3 -c \
  'import json, sys; print(json.load(sys.stdin).get("version", "unknown"))')"
EPISODE_COUNT="$(printf '%s' "${MANIFEST}" | python3 -c \
  'import json, sys; print(json.load(sys.stdin).get("total_episodes", 0))')"
echo "[OK] opened dataset id=${DATASET_ID} version=${VERSION} episodes=${EPISODE_COUNT}"

EPISODES="$(${CURL[@]} \
  "${BASE_URL}/api/datasets/${DATASET_ID}/episodes?offset=0&limit=1")"
EPISODE_INDEX="$(printf '%s' "${EPISODES}" | python3 -c \
  'import json, sys; items=json.load(sys.stdin).get("items", []); assert items, "dataset has no episodes"; print(items[0]["episode_index"])')"
DETAIL="$(${CURL[@]} \
  "${BASE_URL}/api/datasets/${DATASET_ID}/episodes/${EPISODE_INDEX}?profile=policy")"
DETAIL_SUMMARY="$(printf '%s' "${DETAIL}" | python3 -c '
import json, sys
p=json.load(sys.stdin)
print("frames={} fps={} videos={}".format(
    p.get("length"), p.get("fps"), len(p.get("videos", []))
))
')"
echo "[OK] episode ${EPISODE_INDEX} parsed: ${DETAIL_SUMMARY}"

VIDEO_COUNT="$(printf '%s' "${DETAIL}" | python3 -c \
  'import json, sys; print(len(json.load(sys.stdin).get("videos", [])))')"
if [[ "${VIDEO_COUNT}" -gt 0 ]]; then
  RANGE_STATUS=""
  VIDEO_KEY=""
  while IFS=$'\t' read -r candidate_key candidate_path; do
    if [[ "${VERSION}" == v3* ]]; then
      candidate_status="$(curl --silent --output /dev/null --write-out '%{http_code}' \
        --connect-timeout 5 --max-time 180 \
        -H 'Range: bytes=0-1023' \
        --get \
        --data-urlencode "episode_index=${EPISODE_INDEX}" \
        --data-urlencode "video_key=${candidate_key}" \
        "${BASE_URL}/api/datasets/${DATASET_ID}/episode-media")"
    else
      candidate_status="$(curl --silent --output /dev/null --write-out '%{http_code}' \
        --connect-timeout 5 --max-time 30 \
        -H 'Range: bytes=0-1023' \
        --get --data-urlencode "path=${candidate_path}" \
        "${BASE_URL}/api/datasets/${DATASET_ID}/media")"
    fi
    RANGE_STATUS="${candidate_status}"
    if [[ "${candidate_status}" == "206" ]]; then
      VIDEO_KEY="${candidate_key}"
      break
    fi
  done < <(
    printf '%s' "${DETAIL}" | python3 -c '
import json, sys
for video in json.load(sys.stdin).get("videos", []):
    print("{}\t{}".format(video.get("key", ""), video.get("relative_path", "")))
'
  )
  if [[ "${RANGE_STATUS}" != "206" ]]; then
    echo "[FAIL] no declared video returned HTTP Range 206; last status ${RANGE_STATUS:-none}" >&2
    exit 1
  fi
  echo "[OK] media HTTP Range 206 (${VERSION}, ${VIDEO_KEY})"
else
  echo "[SKIP] episode has no video feature"
fi

echo "Deployment verification passed."
