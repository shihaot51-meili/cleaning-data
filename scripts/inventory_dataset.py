#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import SETTINGS
from backend.dataset import LeRobotDatasetHandle


def inventory_files(root: Path) -> dict:
    counts: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    digest = hashlib.sha256()
    errors: list[dict[str, str]] = []
    for current_root, _, filenames in os.walk(root):
        directory = Path(current_root)
        for filename in filenames:
            path = directory / filename
            relative = path.relative_to(root).as_posix()
            suffix = path.suffix.lower() or "<none>"
            try:
                stat = path.stat()
            except OSError as error:
                errors.append({"path": relative, "error": str(error)})
                continue
            counts[suffix] += 1
            sizes[suffix] += stat.st_size
            digest.update(
                f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
            )
    return {
        "file_count": sum(counts.values()),
        "logical_bytes": sum(sizes.values()),
        "counts_by_suffix": dict(sorted(counts.items())),
        "bytes_by_suffix": dict(sorted(sizes.items())),
        "inventory_fingerprint": digest.hexdigest(),
        "stat_errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a read-only file and LeRobot metadata inventory."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset.expanduser().resolve()
    files = inventory_files(root)
    dataset = LeRobotDatasetHandle(root, SETTINGS)
    payload = {
        "inventory_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_root": str(root),
        "dataset_id": dataset.dataset_id,
        "dataset_fingerprint": dataset.fingerprint,
        "lerobot_version": dataset.version,
        "episode_count": len(dataset.episodes),
        "frame_count": sum(episode.length for episode in dataset.episodes),
        "video_stream_count": sum(len(episode.videos) for episode in dataset.episodes),
        **files,
        "note": (
            "logical_bytes is file size on disk, not bytes read by a sparse scan; "
            "this command never decodes full videos"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
