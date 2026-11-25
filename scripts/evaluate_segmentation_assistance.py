#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import SETTINGS
from backend.dataset import LeRobotDatasetHandle
from backend.segmentation_evaluation import evaluate_segmentation_assistance
from backend.storage import ReviewStore


def _read_jsonl(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate atomic-segmentation boundary quality, label agreement and "
            "review time against an optional manual-only baseline."
        )
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--boundary-tolerance-frames", type=int, default=5)
    parser.add_argument("--manual-baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.boundary_tolerance_frames < 0:
        raise SystemExit("--boundary-tolerance-frames must be non-negative")

    dataset = LeRobotDatasetHandle(args.dataset.expanduser().resolve(), SETTINGS)
    store = ReviewStore(args.db.expanduser().resolve())
    proposals = store.list_segmentation_proposals(dataset.dataset_id)
    fps = float(dataset.info.get("fps", 0.0))
    metadata = {
        int(episode.episode_index): {
            "length": int(episode.length),
            "fps": fps,
            "task": episode.task,
        }
        for episode in dataset.episodes
    }
    report = {
        "dataset_id": dataset.dataset_id,
        "dataset_fingerprint": dataset.fingerprint,
        **evaluate_segmentation_assistance(
            proposals=proposals,
            episode_metadata=metadata,
            boundary_tolerance_frames=args.boundary_tolerance_frames,
            manual_baseline=_read_jsonl(args.manual_baseline),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
