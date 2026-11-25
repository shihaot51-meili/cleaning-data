#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.sam2_evaluation import evaluate_sam2_tracking


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SAM2 masks against final human masks. The evaluator does "
            "not require SAM2 weights or a GPU."
        )
    )
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--dataset-fingerprint", required=True)
    parser.add_argument("--manual-baseline", type=Path)
    parser.add_argument("--low-iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sidecar = args.sidecar.expanduser().resolve()
    gold = args.gold.expanduser().resolve()
    report = evaluate_sam2_tracking(
        predictions=read_jsonl(sidecar),
        gold=read_jsonl(gold),
        dataset_fingerprint=args.dataset_fingerprint,
        prediction_base=sidecar.parent,
        gold_base=gold.parent,
        low_iou_threshold=args.low_iou_threshold,
        manual_baseline=(
            read_jsonl(args.manual_baseline.expanduser().resolve())
            if args.manual_baseline
            else []
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
