#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.assistance_feedback import evaluate_vlm_assistance_feedback
from backend.calibration import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize accepted/edited/rejected VLM suggestions saved by human review."
        )
    )
    parser.add_argument(
        "--cleaning-manifest",
        type=Path,
        action="append",
        required=True,
        help="Repeat for multiple exported cleaning_manifest.jsonl files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        row
        for path in args.cleaning_manifest
        for row in read_jsonl(path.expanduser().resolve())
    ]
    report = evaluate_vlm_assistance_feedback(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes_with_feedback": report["episodes_with_feedback"],
                "suggestion_feedback_count": report[
                    "suggestion_feedback_count"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
