#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.retargeting import DexRetargetingRequest, run_dex_retargeting


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official dexsuite/dex-retargeting video example and write "
            "a provenance report. The result is not marked executable."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    request = DexRetargetingRequest.model_validate(payload)
    result = run_dex_retargeting(request, python=args.python)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])


if __name__ == "__main__":
    main()
