#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import SETTINGS
from backend.contract_schema import migrate_contract_v1
from backend.dataset import LeRobotDatasetHandle
from backend.plugins.base import PluginContext
from backend.plugins.forward_dynamics import (
    ForwardDynamicsPredictabilityPlugin,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether recorded action improves held-out future-state "
            "prediction for one LeRobot episode."
        )
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--profile",
        choices=("policy", "world_video", "ego_video"),
        default="policy",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = migrate_contract_v1(
        yaml.safe_load(
            args.contract.expanduser().resolve().read_text(encoding="utf-8")
        )
        or {}
    )
    dataset = LeRobotDatasetHandle(
        args.dataset.expanduser().resolve(),
        SETTINGS,
    )
    result = ForwardDynamicsPredictabilityPlugin().run(
        PluginContext(
            dataset=dataset,
            episode_index=args.episode_index,
            profile=args.profile,
            contract=contract,
        )
    ).model_dump(mode="json")
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if result["status"] == "pass":
        raise SystemExit(0)
    if result["status"] in {
        "needs_review",
        "not_configured",
        "not_applicable",
        "not_run",
    }:
        raise SystemExit(2)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
