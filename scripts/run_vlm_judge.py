#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from backend.config import SETTINGS
from backend.dataset import LeRobotDatasetHandle
from backend.vlm_runner import (
    build_vlm_request,
    build_vlm_sidecar_record,
    file_sha256,
    json_sha256,
    run_command_adapter,
    run_http_adapter,
    sample_episode_frames,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行本地命令或 HTTP VLM judge，并生成统一可审计 sidecar。"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--video-keys", nargs="*")
    parser.add_argument("--samples-per-camera", type=int, default=4)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", type=Path)
    adapter = parser.add_mutually_exclusive_group(required=True)
    adapter.add_argument(
        "--command",
        help="命令模板，必须包含 {request} 和 {response} 占位符",
    )
    adapter.add_argument("--endpoint")
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command and (
        "{request}" not in args.command or "{response}" not in args.command
    ):
        raise ValueError("--command must contain {request} and {response}")

    dataset = LeRobotDatasetHandle(args.dataset.expanduser().resolve(), SETTINGS)
    episode_indices = args.episodes or sorted(dataset.episode_indices)
    invalid = sorted(set(episode_indices) - dataset.episode_indices)
    if invalid:
        raise ValueError(f"Unknown episode indices: {invalid[:20]}")
    prompt_text = args.prompt.read_text(encoding="utf-8")
    prompt_sha256 = file_sha256(args.prompt)
    checkpoint_sha256 = (
        file_sha256(args.checkpoint) if args.checkpoint else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = args.output_dir / "vlm_judge_sidecar.jsonl"
    completed = 0
    failed = 0
    with sidecar_path.open("w", encoding="utf-8") as sidecar:
        for episode_index in episode_indices:
            episode_root = args.output_dir / f"episode-{episode_index:09d}"
            retry_history: list[dict[str, object]] = []
            request_payload = None
            try:
                frames = sample_episode_frames(
                    dataset,
                    episode_index,
                    episode_root / "frames",
                    video_keys=args.video_keys,
                    samples_per_camera=args.samples_per_camera,
                )
                request_payload = build_vlm_request(
                    dataset=dataset,
                    episode_index=episode_index,
                    prompt_text=prompt_text,
                    prompt_version=args.prompt_version,
                    frame_artifacts=frames,
                )
                request_path = episode_root / "request.json"
                request_path.parent.mkdir(parents=True, exist_ok=True)
                request_path.write_text(
                    json.dumps(request_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                response_payload = None
                raw_response_path = episode_root / "response.json"
                for attempt in range(1, max(1, args.max_attempts) + 1):
                    started_at = datetime.now(UTC).isoformat()
                    try:
                        if args.command:
                            invocation = run_command_adapter(
                                args.command,
                                request_path=request_path,
                                response_path=raw_response_path,
                                timeout_seconds=args.timeout_seconds,
                            )
                        else:
                            invocation = run_http_adapter(
                                args.endpoint,
                                request_payload=request_payload,
                                timeout_seconds=args.timeout_seconds,
                            )
                            raw_response_path.write_text(
                                json.dumps(
                                    invocation["response"],
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                        response_payload = invocation["response"]
                        retry_history.append(
                            {
                                "attempt": attempt,
                                "started_at": started_at,
                                "status": "completed",
                                "latency_seconds": invocation["latency_seconds"],
                                "stdout": invocation["stdout"],
                                "stderr": invocation["stderr"],
                            }
                        )
                        break
                    except Exception as error:
                        retry_history.append(
                            {
                                "attempt": attempt,
                                "started_at": started_at,
                                "status": "failed",
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                        if attempt < args.max_attempts:
                            time.sleep(min(2**attempt, 10))
                if response_payload is None:
                    raise RuntimeError("All VLM judge attempts failed")
                record = build_vlm_sidecar_record(
                    request_payload=request_payload,
                    response_payload=response_payload,
                    model_id=args.model_id,
                    checkpoint_sha256=checkpoint_sha256,
                    prompt_sha256=prompt_sha256,
                    retry_history=retry_history,
                    raw_response_path=raw_response_path,
                )
                completed += 1
            except Exception as error:
                record = {
                    "runner_version": "1.0",
                    "status": "failed",
                    "created_at": datetime.now(UTC).isoformat(),
                    "dataset_id": dataset.dataset_id,
                    "dataset_fingerprint": dataset.fingerprint,
                    "episode_index": episode_index,
                    "model_id": args.model_id,
                    "checkpoint_sha256": checkpoint_sha256,
                    "prompt_version": args.prompt_version,
                    "prompt_sha256": prompt_sha256,
                    "request_sha256": (
                        json_sha256(request_payload)
                        if request_payload is not None
                        else None
                    ),
                    "retry_history": retry_history,
                    "task_match": "uncertain",
                    "short_description": "",
                    "subtasks": [],
                    "evidence": [],
                    "error": f"{type(error).__name__}: {error}",
                    "claim_boundary": "Failed model calls cannot be treated as judgments.",
                }
                failed += 1
            sidecar.write(json.dumps(record, ensure_ascii=False) + "\n")
            sidecar.flush()
    print(
        json.dumps(
            {
                "sidecar": str(sidecar_path),
                "completed": completed,
                "failed": failed,
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
