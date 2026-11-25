#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.adaptive_scan import select_deep_scan_indices


TERMINAL = {"completed", "failed", "cancelled"}


def api_json(
    server: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = (
        json.dumps(payload, ensure_ascii=False).encode()
        if payload is not None
        else None
    )
    request = urllib.request.Request(
        f"{server.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path} failed: HTTP {error.code} {detail}") from error


def write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def wait_for_job(
    server: str,
    job_id: str,
    *,
    poll_seconds: float,
    deadline: float | None,
) -> dict[str, Any]:
    while True:
        job = api_json(server, f"/api/jobs/{job_id}")
        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "status": job["status"],
                    "completed": job.get("completed"),
                    "total": job.get("total"),
                    "error_count": job.get("error_count"),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        if job["status"] in TERMINAL:
            return job
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Adaptive scan timed out while waiting for {job_id}")
        time.sleep(max(0.1, poll_seconds))


def list_all_episodes(
    server: str,
    dataset_id: str,
    profile: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "offset": offset,
                "limit": 500,
                "profile": profile,
            }
        )
        page = api_json(
            server,
            f"/api/datasets/{dataset_id}/episodes?{query}",
        )
        items.extend(page["items"])
        offset += len(page["items"])
        if offset >= int(page["filtered_total"]) or not page["items"]:
            return items


def start_scan(
    server: str,
    dataset_id: str,
    *,
    profile: str,
    include_video: bool,
    video_samples: int,
    video_scan_mode: str,
    video_sequential_stride: int,
    episode_indices: list[int] | None,
) -> dict[str, Any]:
    active = api_json(server, f"/api/datasets/{dataset_id}/scan-active").get("job")
    if active is not None:
        raise RuntimeError(
            f"Dataset already has active scan job {active['job_id']}; "
            "resume with the matching state file or wait for it to finish."
        )
    job = api_json(
        server,
        f"/api/datasets/{dataset_id}/scan",
        method="POST",
        payload={
            "profile": profile,
            "include_video": include_video,
            "video_samples": video_samples,
            "video_scan_mode": video_scan_mode,
            "video_sequential_stride": video_sequential_stride,
            "only_unscanned": False,
            "episode_indices": episode_indices,
        },
    )
    if bool(job["include_video"]) != include_video or job["profile"] != profile:
        raise RuntimeError("Server returned a scan job with incompatible scan settings")
    returned_samples = int(
        job.get(
            "video_samples",
            (job.get("payload") or {}).get("video_samples", 7),
        )
    )
    if returned_samples != video_samples:
        raise RuntimeError("Server returned a scan job with incompatible video_samples")
    returned_mode = str(
        job.get(
            "video_scan_mode",
            (job.get("payload") or {}).get("video_scan_mode", "sampled"),
        )
    )
    returned_stride = int(
        job.get(
            "video_sequential_stride",
            (job.get("payload") or {}).get("video_sequential_stride", 1),
        )
    )
    if returned_mode != video_scan_mode or returned_stride != video_sequential_stride:
        raise RuntimeError(
            "Server returned a scan job with incompatible video scan mode/stride"
        )
    return job


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "先运行全量无视频快速扫描，再对 review/drop 候选和分层 clean 抽样运行视频深扫。"
        )
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:7860")
    parser.add_argument(
        "--profile",
        choices=["policy", "world_video", "ego_video"],
        default="policy",
    )
    parser.add_argument("--audit-fraction", type=float, default=0.02)
    parser.add_argument("--audit-minimum", type=int, default=20)
    parser.add_argument("--deep-video-samples", type=int, default=64)
    parser.add_argument(
        "--deep-video-mode",
        choices=["sampled", "sequential"],
        default="sequential",
    )
    parser.add_argument("--deep-video-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=0)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.deep_video_samples <= 2048:
        raise ValueError("--deep-video-samples must be between 1 and 2048")
    if not 1 <= args.deep_video_stride <= 1000:
        raise ValueError("--deep-video-stride must be between 1 and 1000")

    manifest = api_json(
        args.server,
        "/api/datasets/open",
        method="POST",
        payload={"path": str(args.dataset.expanduser().resolve())},
    )
    state = (
        json.loads(args.state.read_text(encoding="utf-8"))
        if args.state.is_file()
        else {
            "protocol_version": "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_id": manifest["dataset_id"],
            "dataset_fingerprint": manifest["fingerprint"],
            "dataset_root": manifest["root"],
            "profile": args.profile,
            "deep_video_samples": args.deep_video_samples,
            "deep_video_mode": args.deep_video_mode,
            "deep_video_stride": args.deep_video_stride,
            "status": "initialized",
        }
    )
    if (
        state.get("dataset_id") != manifest["dataset_id"]
        or state.get("dataset_fingerprint") != manifest["fingerprint"]
        or state.get("profile") != args.profile
        or int(state.get("deep_video_samples", args.deep_video_samples))
        != args.deep_video_samples
        or state.get("deep_video_mode", args.deep_video_mode)
        != args.deep_video_mode
        or int(state.get("deep_video_stride", args.deep_video_stride))
        != args.deep_video_stride
    ):
        raise ValueError("Existing state file belongs to another dataset/profile version")
    deadline = (
        time.monotonic() + args.timeout_seconds
        if args.timeout_seconds > 0
        else None
    )
    write_state(args.state, state)

    if "fast_job_id" not in state:
        fast = start_scan(
            args.server,
            manifest["dataset_id"],
            profile=args.profile,
            include_video=False,
            video_samples=7,
            video_scan_mode="sampled",
            video_sequential_stride=1,
            episode_indices=None,
        )
        state.update({"fast_job_id": fast["job_id"], "status": "fast_running"})
        write_state(args.state, state)
    fast = wait_for_job(
        args.server,
        state["fast_job_id"],
        poll_seconds=args.poll_seconds,
        deadline=deadline,
    )
    state["fast_job"] = fast
    if fast["status"] != "completed":
        state["status"] = f"fast_{fast['status']}"
        write_state(args.state, state)
        raise SystemExit(2)

    if "deep_selection" not in state:
        episodes = list_all_episodes(
            args.server,
            manifest["dataset_id"],
            args.profile,
        )
        state["deep_selection"] = select_deep_scan_indices(
            episodes,
            audit_fraction=args.audit_fraction,
            audit_minimum=args.audit_minimum,
            seed=args.seed,
        )
        state["deep_selection"]["policy"]["video_samples"] = (
            args.deep_video_samples
        )
        state["deep_selection"]["policy"]["video_scan_mode"] = (
            args.deep_video_mode
        )
        state["deep_selection"]["policy"]["video_sequential_stride"] = (
            args.deep_video_stride
        )
        state["status"] = "fast_completed"
        write_state(args.state, state)
    deep_indices = state["deep_selection"]["deep_scan_indices"]
    if not deep_indices:
        state["status"] = "completed"
        write_state(args.state, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    if "deep_job_id" not in state:
        deep = start_scan(
            args.server,
            manifest["dataset_id"],
            profile=args.profile,
            include_video=True,
            video_samples=args.deep_video_samples,
            video_scan_mode=args.deep_video_mode,
            video_sequential_stride=args.deep_video_stride,
            episode_indices=deep_indices,
        )
        state.update({"deep_job_id": deep["job_id"], "status": "deep_running"})
        write_state(args.state, state)
    deep = wait_for_job(
        args.server,
        state["deep_job_id"],
        poll_seconds=args.poll_seconds,
        deadline=deadline,
    )
    state["deep_job"] = deep
    state["status"] = (
        "completed" if deep["status"] == "completed" else f"deep_{deep['status']}"
    )
    write_state(args.state, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    raise SystemExit(0 if deep["status"] == "completed" else 3)


if __name__ == "__main__":
    main()
