#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import SETTINGS
from backend.contracts import ContractStore
from backend.dataset import LeRobotDatasetHandle
from backend.plugins.base import PluginContext
from backend.plugins.runner import merge_plugin_results, run_plugins
from backend.quality import analyze_episode, build_length_baselines


def process_resources() -> tuple[float | None, int | None, int | None]:
    try:
        import psutil
    except ImportError:
        return None, None, None
    process = psutil.Process()
    cpu = process.cpu_times()
    io_reader = getattr(process, "io_counters", None)
    io = io_reader() if io_reader is not None else None
    return (
        float(cpu.user + cpu.system),
        int(process.memory_info().rss),
        int(getattr(io, "read_bytes", 0)) if io is not None else None,
    )


def linux_read_bytes() -> int | None:
    path = Path("/proc/self/io")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split(":", 1)[1].strip())
    return None


def stratified_episode_indices(
    dataset: LeRobotDatasetHandle,
    count: int,
    seed: int,
) -> list[int]:
    episodes = list(dataset.episodes)
    if count >= len(episodes):
        return sorted(episode.episode_index for episode in episodes)
    ordered = sorted(
        episodes,
        key=lambda episode: (
            episode.length,
            len(episode.videos),
            episode.task,
            episode.episode_index,
        ),
    )
    positions = {
        round(index * (len(ordered) - 1) / max(1, count - 1))
        for index in range(count)
    }
    selected = [ordered[position].episode_index for position in sorted(positions)]
    if len(selected) < count:
        selected_set = set(selected)
        remaining = [
            episode.episode_index
            for episode in ordered
            if episode.episode_index not in selected_set
        ]
        random.Random(seed).shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    return sorted(selected)


def represented_source_files(
    dataset: LeRobotDatasetHandle,
    indices: list[int],
) -> dict[str, object]:
    paths: set[Path] = set()
    missing: list[str] = []
    for index in indices:
        episode = dataset.episode(index)
        paths.add(dataset.root / episode.data_relative_path)
        for video in episode.videos:
            paths.add(dataset.root / video.relative_path)
    logical_bytes = 0
    for path in paths:
        try:
            logical_bytes += path.stat().st_size
        except OSError:
            missing.append(str(path))
    return {
        "unique_source_file_count": len(paths),
        "unique_source_file_logical_bytes": logical_bytes,
        "missing_source_files": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark deterministic scan throughput on representative episodes."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--profile", choices=["policy", "world_video", "ego_video"], default="policy")
    parser.add_argument("--include-video", action="store_true")
    parser.add_argument(
        "--sample-strategy",
        choices=("stratified", "head"),
        default="stratified",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = LeRobotDatasetHandle(args.dataset.expanduser().resolve(), SETTINGS)
    sample_count = min(len(dataset.episodes), max(1, args.episodes))
    indices = (
        stratified_episode_indices(dataset, sample_count, args.seed)
        if args.sample_strategy == "stratified"
        else sorted(dataset.episode_indices)[:sample_count]
    )
    contract_store = ContractStore(SETTINGS.workdir / "contracts")
    contract, contract_source = contract_store.get(dataset, args.profile)
    baselines = build_length_baselines(dataset)

    started_io = linux_read_bytes()
    started_cpu, started_rss, started_process_read = process_resources()
    started = time.perf_counter()
    runtimes = []
    failures = []
    for episode_index in indices:
        item_started = time.perf_counter()
        try:
            base = analyze_episode(
                dataset,
                episode_index,
                args.profile,
                args.include_video,
                contract=contract,
                length_baselines=baselines,
            )
            result = merge_plugin_results(
                base,
                run_plugins(
                    PluginContext(
                        dataset=dataset,
                        episode_index=episode_index,
                        profile=args.profile,
                        contract=contract,
                    )
                ),
            )
            recommendation = result["recommendation"]
        except Exception as error:
            recommendation = "failed"
            failures.append(
                {
                    "episode_index": episode_index,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        runtimes.append(
            {
                "episode_index": episode_index,
                "seconds": time.perf_counter() - item_started,
                "result": recommendation,
            }
        )
    elapsed = time.perf_counter() - started
    ended_io = linux_read_bytes()
    ended_cpu, ended_rss, ended_process_read = process_resources()
    actual_read_bytes = (
        ended_io - started_io
        if started_io is not None and ended_io is not None
        else None
    )
    episodes_per_second = len(indices) / elapsed if elapsed else None
    projected_dataset_hours = (
        len(dataset.episodes) / episodes_per_second / 3600
        if episodes_per_second
        else None
    )
    source_coverage = represented_source_files(dataset, indices)
    cpu_seconds = (
        ended_cpu - started_cpu
        if started_cpu is not None and ended_cpu is not None
        else None
    )
    process_read_bytes = (
        ended_process_read - started_process_read
        if started_process_read is not None and ended_process_read is not None
        else None
    )
    failure_rate = len(failures) / len(indices) if indices else 1.0
    payload = {
        "benchmark_version": "2.0",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_root": str(dataset.root),
        "dataset_id": dataset.dataset_id,
        "dataset_fingerprint": dataset.fingerprint,
        "contract_source": contract_source,
        "profile": args.profile,
        "include_video": args.include_video,
        "sample_strategy": args.sample_strategy,
        "sample_seed": args.seed,
        "sample_episode_indices": indices,
        "sample_episode_count": len(indices),
        "total_dataset_episodes": len(dataset.episodes),
        "elapsed_seconds": elapsed,
        "episodes_per_second": episodes_per_second,
        "actual_read_bytes_linux": actual_read_bytes,
        "process_read_bytes": process_read_bytes,
        "actual_read_mib_per_second": (
            actual_read_bytes / 1024 / 1024 / elapsed
            if actual_read_bytes is not None and elapsed
            else None
        ),
        "projected_dataset_hours_at_observed_episode_mix": projected_dataset_hours,
        "cpu_seconds": cpu_seconds,
        "average_process_cpu_cores": (
            cpu_seconds / elapsed if cpu_seconds is not None and elapsed else None
        ),
        "rss_bytes_before": started_rss,
        "rss_bytes_after": ended_rss,
        "rss_growth_bytes": (
            ended_rss - started_rss
            if started_rss is not None and ended_rss is not None
            else None
        ),
        "failure_count": len(failures),
        "failure_rate": failure_rate,
        "failures": failures,
        "episode_runtimes": runtimes,
        "source_file_coverage": source_coverage,
        "limitations": [
            "projection is valid only for a representative episode/camera/codec mix",
            "actual_read_bytes is available only on Linux and may be affected by page cache",
            "source file coverage counts shared v3 shards once and is not bytes actually decoded",
            "sparse video scan throughput must not be reported as full-video TB/s",
            "GPU utilization is recorded by the CPFS wrapper when nvidia-smi is available",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
