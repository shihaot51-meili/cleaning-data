#!/usr/bin/env python3
"""Merge multiple LeRobot v2.1 datasets into one.

Usage:
    python scripts/merge_lerobot_datasets.py \
        --inputs /path/to/dataset1 /path/to/dataset2 ... \
        --output /path/to/merged_output \
        [--workers 4]

Features:
- Re-indexes episode_index, frame_index, and global index.
- Copies video files with correct new episode numbering.
- Validates schema consistency across all inputs.
- Post-merge verification: Parquet row counts vs episodes.jsonl, video existence.
- Source data is read-only; output is written to a new directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow
import pyarrow.parquet as pq


def load_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing meta/info.json in {dataset_dir}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_episodes(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "meta" / "episodes.jsonl"
    episodes = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def load_tasks(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "meta" / "tasks.jsonl"
    tasks = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def load_episodes_stats(dataset_dir: Path) -> list[dict]:
    path = dataset_dir / "meta" / "episodes_stats.jsonl"
    stats = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            line = line.strip()
            if line:
                stats.append(json.loads(line))
    return stats


def extract_features_key(info: dict) -> str:
    """Create a comparable string from features for consistency checking."""
    features = info.get("features", {})
    sig_parts = []
    for key in sorted(features.keys()):
        feat = features[key]
        sig_parts.append(f"{key}:{feat.get('dtype')}:{feat.get('shape')}")
    return "|".join(sig_parts)


def collect_episodes_from_dataset(
    dataset_dir: Path,
    info: dict,
    episodes_meta: list[dict],
) -> list[dict]:
    """Collect per-episode metadata from a single dataset.

    Returns a list of dicts with keys:
      - source_dir: Path to the source dataset
      - source_episode_index: int
      - length: int (number of frames)
      - tasks: list[str]
    """
    data_path_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunks_size = info.get("chunks_size", 1000)

    collected = []
    for ep_meta in episodes_meta:
        src_ep_idx = ep_meta["episode_index"]
        chunk = src_ep_idx // chunks_size
        rel_path = data_path_template.format(episode_chunk=chunk, episode_index=src_ep_idx)
        parquet_path = dataset_dir / rel_path
        if not parquet_path.is_file():
            print(f"  WARNING: Parquet not found for episode {src_ep_idx}: {parquet_path}", file=sys.stderr)
            continue
        collected.append({
            "source_dir": str(dataset_dir),
            "source_episode_index": src_ep_idx,
            "length": ep_meta["length"],
            "tasks": ep_meta.get("tasks", []),
            "parquet_path": str(parquet_path),
        })
    return collected


def _raw_copy(src: str, dst: str) -> None:
    """Copy file content only, no metadata/chmod (for restrictive filesystems)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
        while True:
            chunk = f_in.read(1024 * 1024)
            if not chunk:
                break
            f_out.write(chunk)


def copy_video_task(args: tuple[str, str]) -> str:
    """Copy a single video file. Returns status string."""
    src, dst = args
    try:
        _raw_copy(src, dst)
        return f"OK: {os.path.basename(dst)}"
    except Exception as e:
        return f"FAIL: {os.path.basename(dst)} - {e}"


def main():
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot v2.1 datasets")
    parser.add_argument("--inputs", nargs="+", default=None, help="Input dataset directories")
    parser.add_argument("--output", required=True, help="Output merged dataset directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for video copy")
    parser.add_argument("--pattern", default=None, help="Glob pattern to auto-discover input dirs under a parent")
    parser.add_argument("--parent", default=None, help="Parent directory to search for datasets (use with --pattern)")
    args = parser.parse_args()

    # Resolve input directories
    input_dirs: list[Path] = []
    if args.parent and args.pattern:
        parent = Path(args.parent)
        input_dirs = sorted(parent.glob(args.pattern))
        if not input_dirs:
            print(f"No directories matching {args.pattern} in {args.parent}", file=sys.stderr)
            sys.exit(1)
    elif args.inputs:
        input_dirs = [Path(d) for d in args.inputs]
    else:
        print("ERROR: Either --inputs or --parent+--pattern must be provided", file=sys.stderr)
        sys.exit(1)

    # Validate
    output_dir = Path(args.output)
    if output_dir.exists():
        print(f"ERROR: Output directory already exists: {output_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(input_dirs)} datasets into {output_dir}")

    # Load all dataset infos and validate schema consistency
    all_infos = []
    ref_features_key = None
    ref_info = None

    for d in input_dirs:
        if not d.is_dir():
            print(f"  SKIP: not a directory: {d}", file=sys.stderr)
            continue
        info = load_info(d)
        features_key = extract_features_key(info)
        if ref_features_key is None:
            ref_features_key = features_key
            ref_info = info
        elif features_key != ref_features_key:
            print(f"  ERROR: Schema mismatch in {d}", file=sys.stderr)
            print(f"    Expected: {ref_features_key[:200]}...", file=sys.stderr)
            print(f"    Got:      {features_key[:200]}...", file=sys.stderr)
            continue
        all_infos.append((d, info))
        print(f"  Loaded: {d.name}")

    if not all_infos:
        print("ERROR: No valid datasets to merge", file=sys.stderr)
        sys.exit(1)

    print(f"\n{len(all_infos)} datasets passed schema validation")

    # Collect all episodes
    all_episodes = []
    for d, info in all_infos:
        eps_meta = load_episodes(d)
        collected = collect_episodes_from_dataset(d, info, eps_meta)
        all_episodes.extend(collected)
        print(f"  {d.name}: {len(collected)} episodes")

    total_episodes = len(all_episodes)
    total_frames = sum(ep["length"] for ep in all_episodes)
    print(f"\nTotal: {total_episodes} episodes, {total_frames} frames")

    # Build task mapping (deduplicate tasks)
    task_map: dict[str, int] = {}  # task_string -> task_index
    for ep in all_episodes:
        for t in ep["tasks"]:
            if t not in task_map:
                task_map[t] = len(task_map)

    # Create output directory structure
    output_dir.mkdir(parents=True)
    (output_dir / "meta").mkdir()
    (output_dir / "data" / "chunk-000").mkdir(parents=True)
    for video_key in [k for k, v in ref_info["features"].items() if v.get("dtype") == "video"]:
        (output_dir / "videos" / "chunk-000" / video_key).mkdir(parents=True)

    # Process episodes: re-index Parquet and copy videos
    new_episodes_meta = []
    new_episodes_stats = []
    global_frame_offset = 0
    video_copy_tasks: list[tuple[str, str]] = []
    chunks_size = ref_info.get("chunks_size", 1000)
    video_path_template = ref_info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    data_path_template = ref_info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")

    for new_ep_idx, ep in enumerate(all_episodes):
        src_parquet = Path(ep["parquet_path"])
        src_dir = Path(ep["source_dir"])
        src_ep_idx = ep["source_episode_index"]
        length = ep["length"]
        chunk = new_ep_idx // chunks_size
        chunk_dir = output_dir / "data" / f"chunk-{chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        dst_parquet = chunk_dir / f"episode_{new_ep_idx:06d}.parquet"

        # Read Parquet table directly (no pandas dependency)
        table = pq.read_table(src_parquet)
        column_names = table.column_names

        # Build new columns for index fields
        new_columns = {}
        if "episode_index" in column_names:
            new_columns["episode_index"] = pyarrow.array([new_ep_idx] * length, type=table.schema.field("episode_index").type)
        if "frame_index" in column_names:
            new_columns["frame_index"] = pyarrow.array(range(length), type=table.schema.field("frame_index").type)
        if "index" in column_names:
            new_columns["index"] = pyarrow.array(range(global_frame_offset, global_frame_offset + length), type=table.schema.field("index").type)
        if "task_index" in column_names and ep["tasks"]:
            primary_task = ep["tasks"][0]
            new_columns["task_index"] = pyarrow.array([task_map[primary_task]] * length, type=table.schema.field("task_index").type)

        # Replace columns in the table
        for col_name, col_array in new_columns.items():
            idx = column_names.index(col_name)
            table = table.set_column(idx, col_name, col_array)

        # Write Parquet
        pq.write_table(table, str(dst_parquet))

        # Collect video copy tasks
        for video_key, feat in ref_info["features"].items():
            if feat.get("dtype") != "video":
                continue
            src_chunk = src_ep_idx // chunks_size
            src_video_rel = video_path_template.format(episode_chunk=src_chunk, video_key=video_key, episode_index=src_ep_idx)
            src_video = src_dir / src_video_rel
            if not src_video.is_file():
                print(f"  WARNING: Video not found: {src_video}", file=sys.stderr)
                continue
            dst_video_rel = video_path_template.format(episode_chunk=chunk, video_key=video_key, episode_index=new_ep_idx)
            dst_video = output_dir / dst_video_rel
            video_copy_tasks.append((str(src_video), str(dst_video)))

        # Build episode metadata
        new_episodes_meta.append({
            "episode_index": new_ep_idx,
            "tasks": ep["tasks"],
            "length": length,
        })

        # Try to load source episodes_stats and re-index
        src_stats = load_episodes_stats(src_dir)
        for stat in src_stats:
            if stat.get("episode_index") == src_ep_idx:
                stat["episode_index"] = new_ep_idx
                new_episodes_stats.append(stat)
                break

        global_frame_offset += length

        if (new_ep_idx + 1) % 5 == 0 or new_ep_idx == total_episodes - 1:
            print(f"  Processed {new_ep_idx + 1}/{total_episodes} episodes")

    # Copy videos in parallel
    print(f"\nCopying {len(video_copy_tasks)} video files with {args.workers} workers...")
    failed_copies = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(copy_video_task, task): task for task in video_copy_tasks}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            if result.startswith("FAIL"):
                failed_copies.append(result)
            done += 1
            if done % 10 == 0 or done == len(video_copy_tasks):
                print(f"  Videos: {done}/{len(video_copy_tasks)}")

    if failed_copies:
        print(f"\nWARNING: {len(failed_copies)} video copies failed:", file=sys.stderr)
        for f in failed_copies:
            print(f"    {f}", file=sys.stderr)

    # Write meta files
    print("\nWriting meta files...")

    # tasks.jsonl
    tasks_list = [{"task_index": idx, "task": task} for task, idx in sorted(task_map.items(), key=lambda x: x[1])]
    with open(output_dir / "meta" / "tasks.jsonl", "w", encoding="utf-8") as f:
        for t in tasks_list:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # episodes.jsonl
    with open(output_dir / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in new_episodes_meta:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # episodes_stats.jsonl
    if new_episodes_stats:
        with open(output_dir / "meta" / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
            for stat in new_episodes_stats:
                f.write(json.dumps(stat, ensure_ascii=False) + "\n")

    # Copy modality.json if exists
    for src_d, _ in all_infos:
        modality_path = src_d / "meta" / "modality.json"
        if modality_path.is_file():
            _raw_copy(str(modality_path), str(output_dir / "meta" / "modality.json"))
            break

    # info.json
    merged_info = dict(ref_info)
    merged_info["total_episodes"] = total_episodes
    merged_info["total_frames"] = total_frames
    merged_info["total_videos"] = total_episodes * sum(
        1 for v in ref_info["features"].values() if v.get("dtype") == "video"
    )
    merged_info["total_tasks"] = len(task_map)
    merged_info["total_chunks"] = 1  # All in chunk-000 for now
    merged_info["splits"] = {"train": f"0:{total_episodes}"}
    # Remove dataset-specific fields that don't make sense in a merged dataset
    merged_info.pop("discarded_episode_indices", None)

    with open(output_dir / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(merged_info, f, ensure_ascii=False, indent=4)

    # Post-merge verification
    print("\n=== Post-merge verification ===")
    errors = []

    # Verify Parquet row counts
    for ep_meta in new_episodes_meta:
        ep_idx = ep_meta["episode_index"]
        expected_length = ep_meta["length"]
        chunk = ep_idx // chunks_size
        parquet_path = output_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        if not parquet_path.is_file():
            errors.append(f"Missing Parquet for episode {ep_idx}")
            continue
        table = pq.read_table(parquet_path)
        actual_rows = table.num_rows
        if actual_rows != expected_length:
            errors.append(f"Episode {ep_idx}: row count mismatch (expected {expected_length}, got {actual_rows})")

    # Verify video files
    for ep_meta in new_episodes_meta:
        ep_idx = ep_meta["episode_index"]
        chunk = ep_idx // chunks_size
        for video_key, feat in ref_info["features"].items():
            if feat.get("dtype") != "video":
                continue
            video_path = output_dir / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{ep_idx:06d}.mp4"
            if not video_path.is_file():
                errors.append(f"Missing video for episode {ep_idx}, camera {video_key}")

    if errors:
        print(f"VERIFICATION FAILED: {len(errors)} errors found:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print("✓ All Parquet row counts match episodes.jsonl")
        print("✓ All video files exist")
        print(f"\nMerge complete: {output_dir}")
        print(f"  Episodes: {total_episodes}")
        print(f"  Frames: {total_frames}")
        print(f"  Videos: {merged_info['total_videos']}")
        print(f"  Tasks: {len(task_map)}")


if __name__ == "__main__":
    # Need pyarrow import at top level for ProcessPoolExecutor
    import pyarrow
    main()
