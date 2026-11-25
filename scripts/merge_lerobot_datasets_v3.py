#!/usr/bin/env python3
"""Merge multiple LeRobot v3.0 datasets into one.

Usage:
    python scripts/merge_lerobot_datasets_v3.py \
        --inputs /path/to/dataset1 /path/to/dataset2 ... \
        --output /path/to/merged_output \
        [--workers 4]

Designed for the LeRobot v3.0 layout produced by sim2real/src/lerobot_recorder.py:
  - data/chunk-XXX/file-YYY.parquet            (one file holds MANY episodes)
  - videos/{video_key}/chunk-XXX/file-YYY.mp4  (one file holds MANY episodes;
    episode boundaries are from/to timestamps in the episodes meta table)
  - meta/episodes/chunk-XXX/file-YYY.parquet   (per-episode metadata table)
  - meta/tasks.parquet, meta/stats.json, meta/info.json

Merge strategy (no re-encoding, no frame re-ordering):
  - Source files are kept intact and get fresh sequential file_index values in
    the output (one source file -> one output file).
  - Rows inside each data parquet are re-indexed in place:
      episode_index += episode offset, index += frame offset,
      task_index remapped through the merged task table.
  - Episode meta rows get the new episode_index / data file index / video file
    index; from/to timestamps stay unchanged because file content and the row
    order inside each file are preserved.
  - meta/stats.json is merged analytically: min/max/count/mean/std are exact
    count-weighted combinations; quantiles are taken from the largest input
    (approximation).

Source data is read-only; output is written to a new directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def load_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing meta/info.json in {dataset_dir}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_episodes_table(dataset_dir: Path):
    """Concatenated meta/episodes table, one row per episode (0..N-1)."""
    files = sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No meta/episodes/*.parquet in {dataset_dir}")
    table = pq.read_table(files[0])
    for f in files[1:]:
        table = pa.concat_tables([table, pq.read_table(f)])
    ep_idx = sorted(table.column("episode_index").to_pylist())
    if ep_idx != list(range(len(ep_idx))):
        raise ValueError(
            f"{dataset_dir}: episode_index in meta/episodes is not 0..N-1 "
            f"(got {ep_idx[:10]}...); recorded episodes may have been discarded")
    return table


def load_tasks_table(dataset_dir: Path):
    path = dataset_dir / "meta" / "tasks.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing meta/tasks.parquet in {dataset_dir}")
    return pq.read_table(path)


def _read_tasks(tasks_table):
    """Return ([task names], [task_index]) from a tasks table.

    Tolerates the pandas-written layout used by lerobot_recorder.py where the
    task string is stored as the DataFrame index column (__index_level_0__).
    """
    t = tasks_table
    if "task" in t.column_names:
        names = t.column("task").to_pylist()
    elif "__index_level_0__" in t.column_names:
        names = t.column("__index_level_0__").to_pylist()
    else:
        raise ValueError(f"tasks table has neither 'task' nor '__index_level_0__' "
                         f"column: {t.schema.names}")
    idxs = (t.column("task_index").to_pylist()
            if "task_index" in t.column_names else list(range(len(names))))
    return [str(n) for n in names], idxs


def extract_features_key(info: dict) -> str:
    """Comparable signature of the features dict for consistency checking."""
    features = info.get("features", {})
    return "|".join(
        f"{key}:{features[key].get('dtype')}:{features[key].get('shape')}"
        for key in sorted(features.keys())
    )


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
    """Copy a single video file. Returns a status string."""
    src, dst = args
    try:
        _raw_copy(src, dst)
        return f"OK: {os.path.basename(dst)}"
    except Exception as e:  # noqa: BLE001
        return f"FAIL: {os.path.basename(dst)} - {e}"


def _flatten_leaves(v):
    """Yield scalar leaves of a nested list/scalar tree in order."""
    if isinstance(v, list):
        for x in v:
            yield from _flatten_leaves(x)
    else:
        yield v


def _rebuild_tree(template, leaves):
    """Rebuild a nested list/scalar tree following `template`'s shape."""
    if isinstance(template, list):
        return [_rebuild_tree(x, leaves) for x in template]
    return next(leaves)


def merge_stats(stats_list: list[dict]) -> dict:
    """Merge per-feature stats dicts.

    min/max/count/mean/std are merged exactly via count-weighted combination
    (element-wise to arbitrary nesting depth); q01..q99 are taken from the
    input with the largest count (they cannot be combined without raw data).
    """
    merged: dict[str, dict] = {}
    for stats in stats_list:
        for key, feat in stats.items():
            entry = merged.setdefault(key, {})
            count_leaves = list(_flatten_leaves(feat.get("count", 0)))
            weight = sum(count_leaves)
            entry["count"] = [a + b for a, b in zip(
                entry.get("count", [0] * len(count_leaves)), count_leaves)]
            for q in ("min", "max"):
                if q in feat:
                    entry.setdefault(q, []).append(feat[q])
            if weight and "mean" in feat and "std" in feat:
                entry.setdefault("_agg", []).append(
                    (weight, feat["mean"], feat["std"]))
            if weight > entry.get("_qcount", -1):
                entry["_qcount"] = weight
                entry["_quantiles"] = feat

    result: dict[str, dict] = {}
    for key, entry in merged.items():
        out: dict = {"count": entry["count"]}

        def rebuild(seq, template=None):
            template = template if template is not None else seq[0]
            return _rebuild_tree(template, iter(seq))

        for q in ("min", "max"):
            if q in entry:
                trees = entry[q]
                op = min if q == "min" else max
                leaves = [op(xs) for xs in zip(*[_flatten_leaves(t) for t in trees])]
                out[q] = rebuild(leaves)
        if "_agg" in entry:
            template = entry["_agg"][0][1]
            weights = [w for w, _, _ in entry["_agg"]]
            total = sum(weights)
            mean_leaves, std_leaves = [], []
            mean_rows = [list(_flatten_leaves(m)) for _, m, _ in entry["_agg"]]
            std_rows = [list(_flatten_leaves(s)) for _, _, s in entry["_agg"]]
            for k in range(len(mean_rows[0])):
                ms = [row[k] for row in mean_rows]
                ss = [row[k] for row in std_rows]
                mu = sum(w * m for w, m in zip(weights, ms)) / total
                var = sum(w * (s * s + m * m)
                          for w, m, s in zip(weights, ms, ss)) / total - mu * mu
                mean_leaves.append(mu)
                std_leaves.append(math.sqrt(max(var, 0.0)))
            out["mean"] = _rebuild_tree(template, iter(mean_leaves))
            out["std"] = _rebuild_tree(template, iter(std_leaves))
        src_feat = entry.get("_quantiles", {})
        for q in ("q01", "q10", "q50", "q90", "q99"):
            if q in src_feat:
                out[q] = src_feat[q]
        result[key] = out
    return result


def main():
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot v3.0 datasets")
    parser.add_argument("--inputs", nargs="+", default=None,
                        help="Input dataset directories")
    parser.add_argument("--output", default=None, help="Output dataset directory")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for video copy")
    args = parser.parse_args()
    if not args.inputs or not args.output:
        parser.error("--inputs and --output are required")

    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory exists and is not empty: {output_dir}")

    source_dirs = [Path(p).expanduser().resolve() for p in args.inputs]
    for d in source_dirs:
        if not (d / "meta" / "info.json").is_file():
            parser.error(f"Not a LeRobot dataset (missing meta/info.json): {d}")

    # ---------------------------------------------------------------- load
    infos, episodes_tables, tasks_tables = {}, {}, {}
    print(f"Merging {len(source_dirs)} datasets into {output_dir}")
    for d in source_dirs:
        info = load_info(d)
        if info.get("codebase_version") != "v3.0":
            print(f"ERROR: {d} is codebase_version="
                  f"{info.get('codebase_version')!r}; this script only merges "
                  f"v3.0 datasets (use merge_lerobot_datasets.py for v2.1)",
                  file=sys.stderr)
            sys.exit(1)
        infos[d] = info
        episodes_tables[d] = load_episodes_table(d)
        tasks_tables[d] = load_tasks_table(d)
        print(f"  Loaded: {d.name} ({info['total_episodes']} episodes, "
              f"{info['total_frames']} frames)")

    # -------------------------------------------------- schema consistency
    sigs = {d: extract_features_key(infos[d]) for d in source_dirs}
    if len(set(sigs.values())) > 1:
        print("ERROR: feature schemas differ across inputs:", file=sys.stderr)
        for d, s in sigs.items():
            print(f"  {d.name}: {s}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------- task map
    task_map: dict[str, int] = {}          # global task string -> task_index
    per_source_task_remap: dict[Path, dict[int, int]] = {}
    for d in source_dirs:
        names, idxs = _read_tasks(tasks_tables[d])
        remap: dict[int, int] = {}
        for old_idx, name in zip(idxs, names):
            if name not in task_map:
                task_map[name] = len(task_map)
            remap[int(old_idx)] = task_map[name]
        per_source_task_remap[d] = remap

    # -------------------------------------------- plan data / video files
    # one source file -> one output file, sequential file_index; sources are
    # merged in command-line order, offsets accumulate in the same order.
    data_plan = []   # (src_path, out_file_index, ep_off, frame_off, task_remap)
    ep_file_map = {}  # (src_dir, src_chunk, src_file) -> out data file_index
    video_plan = {}   # camera key -> [(src_dir, src_path, out_file_index), ...]
    episode_offsets: dict[Path, int] = {}
    frame_offsets: dict[Path, int] = {}
    ep_off = fr_off = 0
    for d in source_dirs:
        episode_offsets[d] = ep_off
        frame_offsets[d] = fr_off
        ep_off += int(episodes_tables[d].num_rows)
        fr_off += int(infos[d]["total_frames"])

    for d in source_dirs:
        src_data_files = sorted((d / "data").glob("chunk-*/*.parquet"))
        if not src_data_files:
            print(f"ERROR: no data files in {d}", file=sys.stderr)
            sys.exit(1)
        for f in src_data_files:
            key = (d, int(f.parent.name.split("-")[1]), int(f.stem.split("-")[1]))
            ep_file_map[key] = len(ep_file_map)
            data_plan.append((f, ep_file_map[key], episode_offsets[d],
                              frame_offsets[d], per_source_task_remap[d]))

    video_keys = sorted(k for k, v in infos[source_dirs[0]]["features"].items()
                        if v.get("dtype") == "video")
    for cam in video_keys:
        plan = []
        for d in source_dirs:
            for f in sorted((d / "videos" / cam).glob("chunk-*/*.mp4")):
                plan.append((d, f, len(plan)))
        video_plan[cam] = plan

    # ------------------------------------------------------- rewrite data
    print(f"\nRewriting {len(data_plan)} data parquet files...")
    data_template = infos[source_dirs[0]].get(
        "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    for src_file, out_idx, d_ep_off, d_fr_off, task_remap in data_plan:
        table = pq.read_table(src_file)
        cols = {}
        for name in table.column_names:
            col = table.column(name)
            if name == "episode_index":
                col = pc.add(col, pa.scalar(d_ep_off, type=col.type))
            elif name == "index":
                col = pc.add(col, pa.scalar(d_fr_off, type=col.type))
            elif name == "task_index":
                col = pa.array([task_remap[int(v)] for v in col.to_pylist()],
                               type=col.type)
            cols[name] = col
        new_table = pa.table(cols, schema=table.schema)
        out_rel = data_template.format(chunk_index=out_idx // 1000,
                                       file_index=out_idx)
        out_path = output_dir / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table, out_path, compression="snappy")
        print(f"  file-{out_idx:03d}: {src_file.parent.parent.name}/"
              f"{src_file.name} -> {out_rel} ({table.num_rows} rows)")

    # -------------------------------------------------------- copy videos
    video_template = infos[source_dirs[0]].get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    video_copy_tasks = []
    for cam in sorted(video_plan):
        for d, src_file, out_idx in video_plan[cam]:
            out_rel = video_template.format(video_key=cam,
                                            chunk_index=out_idx // 1000,
                                            file_index=out_idx)
            video_copy_tasks.append((str(src_file), str(output_dir / out_rel)))

    print(f"\nCopying {len(video_copy_tasks)} video files with {args.workers} workers...")
    failed_copies = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(copy_video_task, t): t for t in video_copy_tasks}
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

    # ------------------------------------------------------- episodes meta
    print("\nWriting meta files...")
    ep_tables = []
    for d in source_dirs:
        t = episodes_tables[d]
        cols = {}
        for name in t.column_names:
            col = t.column(name)
            if name == "episode_index":
                col = pc.add(col, pa.scalar(episode_offsets[d], type=col.type))
            elif name == "data/chunk_index":
                col = pa.array([0] * t.num_rows, type=col.type)
            elif name == "data/file_index":
                src_rows = col.to_pylist()
                src_chunks = t.column("data/chunk_index").to_pylist()
                col = pa.array([ep_file_map[(d, c, f)]
                                for c, f in zip(src_chunks, src_rows)],
                               type=col.type)
            elif name.startswith("videos/") and name.endswith("/chunk_index"):
                col = pa.array([0] * t.num_rows, type=col.type)
            elif name.startswith("videos/") and name.endswith("/file_index"):
                cam = name.split("/")[1]
                cam_map = {(int(f.parent.name.split("-")[1]),
                            int(f.stem.split("-")[1])): oi
                           for dd, f, oi in video_plan[cam] if dd == d}
                src_rows = col.to_pylist()
                src_chunks = t.column(
                    name.replace("/file_index", "/chunk_index")).to_pylist()
                col = pa.array([cam_map[(c, f)]
                                for c, f in zip(src_chunks, src_rows)],
                               type=col.type)
            cols[name] = col
        ep_tables.append(pa.table(cols, schema=t.schema))

    merged_eps = ep_tables[0]
    for t in ep_tables[1:]:
        merged_eps = pa.concat_tables([merged_eps, t])
    eps_dir = output_dir / "meta" / "episodes" / "chunk-000"
    eps_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged_eps, eps_dir / "file-000.parquet", compression="snappy")
    print(f"  meta/episodes: {merged_eps.num_rows} episodes")

    # ------------------------------------------------ tasks / stats / info
    tasks_out = pa.table({
        "task_index": pa.array(list(range(len(task_map))), type=pa.int64()),
        "task": pa.array(list(task_map.keys()), type=pa.string()),
    })
    pq.write_table(tasks_out, output_dir / "meta" / "tasks.parquet",
                   compression="snappy")
    print(f"  meta/tasks.parquet: {len(task_map)} tasks")

    stats_list = []
    for d in source_dirs:
        sp = d / "meta" / "stats.json"
        if sp.is_file():
            stats_list.append(json.loads(sp.read_text(encoding="utf-8")))
    merged_stats = merge_stats(stats_list) if stats_list else {}
    (output_dir / "meta" / "stats.json").write_text(
        json.dumps(merged_stats, ensure_ascii=False), encoding="utf-8")

    new_info = dict(infos[source_dirs[0]])
    new_info["total_episodes"] = int(merged_eps.num_rows)
    new_info["total_frames"] = sum(int(infos[d]["total_frames"])
                                   for d in source_dirs)
    new_info["total_tasks"] = len(task_map)
    new_info["splits"] = {"train": f"0:{merged_eps.num_rows}"}
    (output_dir / "meta" / "info.json").write_text(
        json.dumps(new_info, indent=4, ensure_ascii=False), encoding="utf-8")

    # -------------------------------------------------------------- verify
    print("\n=== Post-merge verification ===")
    verify(output_dir, new_info, merged_eps, video_keys)

    print(f"\nMerge complete: {output_dir}")
    print(f"  Episodes: {new_info['total_episodes']}")
    print(f"  Frames:   {new_info['total_frames']}")
    print(f"  Tasks:    {len(task_map)}")
    print(f"  Data files: {len(data_plan)}, Video files: {len(video_copy_tasks)}")


def verify(output_dir: Path, info: dict, eps_table, video_keys: list[str]) -> None:
    """Post-merge consistency checks; exits 2 on failure."""
    ok = True
    total_frames = int(info["total_frames"])

    # 1. per-episode row counts and global index contiguity
    rows_per_ep: dict[int, int] = {}
    idx_seen: set[int] = set()
    for f in sorted((output_dir / "data").glob("chunk-*/*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "index"])
        for e, i in zip(t.column("episode_index").to_pylist(),
                        t.column("index").to_pylist()):
            rows_per_ep[int(e)] = rows_per_ep.get(int(e), 0) + 1
            idx_seen.add(int(i))
    ep_lengths = dict(zip(eps_table.column("episode_index").to_pylist(),
                          eps_table.column("length").to_pylist()))
    mismatch = {e: (c, ep_lengths.get(e)) for e, c in sorted(rows_per_ep.items())
                if ep_lengths.get(e) != c}
    if mismatch:
        print(f"✗ Row count mismatches (episode: rows vs meta length): {mismatch}",
              file=sys.stderr)
        ok = False
    else:
        print(f"✓ All {len(rows_per_ep)} episodes' Parquet row counts "
              f"match episodes meta")
    if idx_seen != set(range(total_frames)):
        print(f"✗ Global index not contiguous 0..{total_frames - 1} "
              f"({len(idx_seen)} distinct values)", file=sys.stderr)
        ok = False
    else:
        print(f"✓ Global index contiguous 0..{total_frames - 1}")

    # 2. video files referenced by the episodes meta all exist
    missing = set()
    for row in eps_table.to_pylist():
        for cam in video_keys:
            rel = (f"videos/{cam}/chunk-{row[f'videos/{cam}/chunk_index']:03d}/"
                   f"file-{row[f'videos/{cam}/file_index']:03d}.mp4")
            if not (output_dir / rel).is_file():
                missing.add(rel)
    if missing:
        print(f"✗ Missing video files: {sorted(missing)[:5]} ...", file=sys.stderr)
        ok = False
    else:
        print(f"✓ All video files referenced by {len(eps_table)} episodes exist")

    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()

