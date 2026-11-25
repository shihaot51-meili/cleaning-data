#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.sam2_protocol import append_prompt_revision, flatten_prompt_revisions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official SAM2VideoPredictor on one video. "
            "This script never downloads checkpoints."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--video-key", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--previous-sidecar", type=Path)
    parser.add_argument("--mark-human-verified", action="store_true")
    parser.add_argument(
        "--review-duration-seconds",
        type=float,
        help="Active human prompt/correction time; omit when it was not measured.",
    )
    return parser.parse_args()


def video_size(path: Path) -> tuple[int, int]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
    if width <= 0 or height <= 0:
        raise ValueError("Could not determine source video size")
    return width, height


def previous_revisions(path: Path | None, episode_index: int, video_key: str) -> list[dict]:
    if path is None:
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (
            int(record.get("episode_index", -1)) == episode_index
            and record.get("video_key") == video_key
        ):
            return list(record.get("prompt_revisions", []))
    raise ValueError("Previous sidecar has no matching episode/video record")


def main() -> None:
    args = parse_args()
    if args.review_duration_seconds is not None and args.review_duration_seconds < 0:
        raise SystemExit("--review-duration-seconds must be non-negative")
    try:
        import torch
        from PIL import Image
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as error:
        raise SystemExit(
            "SAM2 is not installed. Clone facebookresearch/sam2 in the vision "
            "environment and install it before running this bridge."
        ) from error
    raw_prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if isinstance(raw_prompts, dict):
        raw_prompts = raw_prompts.get("prompts", [])
    revisions = append_prompt_revision(
        previous_revisions(args.previous_sidecar, args.episode_index, args.video_key),
        raw_prompts,
        reviewer=args.reviewer,
        note=args.note,
    )
    prompts = flatten_prompt_revisions(revisions)
    width, height = video_size(args.video)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_root = args.output_dir / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    predictor = build_sam2_video_predictor(
        args.model_config,
        str(args.checkpoint),
    )
    object_ids: list[int] = []
    mask_areas: dict[int, list[tuple[int, int]]] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(args.video))
        for prompt in prompts:
            object_id = int(prompt["object_id"])
            object_ids.append(object_id)
            if prompt.get("x_norm") is not None:
                points = np.asarray(
                    [[float(prompt["x_norm"]) * width, float(prompt["y_norm"]) * height]],
                    dtype=np.float32,
                )
                labels = np.asarray([int(prompt.get("label", 1))], dtype=np.int32)
            else:
                points = np.asarray(prompt.get("points", []), dtype=np.float32)
                labels = np.asarray(prompt.get("labels", []), dtype=np.int32)
            box = (
                np.asarray(prompt["box"], dtype=np.float32)
                if prompt.get("box") is not None
                else (
                    np.asarray(
                        [
                            float(prompt["box_norm"][0]) * width,
                            float(prompt["box_norm"][1]) * height,
                            float(prompt["box_norm"][2]) * width,
                            float(prompt["box_norm"][3]) * height,
                        ],
                        dtype=np.float32,
                    )
                    if prompt.get("box_norm") is not None
                    else None
                )
            )
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=int(prompt["frame_index"]),
                obj_id=object_id,
                points=points if points.size else None,
                labels=labels if labels.size else None,
                box=box,
            )
        frame_count = 0
        for frame_index, output_ids, logits in predictor.propagate_in_video(state):
            frame_count = max(frame_count, int(frame_index) + 1)
            for output_index, object_id in enumerate(output_ids):
                mask = (logits[output_index] > 0).cpu().numpy().squeeze()
                area = int(np.sum(mask))
                mask_areas.setdefault(int(object_id), []).append(
                    (int(frame_index), area)
                )
                Image.fromarray((mask.astype(np.uint8) * 255)).save(
                    mask_root / f"{int(frame_index):06d}_{int(object_id)}.png"
                )
    drift_frames: set[int] = set()
    for samples in mask_areas.values():
        for (frame, area), (_, previous_area) in zip(
            samples[1:],
            samples[:-1],
            strict=True,
        ):
            if previous_area == 0 or area == 0:
                drift_frames.add(frame)
                continue
            ratio = area / previous_area
            if ratio > 3.0 or ratio < 1 / 3:
                drift_frames.add(frame)
    sidecar = {
        "episode_index": args.episode_index,
        "video_key": args.video_key,
        "frame_count": frame_count,
        "object_ids": sorted(set(object_ids)),
        "mask_directory": str(mask_root.resolve()),
        "drift_frames": sorted(drift_frames),
        "mask_areas": {
            str(object_id): samples
            for object_id, samples in mask_areas.items()
        },
        "model_config": args.model_config,
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_video": str(args.video.resolve()),
        "prompts": prompts,
        "prompt_revisions": revisions,
        "human_prompted": True,
        "human_verified": bool(args.mark_human_verified),
        "verified_by": args.reviewer if args.mark_human_verified else None,
        "review_duration_seconds": args.review_duration_seconds,
    }
    (args.output_dir / "sam2_tracking.jsonl").write_text(
        json.dumps(sidecar, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sidecar, ensure_ascii=False))


if __name__ == "__main__":
    main()
