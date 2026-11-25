from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


TEST_ROOT = Path(tempfile.mkdtemp(prefix="lerobot-studio-tests-")).resolve()
os.environ.setdefault("LEROBOT_ALLOWED_ROOTS", str(TEST_ROOT))
os.environ.setdefault("LEROBOT_STUDIO_WORKDIR", str(TEST_ROOT / "studio-state"))
os.environ.setdefault("LEROBOT_SCAN_WORKERS", "2")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_video(path: Path, frame_count: int, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 96
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            image = np.full((64, 96, 3), 35, dtype=np.uint8)
            left = 4 + (index * 5) % 72
            image[18:42, left : left + 18] = [220, 150, 35]
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _features() -> dict[str, object]:
    return {
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "action": {"dtype": "float32", "shape": [2], "names": ["arm", "gripper"]},
        "observation.state": {
            "dtype": "float32",
            "shape": [2],
            "names": ["arm", "gripper"],
        },
        "observation.images.cam": {
            "dtype": "video",
            "shape": [64, 96, 3],
            "video_info": {"video.fps": 10, "video.codec": "mpeg4"},
        },
    }


def _data_table(episode_indices: list[int], frame_count: int = 12) -> pa.Table:
    timestamps: list[float] = []
    frames: list[int] = []
    episodes: list[int] = []
    actions: list[list[float]] = []
    states: list[list[float]] = []
    for episode_index in episode_indices:
        for frame in range(frame_count):
            value = frame / max(1, frame_count - 1)
            timestamps.append(frame / 10)
            frames.append(frame)
            episodes.append(episode_index)
            actions.append([value, 0.5 + 0.1 * np.sin(frame)])
            states.append([value * 0.95, 0.5 + 0.1 * np.sin(max(0, frame - 1))])
    return pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frames, type=pa.int64()),
            "episode_index": pa.array(episodes, type=pa.int64()),
            "task_index": pa.array([0] * len(episodes), type=pa.int64()),
            "action": pa.array(actions, type=pa.list_(pa.float32(), 2)),
            "observation.state": pa.array(states, type=pa.list_(pa.float32(), 2)),
        }
    )


def _split_control_features() -> dict[str, object]:
    features = _features()
    features.pop("action")
    features.pop("observation.state")
    features.update(
        {
            "actions.joint.position": {
                "dtype": "float32",
                "shape": [14],
                "names": [
                    *(f"left_arm_{index}" for index in range(7)),
                    *(f"right_arm_{index}" for index in range(7)),
                ],
            },
            "actions.effector.position": {
                "dtype": "float32",
                "shape": [2],
                "names": ["left_gripper", "right_gripper"],
            },
            "states.joint.position": {
                "dtype": "float32",
                "shape": [14],
                "names": [
                    *(f"left_arm_{index}" for index in range(7)),
                    *(f"right_arm_{index}" for index in range(7)),
                ],
            },
            "states.effector.position": {
                "dtype": "float32",
                "shape": [2],
                "names": ["left_gripper", "right_gripper"],
            },
        }
    )
    return features


def _split_control_table(frame_count: int = 12) -> pa.Table:
    frame = np.arange(frame_count, dtype=np.float32)
    joints = np.stack(
        [0.02 * frame + 0.03 * dimension for dimension in range(14)],
        axis=1,
    )
    effectors = np.stack([0.2 + 0.01 * frame, 0.8 - 0.01 * frame], axis=1)
    return pa.table(
        {
            "timestamp": pa.array(frame / 10, type=pa.float32()),
            "frame_index": pa.array(np.arange(frame_count), type=pa.int64()),
            "episode_index": pa.array([0] * frame_count, type=pa.int64()),
            "task_index": pa.array([0] * frame_count, type=pa.int64()),
            "actions.joint.position": pa.array(joints.tolist(), type=pa.list_(pa.float32(), 14)),
            "actions.effector.position": pa.array(effectors.tolist(), type=pa.list_(pa.float32(), 2)),
            "states.joint.position": pa.array((joints * 0.98).tolist(), type=pa.list_(pa.float32(), 14)),
            "states.effector.position": pa.array((effectors * 0.99).tolist(), type=pa.list_(pa.float32(), 2)),
        }
    )


@pytest.fixture(scope="session")
def v21_dataset() -> Path:
    root = TEST_ROOT / "fixture-v21"
    _write_json(
        root / "meta" / "info.json",
        {
            "codebase_version": "v2.1",
            "robot_type": "test_bimanual",
            "total_episodes": 1,
            "total_frames": 12,
            "fps": 10,
            "chunks_size": 1000,
            "key_frame": {
                "0": {
                    "single": [{"frame_index": 8, "type": "Success Frame"}],
                    "dual": [],
                }
            },
            "features": _features(),
        },
    )
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "length": 12, "tasks": ["move the block"]}],
    )
    _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "move the block"}])
    _write_jsonl(
        root / "meta" / "episodes_detailed_task.jsonl",
        [
            {
                "episode_index": 0,
                "action_config": [
                    {
                        "start_frame": 0,
                        "end_frame": 12,
                        "action_text": "reach and move the block",
                        "language_memory": "grasp before lifting",
                        "success_frame": 8,
                    }
                ],
            }
        ],
    )
    data_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_data_table([0]), data_path)
    _write_video(
        root / "videos" / "chunk-000" / "observation.images.cam" / "episode_000000.mp4",
        12,
    )
    return root


@pytest.fixture(scope="session")
def v30_dataset() -> Path:
    root = TEST_ROOT / "fixture-v30"
    _write_json(
        root / "meta" / "info.json",
        {
            "codebase_version": "v3.0",
            "robot_type": "test_bimanual",
            "total_episodes": 2,
            "total_frames": 24,
            "fps": 10,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": _features(),
        },
    )
    episode_meta = pa.table(
        {
            "episode_index": pa.array([0, 1], type=pa.int64()),
            "length": pa.array([12, 12], type=pa.int64()),
            "tasks": pa.array([["move the block"], ["return home"]]),
            "data/chunk_index": pa.array([0, 0], type=pa.int64()),
            "data/file_index": pa.array([0, 0], type=pa.int64()),
            "dataset_from_index": pa.array([0, 12], type=pa.int64()),
            "dataset_to_index": pa.array([12, 24], type=pa.int64()),
            "videos/observation.images.cam/chunk_index": pa.array([0, 0], type=pa.int64()),
            "videos/observation.images.cam/file_index": pa.array([0, 0], type=pa.int64()),
            "videos/observation.images.cam/from_timestamp": pa.array([0.0, 1.2]),
            "videos/observation.images.cam/to_timestamp": pa.array([1.2, 2.4]),
        }
    )
    meta_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(episode_meta, meta_path)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_data_table([0, 1]), data_path)
    _write_video(
        root / "videos" / "observation.images.cam" / "chunk-000" / "file-000.mp4",
        24,
    )
    return root


@pytest.fixture(scope="session")
def split_v21_dataset() -> Path:
    root = TEST_ROOT / "fixture-v21-split-controls"
    _write_json(
        root / "meta" / "info.json",
        {
            "codebase_version": "v2.1",
            "robot_type": "interndata_a1_style",
            "total_episodes": 1,
            "total_frames": 12,
            "fps": 10,
            "chunks_size": 1000,
            "features": _split_control_features(),
        },
    )
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "length": 12, "tasks": ["bimanual transfer"]}],
    )
    data_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_split_control_table(), data_path)
    _write_video(
        root / "videos" / "chunk-000" / "observation.images.cam" / "episode_000000.mp4",
        12,
    )
    return root


@pytest.fixture(scope="session")
def large_v21_dataset() -> Path:
    """Metadata-only fixture for catching accidental O(N) work on ordinary paging."""
    root = TEST_ROOT / "fixture-v21-50000"
    count = 50_000
    _write_json(
        root / "meta" / "info.json",
        {
            "codebase_version": "v2.1",
            "robot_type": "metadata_scale_test",
            "total_episodes": count,
            "total_frames": count * 12,
            "fps": 10,
            "chunks_size": 1000,
            "features": {},
        },
    )
    _write_jsonl(
        root / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": index,
                "length": 12,
                "tasks": [f"sort object group {index % 17}"],
            }
            for index in range(count)
        ],
    )
    return root
