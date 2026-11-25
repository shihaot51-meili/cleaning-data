from __future__ import annotations

from pathlib import Path

from backend.config import SETTINGS
from backend.dataset import LeRobotDatasetHandle
from backend.quality import THRESHOLDS, _video_flags, analyze_episode


def test_v21_manifest_payload_and_quality(v21_dataset: Path) -> None:
    dataset = LeRobotDatasetHandle(v21_dataset, SETTINGS)
    manifest = dataset.manifest()
    assert manifest["version"] == "v2.1"
    assert manifest["total_episodes"] == 1
    assert manifest["video_keys"] == ["observation.images.cam"]
    assert all(check["status"] == "pass" for check in manifest["structural_checks"])

    payload = dataset.episode_payload(0, max_points=200)
    assert payload["length"] == 12
    assert payload["segments"][0]["instruction"] == "reach and move the block"
    assert payload["segments"][0]["success_frame"] == 8
    assert payload["key_frames"]["single"][0]["type"] == "Success Frame"
    assert len(payload["series"]["action"]["values"]) == 12

    quality = analyze_episode(dataset, 0, include_video=True)
    assert quality["recommendation"] in {"keep", "review"}
    assert quality["metrics"]["videos"][0]["decodable"] is True
    assert quality["metrics"]["action_dim"] == 2


def test_v30_shared_shards_are_sliced_per_episode(v30_dataset: Path) -> None:
    dataset = LeRobotDatasetHandle(v30_dataset, SETTINGS)
    assert len(dataset.episodes) == 2
    assert dataset.episode(1).videos[0].start_time == 1.2
    table = dataset.read_episode_table(1)
    assert table.num_rows == 12
    assert set(table["episode_index"].to_pylist()) == {1}

    payload = dataset.episode_payload(1, max_points=200)
    assert payload["tasks"] == ["return home"]
    assert payload["videos"][0]["start_time"] == 1.2
    assert payload["timestamps"][0] == 0.0


def test_dataset_path_escape_is_rejected(v21_dataset: Path) -> None:
    dataset = LeRobotDatasetHandle(v21_dataset, SETTINGS)
    try:
        dataset.resolve_relative("../../etc/passwd")
    except PermissionError:
        return
    raise AssertionError("path traversal was not rejected")


def test_feature_names_preserve_bimanual_semantics(v21_dataset: Path) -> None:
    dataset = LeRobotDatasetHandle(v21_dataset, SETTINGS)
    dataset.info["features"]["action"] = {
        "dtype": "float32",
        "shape": [16],
        "field_descriptions": {
            "action/joint/position": {
                "description": "G2A left then right arm joint targets",
                "indices": list(range(14)),
            },
            "action/effector/position": {
                "description": "Left then right gripper target",
                "indices": [14, 15],
            },
        },
    }

    assert dataset.feature_names("action", 16) == [
        *(f"left.joint.position.{index}" for index in range(7)),
        *(f"right.joint.position.{index}" for index in range(7)),
        "left.effector.position.0",
        "right.effector.position.0",
    ]


def test_rgb_quality_thresholds_are_not_applied_to_depth_maps() -> None:
    flags: list[dict[str, str]] = []
    _video_flags(
        {
            "key": "observation.images.head_depth",
            "path": "head_depth.mp4",
            "exists": True,
            "decodable": True,
            "is_depth_map": True,
            "black_frame_ratio": 1.0,
            "median_blur_score": 0.0,
            "freeze_ratio": 1.0,
        },
        flags,
        THRESHOLDS,
    )
    assert flags == []


def test_split_action_and_state_fields_are_grouped_and_scanned(
    split_v21_dataset: Path,
) -> None:
    dataset = LeRobotDatasetHandle(split_v21_dataset, SETTINGS)
    assert dataset.control_feature_keys("action") == [
        "actions.joint.position",
        "actions.effector.position",
    ]
    payload = dataset.episode_payload(0, max_points=200)
    joint_series = payload["series"]["actions.joint.position"]
    assert joint_series["family"] == "action"
    assert joint_series["groups"][:7] == ["left_arm"] * 7
    assert joint_series["groups"][7:] == ["right_arm"] * 7

    quality = analyze_episode(dataset, 0, include_video=False)
    assert "missing_action" not in {flag["code"] for flag in quality["flags"]}
    assert quality["metrics"]["action_dim"] == 16
    assert quality["metrics"]["state_dim"] == 16
    assert quality["metrics"]["action_feature_keys"] == [
        "actions.joint.position",
        "actions.effector.position",
    ]
