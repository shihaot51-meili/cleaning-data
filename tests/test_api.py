from __future__ import annotations

import io
import importlib
import json
import shutil
import threading
import time
import zipfile
from pathlib import Path

import av
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from backend.app import app, store
from backend.config import SETTINGS
from backend.dataset import LeRobotDatasetHandle


client = TestClient(app)


def _open(path: Path) -> dict[str, object]:
    response = client.post("/api/datasets/open", json={"path": str(path)})
    assert response.status_code == 200, response.text
    return response.json()


def test_api_review_scan_range_and_export(v21_dataset: Path) -> None:
    manifest = _open(v21_dataset)
    dataset_id = str(manifest["dataset_id"])

    page = client.get(f"/api/datasets/{dataset_id}/episodes").json()
    assert page["filtered_total"] == 1
    detail = client.get(f"/api/datasets/{dataset_id}/episodes/0").json()
    assert detail["series"]["action"]["names"] == ["arm", "gripper"]

    media_path = detail["videos"][0]["relative_path"]
    media = client.get(
        f"/api/datasets/{dataset_id}/media",
        params={"path": media_path},
        headers={"Range": "bytes=0-127"},
    )
    assert media.status_code == 206
    assert media.headers["content-range"].startswith("bytes 0-127/")
    assert len(media.content) == 128

    review = client.put(
        f"/api/datasets/{dataset_id}/episodes/0/review",
        json={"decision": "keep", "reasons": ["verified"], "note": "looks good"},
    )
    assert review.status_code == 200
    assert review.json()["review"]["source"] == "manual"

    scan = client.post(
        f"/api/datasets/{dataset_id}/scan",
        json={"profile": "policy", "include_video": True},
    ).json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{scan['job_id']}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "completed", job
    assert job["completed"] == 1

    applied = client.post(
        f"/api/datasets/{dataset_id}/apply-recommendations?profile=policy",
        json={"apply_drop": True, "apply_review": True, "overwrite_manual": False},
    ).json()
    assert applied["applied"] == 0

    exported = client.get(f"/api/datasets/{dataset_id}/export?profile=policy")
    assert exported.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        assert {
            "cleaning_manifest.jsonl",
            "cleaning_manifest.csv",
            "keep_episode_indices.txt",
            "drop_episode_indices.txt",
            "review_episode_indices.txt",
            "summary.json",
        }.issubset(archive.namelist())
        assert "looks good" in archive.read("cleaning_manifest.jsonl").decode("utf-8")


def test_open_rejects_path_outside_allowlist() -> None:
    response = client.post("/api/datasets/open", json={"path": "/etc"})
    assert response.status_code == 403


def test_scan_can_reconnect_and_cancel(v30_dataset: Path, monkeypatch) -> None:
    app_module = importlib.import_module("backend.app")
    manifest = _open(v30_dataset)
    dataset_id = str(manifest["dataset_id"])
    started = threading.Event()
    release = threading.Event()
    original = app_module.analyze_episode

    def slow_analyze(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5), "test scan was not released"
        return original(*args, **kwargs)

    monkeypatch.setattr(app_module, "analyze_episode", slow_analyze)
    scan = client.post(
        f"/api/datasets/{dataset_id}/scan",
        json={
            "profile": "policy",
            "include_video": False,
            "episode_indices": [0, 1],
        },
    )
    assert scan.status_code == 200, scan.text
    job = scan.json()
    assert job["error_count"] == 0
    assert started.wait(timeout=2)

    active = client.get(f"/api/datasets/{dataset_id}/scan-active").json()["job"]
    assert active["job_id"] == job["job_id"]
    reused = client.post(
        f"/api/datasets/{dataset_id}/scan",
        json={"profile": "policy", "include_video": False},
    ).json()
    assert reused["job_id"] == job["job_id"]

    cancelling = client.post(f"/api/jobs/{job['job_id']}/cancel")
    assert cancelling.status_code == 200
    assert cancelling.json()["status"] in {"cancelling", "cancelled"}
    release.set()
    for _ in range(100):
        final = client.get(f"/api/jobs/{job['job_id']}").json()
        if final["status"] == "cancelled":
            break
        time.sleep(0.05)
    assert final["status"] == "cancelled", final
    assert client.get(f"/api/datasets/{dataset_id}/scan-active").json()["job"] is None


def test_v30_episode_media_proxy_is_seekable_h264_and_cached(
    v30_dataset: Path,
) -> None:
    manifest = _open(v30_dataset)
    dataset_id = str(manifest["dataset_id"])
    params = {
        "episode_index": 1,
        "video_key": "observation.images.cam",
    }

    first = client.get(
        f"/api/datasets/{dataset_id}/episode-media",
        params=params,
    )
    assert first.status_code == 200, first.text
    assert first.headers["content-type"].startswith("video/mp4")
    assert first.headers["x-lerobot-proxy-frames"] == "12"
    with av.open(io.BytesIO(first.content), mode="r") as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert sum(1 for _ in container.decode(stream)) == 12

    ranged = client.get(
        f"/api/datasets/{dataset_id}/episode-media",
        params=params,
        headers={"Range": "bytes=0-127"},
    )
    assert ranged.status_code == 206
    assert ranged.headers["x-lerobot-proxy-cache"] == "hit"
    assert ranged.headers["content-range"].startswith("bytes 0-127/")
    assert len(ranged.content) == 128


def test_batch_apply_preserves_manual_reviews(v30_dataset: Path) -> None:
    manifest = _open(v30_dataset)
    dataset_id = str(manifest["dataset_id"])
    store.upsert_review(dataset_id, 0, "keep", ["verified"], "manual", "manual")
    for episode_index, recommendation in [(0, "review"), (1, "drop")]:
        store.upsert_quality(
            dataset_id,
            episode_index,
            "world_video",
            {
                "score": 50,
                "recommendation": recommendation,
                "flags": [
                    {
                        "code": f"fixture_{recommendation}",
                        "severity": "warning",
                        "label": recommendation,
                        "detail": "fixture",
                    }
                ],
                "metrics": {},
                # 只有带 hard-gate 批准的 drop 建议才会落为 discard；
                # 未批准的 drop 会在写入时降级为 review，最终变成 pending。
                "hard_gate": {
                    "automatic_block_approved": recommendation == "drop"
                },
            },
        )
    applied = client.post(
        f"/api/datasets/{dataset_id}/apply-recommendations?profile=world_video",
        json={"apply_drop": True, "apply_review": True, "overwrite_manual": False},
    )
    assert applied.status_code == 200, applied.text
    assert applied.json()["applied"] == 1
    assert store.get_review(dataset_id, 0)["decision"] == "keep"
    episode_one = store.get_review(dataset_id, 1)
    assert episode_one["decision"] == "discard"
    assert episode_one["source"] == "auto"
    assert episode_one["reasons"] == ["fixture_drop"]


def test_large_metadata_dataset_paging_and_filters_are_responsive(
    large_v21_dataset: Path,
) -> None:
    manifest = _open(large_v21_dataset)
    dataset_id = str(manifest["dataset_id"])

    started = time.perf_counter()
    page = client.get(
        f"/api/datasets/{dataset_id}/episodes",
        params={"offset": 49_900, "limit": 50},
    )
    elapsed = time.perf_counter() - started
    assert page.status_code == 200, page.text
    payload = page.json()
    assert payload["filtered_total"] == 50_000
    assert [item["episode_index"] for item in payload["items"]] == list(
        range(49_900, 49_950)
    )
    assert elapsed < 1.0, f"ordinary page took {elapsed:.3f}s"

    store.upsert_review(dataset_id, 1, "keep", ["verified"], "manual", "manual")
    store.upsert_review(dataset_id, 2, "drop", ["bad"], "manual", "manual")
    store.upsert_quality(
        dataset_id,
        2,
        "policy",
        {
            "score": 20,
            "recommendation": "drop",
            "flags": [
                {
                    "code": "missing_parquet",
                    "severity": "error",
                    "label": "missing",
                    "detail": "fixture",
                }
            ],
            "metrics": {"large_blob": list(range(2_000))},
        },
    )

    unreviewed = client.get(
        f"/api/datasets/{dataset_id}/episodes",
        params={"decision": "unreviewed", "limit": 5},
    ).json()
    assert unreviewed["filtered_total"] == 49_998
    assert {item["episode_index"] for item in unreviewed["items"]}.isdisjoint({1, 2})

    unscanned = client.get(
        f"/api/datasets/{dataset_id}/episodes",
        params={"recommendation": "unscanned", "limit": 5},
    ).json()
    assert unscanned["filtered_total"] == 49_999
    assert 2 not in {item["episode_index"] for item in unscanned["items"]}

    flagged = client.get(
        f"/api/datasets/{dataset_id}/episodes",
        params={"flag": "missing_parquet", "limit": 5},
    ).json()
    assert flagged["filtered_total"] == 1
    assert flagged["items"][0]["episode_index"] == 2
    assert flagged["items"][0]["quality"]["metrics"] == {}
    assert flagged["summary"]["decisions"] == {
        "keep": 1,
        "correct": 0,
        "mask": 0,
        "segment": 0,
        "discard": 1,
        "pending": 0,
        "unreviewed": 49_998,
    }

    exported = client.get(f"/api/datasets/{dataset_id}/export?profile=policy")
    assert exported.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        summary = json.loads(archive.read("summary.json"))
        assert sum(summary["counts"].values()) == 50_000
        assert archive.read("cleaning_manifest.jsonl").count(b"\n") == 50_000


def _extract_zip(response, target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(target)
    return target


def test_clip_export_builds_openable_v21_dataset(v21_dataset: Path) -> None:
    source = _open(v21_dataset)
    response = client.post(
        f"/api/datasets/{source['dataset_id']}/export-clip",
        json={
            "episode_index": 0,
            "start_frame": 2,
            "end_frame": 9,
            "target_version": "v2.1",
            "task": "verified useful interval",
        },
    )
    assert response.status_code == 200, response.text
    extracted = _extract_zip(response, v21_dataset.parent / "clip-export-v21")
    manifest = _open(extracted)
    assert manifest["version"] == "v2.1"
    detail = client.get(f"/api/datasets/{manifest['dataset_id']}/episodes/0").json()
    assert detail["length"] == 7
    assert detail["tasks"] == ["verified useful interval"]
    assert detail["timestamps"][0] == 0.0
    assert detail["segments"][0]["start_frame"] == 0
    assert detail["segments"][0]["end_frame"] == 7
    provenance = json.loads((extracted / "meta" / "clip_provenance.json").read_text())
    assert provenance["source_start_frame"] == 2
    assert provenance["source_end_frame_exclusive"] == 9


def test_clip_export_omits_metadata_video_missing_on_disk(
    v21_dataset: Path,
) -> None:
    source_root = v21_dataset.parent / "fixture-v21-with-missing-video"
    if source_root.exists():
        shutil.rmtree(source_root)
    shutil.copytree(v21_dataset, source_root)
    info_path = source_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.missing_depth"] = {
        "dtype": "video",
        "shape": [64, 96, 1],
        "video_info": {
            "video.fps": 10,
            "video.codec": "ffv1",
            "video.is_depth_map": True,
        },
    }
    info_path.write_text(json.dumps(info, indent=2))

    source = _open(source_root)
    response = client.post(
        f"/api/datasets/{source['dataset_id']}/export-clip",
        json={
            "episode_index": 0,
            "start_frame": 2,
            "end_frame": 9,
            "target_version": "v2.1",
            "task": "clip with available camera only",
        },
    )
    assert response.status_code == 200, response.text
    extracted = _extract_zip(response, v21_dataset.parent / "clip-export-missing-video")
    exported_info = json.loads((extracted / "meta" / "info.json").read_text())
    assert exported_info["total_videos"] == 1
    assert "observation.images.cam" in exported_info["features"]
    assert "observation.images.missing_depth" not in exported_info["features"]
    assert (
        extracted
        / "videos"
        / "chunk-000"
        / "observation.images.cam"
        / "episode_000000.mp4"
    ).is_file()

    provenance = json.loads((extracted / "meta" / "clip_provenance.json").read_text())
    assert provenance["exported_video_keys"] == ["observation.images.cam"]
    assert provenance["omitted_video_streams"] == [
        {
            "key": "observation.images.missing_depth",
            "relative_path": (
                "videos/chunk-000/observation.images.missing_depth/episode_000000.mp4"
            ),
            "reason": "missing_source_file",
        }
    ]

    manifest = _open(extracted)
    detail = client.get(f"/api/datasets/{manifest['dataset_id']}/episodes/0").json()
    assert [video["key"] for video in detail["videos"]] == ["observation.images.cam"]


def test_clip_export_converts_v3_shared_shard_to_openable_v3(v30_dataset: Path) -> None:
    source = _open(v30_dataset)
    response = client.post(
        f"/api/datasets/{source['dataset_id']}/export-clip",
        json={
            "episode_index": 1,
            "start_frame": 3,
            "end_frame": 10,
            "target_version": "v3.0",
            "task": "",
        },
    )
    assert response.status_code == 200, response.text
    extracted = _extract_zip(response, v30_dataset.parent / "clip-export-v30")
    manifest = _open(extracted)
    assert manifest["version"] == "v3.0"
    detail = client.get(f"/api/datasets/{manifest['dataset_id']}/episodes/0").json()
    assert detail["length"] == 7
    assert detail["tasks"] == ["return home"]
    assert detail["videos"][0]["start_time"] == 0.0
    assert abs(detail["videos"][0]["end_time"] - 0.7) < 1e-6


def _tree_sha256(root: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_nested_subtask_annotations_roundtrip_and_export(
    v21_dataset: Path,
    tmp_path: Path,
) -> None:
    dataset_root = v21_dataset.parent / f"subtask-annotation-{tmp_path.name}"
    shutil.copytree(v21_dataset, dataset_root)
    dataset_id = str(_open(dataset_root)["dataset_id"])
    endpoint = f"/api/datasets/{dataset_id}/episodes/0/subtasks"

    parent_response = client.post(
        endpoint,
        json={
            "start_frame": 0,
            "end_frame": 8,
            "instruction": "reach the block and close the gripper",
            "reviewer": "reviewer-a",
            "note": "coarse action inside the official instruction",
            "source_segment_index": 0,
        },
    )
    assert parent_response.status_code == 200, parent_response.text
    parent = parent_response.json()["item"]
    assert parent["depth"] == 0
    assert parent["source_segment_index"] == 0

    child_response = client.post(
        endpoint,
        json={
            "start_frame": 2,
            "end_frame": 5,
            "instruction": "close the gripper around the block",
            "reviewer": "reviewer-a",
            "parent_annotation_id": parent["annotation_id"],
        },
    )
    assert child_response.status_code == 200, child_response.text
    child = child_response.json()["item"]
    assert child["depth"] == 1
    assert child["source_segment_index"] == 0

    overlap = client.post(
        endpoint,
        json={
            "start_frame": 4,
            "end_frame": 6,
            "instruction": "overlapping child",
            "reviewer": "reviewer-b",
            "parent_annotation_id": parent["annotation_id"],
        },
    )
    assert overlap.status_code == 422
    assert "overlaps another annotation" in overlap.text

    detail = client.get(f"/api/datasets/{dataset_id}/episodes/0").json()
    assert len(detail["subtask_annotations"]) == 2
    active_tags = [
        tag
        for tag in detail["frame_tags"]
        if tag["kind"] == "subtask_instruction" and tag["start_frame"] <= 3 < tag["end_frame"]
    ]
    assert {tag["instruction"] for tag in active_tags} == {
        parent["instruction"],
        child["instruction"],
    }

    updated_response = client.put(
        f"{endpoint}/{child['annotation_id']}",
        json={
            "start_frame": 2,
            "end_frame": 5,
            "instruction": "secure the block with the gripper",
            "reviewer": "reviewer-b",
            "parent_annotation_id": parent["annotation_id"],
            "expected_updated_at": child["updated_at"],
        },
    )
    assert updated_response.status_code == 200, updated_response.text
    updated_child = updated_response.json()["item"]
    stale_update = client.put(
        f"{endpoint}/{child['annotation_id']}",
        json={
            "start_frame": 2,
            "end_frame": 5,
            "instruction": "stale edit",
            "reviewer": "reviewer-c",
            "parent_annotation_id": parent["annotation_id"],
            "expected_updated_at": child["updated_at"],
        },
    )
    assert stale_update.status_code == 409

    parent_delete = client.request(
        "DELETE",
        f"{endpoint}/{parent['annotation_id']}",
        json={"reviewer": "reviewer-a", "expected_updated_at": parent["updated_at"]},
    )
    assert parent_delete.status_code == 422
    assert "delete child" in parent_delete.text

    exported = client.get(f"/api/datasets/{dataset_id}/export?profile=policy")
    assert exported.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        assert "subtask_annotations.jsonl" in archive.namelist()
        rows = [
            json.loads(line)
            for line in archive.read("subtask_annotations.jsonl").decode("utf-8").splitlines()
        ]
        summary = json.loads(archive.read("summary.json"))
    assert [row["instruction"] for row in rows] == [
        parent["instruction"],
        updated_child["instruction"],
    ]
    assert summary["subtask_annotation_count"] == 2
    assert summary["subtask_annotated_episodes"] == 1

    child_delete = client.request(
        "DELETE",
        f"{endpoint}/{child['annotation_id']}",
        json={
            "reviewer": "reviewer-b",
            "expected_updated_at": updated_child["updated_at"],
        },
    )
    assert child_delete.status_code == 200, child_delete.text
    parent_delete = client.request(
        "DELETE",
        f"{endpoint}/{parent['annotation_id']}",
        json={"reviewer": "reviewer-a", "expected_updated_at": parent["updated_at"]},
    )
    assert parent_delete.status_code == 200, parent_delete.text
    assert parent_delete.json()["items"] == []


def test_subtask_dataset_export_preserves_complete_episode_without_touching_source(
    v21_dataset: Path,
    tmp_path: Path,
) -> None:
    dataset_root = v21_dataset.parent / f"subtask-materialize-{tmp_path.name}"
    shutil.copytree(v21_dataset, dataset_root)
    source_before = _tree_sha256(dataset_root)
    dataset_id = str(_open(dataset_root)["dataset_id"])
    endpoint = f"/api/datasets/{dataset_id}/episodes/0/subtasks"

    parent = client.post(
        endpoint,
        json={
            "start_frame": 0,
            "end_frame": 8,
            "instruction": "reach and grasp",
            "reviewer": "reviewer-a",
            "source_segment_index": 0,
        },
    ).json()["item"]
    child = client.post(
        endpoint,
        json={
            "start_frame": 2,
            "end_frame": 5,
            "instruction": "close gripper on block",
            "reviewer": "reviewer-a",
            "parent_annotation_id": parent["annotation_id"],
        },
    ).json()["item"]
    sibling = client.post(
        endpoint,
        json={
            "start_frame": 8,
            "end_frame": 12,
            "instruction": "place the block",
            "reviewer": "reviewer-b",
            "source_segment_index": 0,
        },
    ).json()["item"]

    for target_version in ("v2.1", "v3.0"):
        response = client.post(
            f"/api/datasets/{dataset_id}/export-subtask-dataset",
            json={"target_version": target_version},
        )
        assert response.status_code == 200, response.text
        extracted = v21_dataset.parent / (
            f"subtask-export-{target_version.replace('.', '')}-{tmp_path.name}"
        )
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            archive.extractall(extracted)
        info = json.loads((extracted / "meta" / "info.json").read_text())
        assert info["codebase_version"] == target_version
        assert info["total_episodes"] == 1
        assert info["total_frames"] == 12
        provenance = [
            json.loads(line)
            for line in (extracted / "meta" / "subtask_provenance.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(provenance) == 1
        assert provenance[0]["source_episode_index"] == 0
        assert provenance[0]["source_length"] == 12
        assert provenance[0]["output_length"] == 12
        assert [
            (item["start_frame"], item["end_frame"], item["instruction"])
            for item in provenance[0]["flattened_instruction_segments"]
        ] == [
            (0, 2, "reach and grasp"),
            (2, 5, "close gripper on block"),
            (5, 8, "reach and grasp"),
            (8, 12, "place the block"),
        ]
        annotation_ids = {
            item["annotation_id"] for item in provenance[0]["annotations"]
        }
        assert annotation_ids == {
            parent["annotation_id"],
            child["annotation_id"],
            sibling["annotation_id"],
        }
        assert json.loads(
            (extracted / "meta" / "subtask_export.json").read_text()
        )["source_modified"] is False
        tables = [pq.read_table(path) for path in sorted((extracted / "data").rglob("*.parquet"))]
        assert [table.num_rows for table in tables] == [12]
        assert set(tables[0]["episode_index"].to_pylist()) == {0}
        assert set(tables[0]["task_index"].to_pylist()) == {0}
        derived = LeRobotDatasetHandle(extracted, SETTINGS)
        assert len(derived.episodes) == 1
        assert derived.episodes[0].task == "move the block"
        assert derived.read_episode_table(0).num_rows == 12
        assert [
            (segment["start_frame"], segment["end_frame"], segment["instruction"])
            for segment in derived.episodes[0].segments
        ] == [
            (0, 2, "reach and grasp"),
            (2, 5, "close gripper on block"),
            (5, 8, "reach and grasp"),
            (8, 12, "place the block"),
        ]

    assert _tree_sha256(dataset_root) == source_before


def test_staged_edit_roundtrip_and_export_approved(v21_dataset: Path) -> None:
    source = _open(v21_dataset)
    dataset_id = str(source["dataset_id"])

    saved = client.put(
        f"/api/datasets/{dataset_id}/episodes/0/staged-edit",
        json={
            "start_frame": 2,
            "end_frame": 9,
            "task": "approved interval",
            "target_version": "v2.1",
        },
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["item"]["start_frame"] == 2
    assert saved.json()["item"]["end_frame"] == 9

    loaded = client.get(f"/api/datasets/{dataset_id}/episodes/0/staged-edit")
    assert loaded.status_code == 200
    assert loaded.json()["item"]["task"] == "approved interval"
    assert loaded.json()["item"]["target_version"] == "v2.1"

    # 未通过的 episode 不应进入统一导出
    pending = client.put(
        f"/api/datasets/{dataset_id}/episodes/0/review",
        json={"decision": "pending", "reasons": [], "note": ""},
    )
    assert pending.status_code == 200, pending.text
    blocked = client.post(
        f"/api/datasets/{dataset_id}/export-approved",
        json={
            "output_dir": str(SETTINGS.workdir / "exports" / "approved-v21"),
            "target_version": "v2.1",
        },
    )
    assert blocked.status_code == 422

    # 通过后统一导出，按暂存区间裁剪
    keep = client.put(
        f"/api/datasets/{dataset_id}/episodes/0/review",
        json={"decision": "keep", "reasons": ["ok"], "note": ""},
    )
    assert keep.status_code == 200, keep.text
    output_root = SETTINGS.workdir / "exports" / "approved-v21"
    exported = client.post(
        f"/api/datasets/{dataset_id}/export-approved",
        json={"output_dir": str(output_root), "target_version": "v2.1"},
    )
    assert exported.status_code == 200, exported.text
    payload = exported.json()
    assert [item["episode_index"] for item in payload["exported"]] == [0]
    assert payload["exported"][0]["start_frame"] == 2
    assert payload["exported"][0]["end_frame"] == 9
    assert payload["omitted"] == []

    episode_dir = output_root / "episode_000000"
    assert (episode_dir / "meta" / "info.json").is_file()
    assert (episode_dir / "data" / "chunk-000" / "episode_000000.parquet").is_file()
    provenance = json.loads((episode_dir / "meta" / "clip_provenance.json").read_text())
    assert provenance["source_start_frame"] == 2
    assert provenance["source_end_frame_exclusive"] == 9
    assert provenance["raw_dataset_modified"] is False
    manifest = json.loads((output_root / "export_manifest.json").read_text())
    assert manifest["target_version"] == "v2.1"
    assert [item["episode_index"] for item in manifest["exported"]] == [0]

    # 导出的单 episode 数据集可被重新打开
    reopened = _open(episode_dir)
    detail = client.get(f"/api/datasets/{reopened['dataset_id']}/episodes/0").json()
    assert detail["length"] == 7


def test_export_approved_rejects_path_outside_export_root(
    v21_dataset: Path,
    tmp_path: Path,
) -> None:
    dataset_id = str(_open(v21_dataset)["dataset_id"])
    review = client.put(
        f"/api/datasets/{dataset_id}/episodes/0/review",
        json={"decision": "keep", "reasons": ["ok"], "note": ""},
    )
    assert review.status_code == 200, review.text
    response = client.post(
        f"/api/datasets/{dataset_id}/export-approved",
        json={"output_dir": str(tmp_path / "outside"), "target_version": "v3.0"},
    )
    assert response.status_code == 422


def test_export_approved_rejects_source_dataset_dir(v21_dataset: Path) -> None:
    dataset_id = str(_open(v21_dataset)["dataset_id"])
    review = client.put(
        f"/api/datasets/{dataset_id}/episodes/0/review",
        json={"decision": "keep", "reasons": ["ok"], "note": ""},
    )
    assert review.status_code == 200, review.text
    response = client.post(
        f"/api/datasets/{dataset_id}/export-approved",
        json={"output_dir": str(v21_dataset), "target_version": "v2.1"},
    )
    assert response.status_code == 422
