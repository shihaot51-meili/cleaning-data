#!/usr/bin/env python3
"""逐 segment 调用 Qwen3-VL，验证 instruction 与三路视频行为是否一致。

基于 scripts/run_vlm_judge.py 的可审计 sidecar 风格和 /cpfs01/fc.liu/test.py 的
Qwen3-VL 视频调用方式开发。对每个 instruction segment，用 ffmpeg 从指定三路相机
视频中截取该段子片段，base64 整段送 VLM，判断行为与 instruction 是否一致。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import SETTINGS
from backend.dataset import LeRobotDatasetHandle

DEFAULT_ENDPOINT = "http://109.22.115.74:8000/v1"
DEFAULT_MODEL_ID = "/cpfs01/haoqingwang/huggingface_models/Qwen3-VL-32B-Instruct"
PROMPT_VERSION = "segment-consistency-v1"
RUNNER_VERSION = "segment-consistency-1.0"

PROMPT_TEMPLATE = """下面是同一段机器人操作的 {n_cameras} 个相机视角视频，按顺序对应：{camera_labels}。
指令段标注的 instruction 是：「{instruction}」
请只关注这一段子片段时间范围内的行为，忽略前后上下文，判断视频中的实际行为是否与该 instruction 一致。

返回严格 JSON（不要 markdown 代码块，不要额外文字）：
{{"task_match": "match" | "mismatch" | "uncertain", "rationale": "用中文简述你看到的行为和判断依据", "evidence_frame": "指出关键行为发生的大致位置或时间描述"}}"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:
        raise RuntimeError(
            "需要 imageio_ffmpeg 来截取视频子片段，请 pip install imageio-ffmpeg"
        ) from error


def _decord_can_read(video_path: str) -> bool:
    try:
        import decord

        vr = decord.VideoReader(video_path)
        return len(vr) > 0
    except Exception:
        return False


def ensure_decord_compatible(video_path: str) -> tuple[str, str | None]:
    """若 decord 无法读取视频，用 ffmpeg + libx264 重新编码一份兼容版本。

    返回 (实际使用路径, 临时目录或 None)。临时目录由调用方清理。
    """
    if _decord_can_read(video_path):
        return video_path, None

    ffmpeg = _ffmpeg_exe()
    src = os.path.abspath(video_path)
    dst_dir = tempfile.mkdtemp(prefix="vlm_reenc_")
    base = os.path.splitext(os.path.basename(src))[0]
    dst = os.path.join(dst_dir, f"{base}_reenc.mp4")

    print(f"[转码] decord 无法读取 {src}，正在用 ffmpeg 重新编码到 {dst} ...")
    decode_args: list[str] = []
    codec = _detect_codec(Path(src))
    if codec == "av1":
        decode_args = ["-c:v", "libaom-av1"]
    result = subprocess.run(
        [ffmpeg, "-y", "-err_detect", "ignore_err", *decode_args,
         "-i", src, "-c:v", "libx264", "-an", "-movflags", "+faststart", dst],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 转码失败 (returncode={result.returncode}):\n"
            f"{result.stderr[-2000:]}"
        )
    if not _decord_can_read(dst):
        raise RuntimeError(f"转码后 decord 仍无法读取: {dst}")
    print(f"[转码] 完成，使用转码后的文件: {dst}")
    return dst, dst_dir


def encode_video_to_base64(video_path: str) -> str:
    with open(video_path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def _detect_codec(src_path: Path) -> str | None:
    """探测视频编码，用于选择 AV1 专用解码器。

    用 ffmpeg -i 从 stderr 的 "Stream ... Video: ..." 行解析 codec 名，
    不依赖 ffprobe（static build 可能不含 ffprobe / 不支持 -select_streams）。
    """
    ffmpeg = _ffmpeg_exe()
    try:
        result = subprocess.run(
            [ffmpeg, "-i", str(src_path)],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return None
    # stderr 里形如: Stream #0:0: Video: av1 ...
    for line in result.stderr.splitlines():
        if "Video:" in line:
            parts = line.split("Video:", 1)[1].strip().split()
            if parts:
                return parts[0].rstrip(",")
    return None


def cut_segment_clip(
    src_path: Path,
    start_sec: float,
    end_sec: float,
    dst_path: Path,
) -> None:
    """用 ffmpeg 从 src 截取 [start_sec, end_sec] 子片段到 dst_path。

    先尝试无损流拷贝（-c copy，快），失败则用 libx264 重编码为 H264（decord 友好）。
    AV1 源用 libaom-av1 解码器（libdav1d 在某些 AV1 流上有崩溃 bug）。
    """
    ffmpeg = _ffmpeg_exe()
    duration = max(0.1, end_sec - start_sec)

    # 尝试无损流拷贝（快，但保留原编码如 AV1，decord 可能读不了）
    result = subprocess.run(
        [ffmpeg, "-y", "-ss", f"{start_sec:.4f}", "-t", f"{duration:.4f}",
         "-i", str(src_path), "-c", "copy", "-an", str(dst_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and dst_path.is_file() and dst_path.stat().st_size > 0:
        return

    # 回退：libx264 重编码（解码源 + 重新编码为 H264，decord 友好）
    # AV1 源指定 libaom-av1 解码器（放 -i 前），避免 libdav1d 崩溃
    decode_args: list[str] = []
    codec = _detect_codec(src_path)
    if codec == "av1":
        decode_args = ["-c:v", "libaom-av1"]
    result = subprocess.run(
        [ffmpeg, "-y", "-err_detect", "ignore_err",
         *decode_args,
         "-ss", f"{start_sec:.4f}", "-t", f"{duration:.4f}",
         "-i", str(src_path), "-c:v", "libx264", "-an",
         "-movflags", "+faststart", str(dst_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not dst_path.is_file() or dst_path.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg 截取子片段失败: {result.stderr[-1500:]}"
        )


def call_vlm(
    client: Any,
    model_id: str,
    video_clips: list[tuple[str, str]],
    instruction: str,
    camera_labels: list[str],
    fps: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float]:
    """调用 VLM，返回 (response_payload, latency_seconds)。

    video_clips: [(camera_label, base64_str), ...]
    """
    content: list[dict[str, Any]] = []
    for _label, b64 in video_clips:
        content.append({
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{b64}"},
        })
    prompt_text = PROMPT_TEMPLATE.format(
        n_cameras=len(video_clips),
        camera_labels="、".join(camera_labels),
        instruction=instruction,
    )
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=2048,
        timeout=timeout_seconds,
        extra_body={"mm_processor_kwargs": {"fps": fps, "do_sample_frames": True}},
    )
    latency = time.perf_counter() - started

    raw_text = response.choices[0].message.content or ""
    payload = parse_vlm_response(raw_text)
    return payload, latency


def parse_vlm_response(raw_text: str) -> dict[str, Any]:
    """从 VLM 返回文本中提取 JSON。容忍 markdown 代码块包裹。"""
    text = raw_text.strip()
    # 去掉可能的 ```json ... ``` 包裹
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {
            "task_match": "uncertain",
            "rationale": f"VLM 返回无法解析为 JSON：{raw_text[:500]}",
            "evidence_frame": "",
            "_raw_text": raw_text,
        }
    task_match = str(parsed.get("task_match", "uncertain"))
    if task_match not in {"match", "mismatch", "uncertain"}:
        task_match = "uncertain"
    return {
        "task_match": task_match,
        "rationale": str(parsed.get("rationale", "")),
        "evidence_frame": str(parsed.get("evidence_frame", "")),
        "_raw_text": raw_text,
    }


def build_record(
    *,
    dataset: LeRobotDatasetHandle,
    episode_index: int,
    segment_index: int,
    segment: dict[str, Any],
    video_keys: list[str],
    model_id: str,
    endpoint: str,
    fps: int,
    prompt_version: str,
    retry_history: list[dict[str, Any]],
    response_payload: dict[str, Any] | None,
    raw_response_path: Path | None,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    start_frame = int(segment.get("start_frame", 0))
    end_frame = int(segment.get("end_frame", 0))
    record: dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "status": status,
        "created_at": _now(),
        "dataset_id": dataset.dataset_id,
        "dataset_fingerprint": dataset.fingerprint,
        "episode_index": episode_index,
        "segment_index": segment_index,
        "segment_start_frame": start_frame,
        "segment_end_frame": end_frame,
        "instruction": str(segment.get("instruction", "")),
        "track": str(segment.get("track", "")),
        "video_keys": video_keys,
        "model_id": model_id,
        "endpoint": endpoint,
        "fps": fps,
        "prompt_version": prompt_version,
        "retry_history": retry_history,
        "raw_response_path": str(raw_response_path.resolve()) if raw_response_path else None,
        "claim_boundary": "VLM output is advisory and requires calibrated human review.",
    }
    if error:
        record["error"] = error
    if response_payload is not None:
        record["task_match"] = response_payload.get("task_match", "uncertain")
        record["rationale"] = response_payload.get("rationale", "")
        record["evidence_frame"] = response_payload.get("evidence_frame", "")
    else:
        record["task_match"] = "uncertain"
        record["rationale"] = ""
        record["evidence_frame"] = ""
    return record


def process_segment(
    *,
    dataset: LeRobotDatasetHandle,
    episode_index: int,
    segment_index: int,
    segment: dict[str, Any],
    video_keys: list[str],
    client: Any,
    model_id: str,
    endpoint: str,
    fps: int,
    timeout_seconds: float,
    max_attempts: int,
    prompt_version: str,
    episode_output_dir: Path,
    max_clip_seconds: float,
) -> dict[str, Any]:
    instruction = str(segment.get("instruction", "")).strip()
    start_frame = int(segment.get("start_frame", 0))
    end_frame = int(segment.get("end_frame", 0))

    if not instruction:
        return build_record(
            dataset=dataset, episode_index=episode_index, segment_index=segment_index,
            segment=segment, video_keys=video_keys, model_id=model_id, endpoint=endpoint,
            fps=fps, prompt_version=prompt_version, retry_history=[],
            response_payload=None, raw_response_path=None,
            status="skipped", error="empty instruction",
        )

    episode = dataset.episode(episode_index)
    dataset_fps = float(dataset.info.get("fps", 30))

    # 筛选指定的三路视频
    selected_videos = [
        v for v in episode.videos
        if v.key in set(video_keys) and not v.is_depth_map
    ]
    if not selected_videos:
        return build_record(
            dataset=dataset, episode_index=episode_index, segment_index=segment_index,
            segment=segment, video_keys=video_keys, model_id=model_id, endpoint=endpoint,
            fps=fps, prompt_version=prompt_version, retry_history=[],
            response_payload=None, raw_response_path=None,
            status="failed", error="no matching RGB videos for requested keys",
        )
    # 保持 --video-keys 指定的顺序
    key_to_video = {v.key: v for v in selected_videos}
    ordered_videos = [key_to_video[k] for k in video_keys if k in key_to_video]

    # 帧号 → 各路视频的绝对时间秒
    # segment 帧号是 episode 级别的；视频的 start_time/end_time 是该 mp4 在 episode 中的时间范围
    start_sec = start_frame / dataset_fps
    end_sec = end_frame / dataset_fps
    # 超长 segment 会产生过大的 base64，超过模型 max_model_len；
    # 取该段居中 max_clip_seconds 子窗口（覆盖行为最可能发生的中段）。
    clip_duration = end_sec - start_sec
    if max_clip_seconds and clip_duration > max_clip_seconds:
        mid = (start_sec + end_sec) / 2.0
        start_sec = mid - max_clip_seconds / 2.0
        end_sec = mid + max_clip_seconds / 2.0

    seg_dir = episode_output_dir / f"segment-{segment_index:04d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # 截取三路子片段
    clip_paths: list[Path] = []
    reenc_dirs: list[str] = []
    retry_history: list[dict[str, Any]] = []
    try:
        for video in ordered_videos:
            src = dataset.resolve_relative(video.relative_path)
            if not src.is_file():
                raise FileNotFoundError(f"视频文件不存在: {src}")
            # 把 episode 级时间换算到该 mp4 文件内的相对时间
            clip_start = max(0.0, start_sec - video.start_time)
            clip_end = max(clip_start + 0.1, end_sec - video.start_time)
            safe_key = video.key.replace("/", "_").replace(".", "_")
            clip_dst = seg_dir / f"{safe_key}.mp4"
            cut_segment_clip(src, clip_start, clip_end, clip_dst)

            # 确保 vLLM 服务端 decord 能读
            usable, reenc_dir = ensure_decord_compatible(str(clip_dst))
            clip_paths.append(Path(usable))
            if reenc_dir:
                reenc_dirs.append(reenc_dir)

        # base64 编码三路子片段
        video_clips: list[tuple[str, str]] = []
        for video, clip_path in zip(ordered_videos, clip_paths):
            b64 = encode_video_to_base64(str(clip_path))
            video_clips.append((video.key, b64))

        camera_labels = [v.key for v in ordered_videos]

        # 调用 VLM（带重试）
        response_payload: dict[str, Any] | None = None
        raw_response_path = seg_dir / "response.json"
        last_error: str | None = None

        for attempt in range(1, max(1, max_attempts) + 1):
            started_at = _now()
            try:
                response_payload, latency = call_vlm(
                    client=client,
                    model_id=model_id,
                    video_clips=video_clips,
                    instruction=instruction,
                    camera_labels=camera_labels,
                    fps=fps,
                    timeout_seconds=timeout_seconds,
                )
                raw_response_path.write_text(
                    json.dumps(response_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                retry_history.append({
                    "attempt": attempt,
                    "started_at": started_at,
                    "status": "completed",
                    "latency_seconds": latency,
                })
                last_error = None
                break
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                retry_history.append({
                    "attempt": attempt,
                    "started_at": started_at,
                    "status": "failed",
                    "error": last_error,
                })
                if attempt < max_attempts:
                    time.sleep(min(2 ** attempt, 10))

        if response_payload is None:
            return build_record(
                dataset=dataset, episode_index=episode_index, segment_index=segment_index,
                segment=segment, video_keys=video_keys, model_id=model_id, endpoint=endpoint,
                fps=fps, prompt_version=prompt_version, retry_history=retry_history,
                response_payload=None, raw_response_path=None,
                status="failed", error=last_error or "All VLM attempts failed",
            )

        return build_record(
            dataset=dataset, episode_index=episode_index, segment_index=segment_index,
            segment=segment, video_keys=video_keys, model_id=model_id, endpoint=endpoint,
            fps=fps, prompt_version=prompt_version, retry_history=retry_history,
            response_payload=response_payload, raw_response_path=raw_response_path,
            status="completed", error=None,
        )
    except Exception as error:
        # 视频截取/转码/base64 阶段的异常（如 AV1 解码器崩溃）不应中断整体流程
        return build_record(
            dataset=dataset, episode_index=episode_index, segment_index=segment_index,
            segment=segment, video_keys=video_keys, model_id=model_id, endpoint=endpoint,
            fps=fps, prompt_version=prompt_version,
            retry_history=locals().get("retry_history", []),
            response_payload=None, raw_response_path=None,
            status="failed", error=f"{type(error).__name__}: {error}",
        )
    finally:
        # 清理转码临时目录
        for d in reenc_dirs:
            shutil.rmtree(d, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="逐 segment 调用 Qwen3-VL，验证 instruction 与三路视频行为是否一致。"
    )
    parser.add_argument("dataset", type=Path, help="LeRobot 数据集根目录")
    parser.add_argument("--episodes", type=int, nargs="*", help="指定 episode 编号，不指定则全跑")
    parser.add_argument(
        "--video-keys", nargs="+", required=True,
        help="三路视频 key，如 observation.images.top_head observation.images.hand_right observation.images.hand_left",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"VLM 服务地址，默认 {DEFAULT_ENDPOINT}")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help=f"模型 ID，默认 {DEFAULT_MODEL_ID}")
    parser.add_argument("--fps", type=int, default=4, help="视频采样帧率，默认 4")
    parser.add_argument("--timeout-seconds", type=float, default=3600, help="单次 VLM 调用超时秒数，默认 3600")
    parser.add_argument("--max-attempts", type=int, default=3, help="VLM 调用最大重试次数，默认 3")
    parser.add_argument(
        "--max-clip-seconds", type=float, default=10.0,
        help="单段子片段最大时长（秒），超长 segment 取居中子窗口，避免超过模型 max_model_len，默认 10",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="sidecar 输出目录")
    args = parser.parse_args()

    if len(args.video_keys) < 1:
        parser.error("--video-keys 至少指定一个视频 key")

    # 加载数据集
    dataset = LeRobotDatasetHandle(args.dataset.expanduser().resolve(), SETTINGS)
    episode_indices = args.episodes or sorted(dataset.episode_indices)
    invalid = sorted(set(episode_indices) - dataset.episode_indices)
    if invalid:
        raise ValueError(f"未知 episode 编号: {invalid[:20]}")

    # 初始化 VLM 客户端
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("需要 openai 库：pip install openai") from error
    client = OpenAI(api_key="EMPTY", base_url=args.endpoint, timeout=args.timeout_seconds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = args.output_dir / "segment_vlm_sidecar.jsonl"
    completed = 0
    failed = 0
    skipped = 0

    with sidecar_path.open("w", encoding="utf-8") as sidecar:
        for episode_index in episode_indices:
            episode = dataset.episode(episode_index)
            segments = episode.segments
            if not segments:
                print(f"[episode {episode_index}] 无 instruction segment，跳过")
                continue

            episode_output_dir = args.output_dir / f"episode-{episode_index:09d}"
            episode_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[episode {episode_index}] {len(segments)} 段 instruction segment")

            for segment_index, segment in enumerate(segments):
                instruction = str(segment.get("instruction", "")).strip()
                start_frame = int(segment.get("start_frame", 0))
                end_frame = int(segment.get("end_frame", 0))
                label = f"ep{episode_index} seg{segment_index} [{start_frame}:{end_frame}]"
                if not instruction:
                    print(f"  {label} 跳过（空 instruction）")
                else:
                    print(f"  {label} 「{instruction[:40]}」 调用 VLM ...")

                record = process_segment(
                    dataset=dataset,
                    episode_index=episode_index,
                    segment_index=segment_index,
                    segment=segment,
                    video_keys=args.video_keys,
                    client=client,
                    model_id=args.model_id,
                    endpoint=args.endpoint,
                    fps=args.fps,
                    timeout_seconds=args.timeout_seconds,
                    max_attempts=args.max_attempts,
                    prompt_version=PROMPT_VERSION,
                    episode_output_dir=episode_output_dir,
                    max_clip_seconds=args.max_clip_seconds,
                )

                sidecar.write(json.dumps(record, ensure_ascii=False) + "\n")
                sidecar.flush()

                if record["status"] == "completed":
                    completed += 1
                    print(f"    → {record['task_match']}  {record['rationale'][:60]}")
                elif record["status"] == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    print(f"    → 失败: {record.get('error', '')[:80]}")

    summary = {
        "sidecar": str(sidecar_path),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
