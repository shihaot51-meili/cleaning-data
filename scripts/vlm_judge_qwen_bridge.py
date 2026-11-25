#!/usr/bin/env python3
"""Bridge: run_vlm_judge.py 的 lerobot-vlm-judge-v1 request -> Qwen3-VL vLLM -> response.json。

run_vlm_judge.py 的 --endpoint 适配器直接把 request payload POST 给 HTTP 服务，
但 vLLM 的 /v1/chat/completions 不认 lerobot-vlm-judge-v1 schema。
本脚本通过 --command 调用，读 {request} 文件，把其中已抽好的 JPEG 帧
base64 后按相机分组成多图输入送给 Qwen3-VL，要求其返回严格 JSON，
再写出到 {response} 文件，schema 符合 backend.vlm_runner.VLMJudgeResponse。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any


def build_prompt(request: dict[str, Any], grouped_frames: dict[str, list[Path]]) -> str:
    instruction = request.get("instruction", "")
    camera_lines = []
    for key, frames in grouped_frames.items():
        idxs = ", ".join(str(f.stem) for f in frames) or "无"
        camera_lines.append(f"- {key}: {len(frames)} 帧")
    cameras = "\n".join(camera_lines)
    return f"""下面是同一段机器人操作的多相机视角抽帧图片。每路相机按时间顺序给出若干帧。

数据集: {request.get('dataset_id', '?')}
Episode: {request.get('episode_index', '?')}
Episode 指令: 「{instruction}」

各相机帧数:
{cameras}

请判断这些帧展示的实际行为是否与 Episode 指令一致。只返回严格 JSON，不要 markdown 代码块或任何额外文字，schema 如下：
{{"task_match": "match" | "mismatch" | "uncertain", "short_description": "简述行为", "evidence": [{{"start_frame": int, "end_frame": int, "reason": "string"}}], "confidence": 0.0到1.0的浮点, "rationale": "中文判断依据"}}"""


def group_frames(request: dict[str, Any]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for frame in request.get("frames", []):
        path = Path(frame["path"])
        grouped.setdefault(frame["video_key"], []).append(path)
    for key in grouped:
        grouped[key].sort()
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--endpoint", default="http://109.22.115.74:8000/v1")
    parser.add_argument(
        "--model-id",
        default="/cpfs01/haoqingwang/huggingface_models/Qwen3-VL-32B-Instruct",
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    grouped = group_frames(request)

    content: list[dict[str, Any]] = []
    for _key, frames in grouped.items():
        for frame_path in frames:
            b64 = base64.b64encode(frame_path.read_bytes()).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
    content.append({"type": "text", "text": build_prompt(request, grouped)})

    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url=args.endpoint, timeout=args.timeout_seconds)
    resp = client.chat.completions.create(
        model=args.model_id,
        messages=[{"role": "user", "content": content}],
        max_tokens=2048,
    )
    raw = resp.choices[0].message.content or ""

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {
            "task_match": "uncertain",
            "rationale": f"VLM 返回无法解析为 JSON: {raw[:500]}",
            "short_description": "",
            "evidence": [],
            "confidence": 0.0,
        }

    payload.setdefault("task_match", "uncertain")
    payload.setdefault("short_description", "")
    payload.setdefault("evidence", [])
    payload.setdefault("rationale", "")
    if payload["task_match"] not in {"match", "mismatch", "uncertain"}:
        payload["task_match"] = "uncertain"

    Path(args.response).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[bridge] wrote {args.response}: {payload.get('task_match')}", file=sys.stderr)


if __name__ == "__main__":
    main()
