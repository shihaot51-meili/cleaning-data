# 开发指南

> 返回 [README](../README.md) · 同类文档：[使用说明](USAGE.md) · [部署与配置](DEPLOYMENT.md) · [开发指南](DEVELOPMENT.md)

## 1. 代码目录

```text
backend/
  app.py                 API、扫描任务、媒体 Range、清单与片段导出入口
  contracts.py           cleaning_contract 推断、校验和 workdir 持久化
  contract_schema.py     Contract v1 Pydantic 模型和 JSON Schema
  scanner_protocol.py    扫描器版本、输入/配置摘要和证据协议
  dataset.py             v2/v3 schema、共享 shard、字段家族、语义组和帧标签
  video_proxy.py         v3 共享 shard 按 episode 转 H.264 并缓存
  quality.py             结构、时序、动作/状态和视频规则及证据定位
  ray_worker.py          可选 Ray episode worker、资源声明、超时和强制取消
  vlm_runner.py          多相机取帧、命令/HTTP judge、provenance 与响应校验
  clip_export.py         Parquet/视频同步裁剪，生成 v2.1/v3 数据集
  materialize.py         按清洗清单批量物化已审核的 segment
  artifact_store.py      分区、压缩、带 SHA256 的不可变扫描对象
  storage.py             SQLite 人工结论、任务状态与扫描对象索引
  plugins/               FK、重投影、语义对齐、切段、VLM/SAM2/回放插件
  replay.py              版本锁定的仿真回放协议
  action_layers.py       原生/模型输入/可执行动作三层
  retargeting.py         dex-retargeting 候选桥与关节名重排

frontend/src/
  App.tsx                数据集、播放、扫描、区间和导出状态
  components/VideoGrid.tsx
                         多相机排序、选择和同步
  components/ChartPanel.tsx
                         任意字段/维度组合、分组、悬停帧详情
  components/InspectorPanel.tsx
                         自动检查、人工审核、当前帧、指令段和递归子任务标注

tests/
  conftest.py            自生成 v2.1、v3、拆分动作字段和 50k metadata fixture
  test_dataset.py        parser、分组、质量规则测试
  test_api.py            扫描、五类处置、递归子任务、清单和双版本片段导出测试

scripts/run_local.sh     本地 Python 环境启动
scripts/verify_deployment.sh
                         服务器真实数据验收
scripts/inventory_dataset.py
                         一次性全库文件/逻辑容量清册
scripts/benchmark_scan_throughput.py
                         代表样本 episode 吞吐和 Linux 实际读取量
scripts/package_source.sh
                         生成邮件/内网源码包
scripts/package_transfer_bundle.sh
                         打包工作台和 InternVLA 最小覆盖层
scripts/run_vlm_judge.py
                         通用 episode 级 VLM judge 框架（命令/HTTP 适配器）
scripts/vlm_judge_qwen_bridge.py
                         run_vlm_judge.py 与 Qwen3-VL vLLM 的 bridge
scripts/check_segment_vlm.py
                         逐 segment 调用 Qwen3-VL 核验 instruction 一致性
```

这里的 fixture 是测试程序生成的一小份可控数据，用来验证代码契约，不是假装成真实数据集的训练样本。

## 2. 开发和测试

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
npm --prefix frontend ci
npm --prefix frontend run build
.venv/bin/python -m pytest -q
```

开发模式：

```bash
LEROBOT_ALLOWED_ROOTS=/path/to/datasets \
.venv/bin/uvicorn backend.app:app --reload --port 8000

npm --prefix frontend run dev
```

Vite 默认在 5173，并把 `/api` 代理到 8000。

服务器部署后验收：

```bash
./scripts/verify_deployment.sh /cpfs01/Embodied/datasets/path/to/lerobot
```

它依次检查前端、health API、metadata、首条 episode 和视频 Range。对于 v3，
它会实际生成首路相机的 episode H.264 代理，因此首次检查允许最多等待 180 秒。
最终应看到：

```text
Deployment verification passed.
```

## 3. VLM 验证

工作台不内置 VLM，VLM 核验以外部 sidecar 形式运行，结果只供人工复核。仓库提供两个入口：

- `scripts/run_vlm_judge.py`：通用 episode 级 judge 框架，后端无关。用 PyAV 对每路 RGB
  均匀抽帧（`--samples-per-camera`，默认 4），生成 `lerobot-vlm-judge-v1` request，
  通过 `--command`（外部命令）或 `--endpoint`（裸 HTTP POST）两种适配器调用 judge，
  最后校验响应并写出统一 sidecar。`--endpoint` 直接把 request payload POST 给 HTTP 服务，
  因此只适用于自建的、能直接消费该 schema 的 judge 服务；对 vLLM 这类 OpenAI 兼容服务
  需用 `--command` 配一个 bridge 脚本（见下方）。
- `scripts/check_segment_vlm.py`：逐 instruction segment 核验，固定调用 Qwen3-VL vLLM
  服务。对每个 segment 用 ffmpeg 截取三路相机子片段，base64 整段送模型，判断行为是否与
  该段 instruction 一致。开箱即用，不通用。

两个脚本都写 `claim_boundary: "VLM output is advisory and requires calibrated human review."`，
不自动删除数据。

### 13.1 episode 级核验（run_vlm_judge.py + bridge）

`run_vlm_judge.py` 的 `--endpoint` 适配器与 vLLM 的 `/v1/chat/completions` schema 不兼容，
需通过 `--command` 调用 bridge 脚本 `scripts/vlm_judge_qwen_bridge.py`：bridge 读 request
中已抽好的 JPEG 帧，base64 后按相机分组送 Qwen3-VL chat，写出符合
`backend.vlm_runner.VLMJudgeResponse` 的 response。默认服务 `http://109.22.115.74:8000/v1`，
默认模型 `Qwen3-VL-32B-Instruct`，均可用 bridge 的 `--endpoint` / `--model-id` 覆盖。

以 AgiBotWorld2026_v30 的 313498_314085 数据集 episode 0 为例：

```bash
LEROBOT_ALLOWED_ROOTS=/cpfs01/Embodied/datasets \
.venv/bin/python scripts/run_vlm_judge.py \
  /cpfs01/Embodied/datasets/AgiBotWorld2026_v30/ImitationLearning/CommercialSpaces/task_3400/313498_314085 \
  --episodes 0 \
  --video-keys observation.images.top_head observation.images.hand_right observation.images.hand_left \
  --samples-per-camera 2 \
  --prompt scripts/vlm_prompt.txt --prompt-version demo-v1 \
  --model-id '/cpfs01/haoqingwang/huggingface_models/Qwen3-VL-32B-Instruct' \
  --command '.venv/bin/python scripts/vlm_judge_qwen_bridge.py --request {request} --response {response} --endpoint http://109.22.115.74:8000/v1' \
  --output-dir audits/vlm
```

`--command` 模板中的 `{request}` / `{response}` 是文件路径占位符，由 `run_vlm_judge.py`
替换后执行；bridge 读前者、写后者。`scripts/vlm_prompt.txt` 是示例 prompt 文件，可自行替换。
输出 `vlm_judge_sidecar.jsonl`，每行一个 episode，含 `task_match`/`confidence`/`rationale`、
输入帧 SHA256、`retry_history` 和 `raw_response_path`。单 episode 失败不中断整体流程。

### 13.2 segment 级核验（check_segment_vlm.py）

逐 segment 验证 instruction 与三路视频行为是否一致（Qwen3-VL）。默认服务
`http://109.22.115.74:8000/v1`，默认模型 `Qwen3-VL-32B-Instruct`，可用 `--endpoint` /
`--model-id` 覆盖。以同一数据集跑 episode 0 为例：

```bash
LEROBOT_ALLOWED_ROOTS=/cpfs01/Embodied/datasets \
.venv/bin/python scripts/check_segment_vlm.py \
  /cpfs01/Embodied/datasets/AgiBotWorld2026_v30/ImitationLearning/CommercialSpaces/task_3400/313498_314085 \
  --episodes 0 \
  --video-keys observation.images.top_head observation.images.hand_right observation.images.hand_left \
  --output-dir /tmp/vlm-segment-check
```

输出 `segment_vlm_sidecar.jsonl`，每行一个 segment 的判定结果（`match`/`mismatch`/`uncertain`），
包含 instruction、帧范围、VLM rationale 和重试历史。空 instruction 的 segment 自动跳过，
单个 segment 失败不中断整体流程。

## 4. 按需物化片段

先解压网页导出的清洗 ZIP，再执行：

```bash
python -m backend.materialize \
  --dataset /cpfs01/Embodied/datasets/my_dataset \
  --manifest /path/to/export/cleaning_manifest.jsonl \
  --output /cpfs01/Embodied/processed/my_dataset_segments \
  --target-version v3.0
```

命令为 `segment` 生成子区间数据集，为 `mask` 和 `correct` 生成完整回合的新数据集，
并写出 `materialization_report.json`：

- `mask` 会写入逐帧 `action_valid_mask` 与 `video_valid_mask`，同时保留
  `meta/cleaning_materialization.json`。
- `correct` 第一版支持 `replace_task`、`time_shift`、`add_offset` 和 `scale` 四种
  确定性修正；可在“人工审核 → 内部复核 → 确定性修正”中添加。必须填写审核人和
  修正依据；字段、维度、有限数值及时间偏移都会在保存时校验。未知类型直接报错，
  不会静默跳过。
- 时间平移产生的边界填充值会自动标为 action invalid。
- 原始数据不会修改，输出目录也禁止位于原始数据目录内部。

WAN 的未来视频窗口仍应同时携带导出的 manifest，以判断窗口是否与坏视频区间相交；
单帧 `video_valid_mask` 不能代替完整区间证据。

## 5. InternVLA 训练交接

训练配置增加按 repo 指定清洗清单：

```yaml
dataset:
  repo_id:
    - org/agibot_clean
    - org/interndata_clean
  cleaning_manifests:
    org/agibot_clean: /path/agibot/cleaning_manifest.jsonl
    org/interndata_clean: /path/interndata/cleaning_manifest.jsonl
  require_cleaning_approval: true
```

InternVLA 的数据加载器只直接选择 `training_approved` 且处置为
`keep/mask` 的回合。`correct` 和 `segment` 必须先物化成新数据集；物化包内的
`meta/cleaning_manifest.jsonl` 会把已经落盘的结果重新标成 `keep` 或 `mask`。
新增的 `action_valid_mask` 与原有
`action_mask_spec` 不是一回事：

- `action_mask_spec` 继续表示哪些动作维度做 delta、哪些保持 absolute。
- `action_valid_mask` 表示哪些时间步和维度可以贡献训练 loss。

统计量需用 InternVLA 原脚本重新计算，只增加清洗清单过滤和来源摘要。
`stats.json.cleaning.json` 保存清洗清单摘要；摘要不匹配时训练预检直接报统计量过期。

第一版训练 loss 支持按时间步和动作维度掩码；统计量脚本对”仅部分维度无效”的时间步采用
保守策略，整行不进入统计，以免坏维度污染 qstats。
