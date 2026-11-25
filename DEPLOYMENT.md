# 部署与配置

> 返回 [README](../README.md) · 同类文档：[使用说明](USAGE.md) · [部署与配置](DEPLOYMENT.md) · [开发指南](DEVELOPMENT.md)

## 1. 使用 conda 安装（可选）

`run_local.sh` 固定使用 uv 创建的 `.venv`（uv 优先，无 uv 时退回
`python -m venv`）。如果本机已有 Miniconda/Anaconda，也可以完全不依赖
`.venv`，手动用 conda 环境运行：

```bash
# 1. 建环境（Python 3.11，与 uv 默认一致）
conda create -y -n lerobot-studio python=3.11

# 2. 激活
conda activate lerobot-studio

# 3. 安装项目（含 dev 依赖）
pip install -e '.[dev]'

# 4. 手动启动（不经过 run_local.sh，直接用 conda 的 Python）
LEROBOT_ALLOWED_ROOTS=/cpfs01/Embodied/datasets \
LEROBOT_STUDIO_PORT=7860 \
python -m uvicorn backend.app:app --host 127.0.0.1 --port 7860
```

用 conda 时不要混用 `.venv`：项目根目录是否存在 `.venv`、系统是否装有 `uv`
都不影响上面的手动方式，只要用 conda 的 Python 直接启动即可。

激活环境后，可选重模块和多机扫描直接：

```bash
pip install -e '.[geometry,segmentation,retargeting]'
pip install -e '.[cluster]'
```

开发模式与测试同样使用 conda 的 Python：

```bash
conda activate lerobot-studio
LEROBOT_ALLOWED_ROOTS=/path/to/datasets uvicorn backend.app:app --reload --port 8000

# 测试：激活后直接 python，或不经激活
conda run -n lerobot-studio python -m pytest -q
```

注意：`pyproject.toml` 的 `readme` 字段必须指向仓库中真实存在的文件
（当前为 `README.md`）。如果安装时报 `Readme file does not exist:
README_CN.md`，说明该字段仍指向旧文件名，把它改成 `README.md` 后重试即可。
该问题同样影响 `uv pip install -e .`，与是否使用 conda 无关。

## 2. 环境变量

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `LEROBOT_ALLOWED_ROOTS` | home、项目父目录、`/tmp` | 可读取的数据根目录 |
| `LEROBOT_EXPORT_ROOTS` | `LEROBOT_ALLOWED_ROOTS` + `workdir/exports` | 右上角统一导出允许写入的服务器根目录；禁止写回源数据目录 |
| `LEROBOT_STUDIO_WORKDIR` | `~/.lerobot-cleaning-studio` | SQLite、不可变扫描对象、导出 ZIP 和 v3 浏览代理缓存 |
| `LEROBOT_STUDIO_HOST` | `127.0.0.1` | 服务监听地址 |
| `LEROBOT_STUDIO_PORT` | `7860` | 服务端口 |
| `LEROBOT_SCAN_WORKERS` | `4` | 同时扫描的 episode 数 |
| `LEROBOT_SCAN_EXECUTION_BACKEND` | `thread` | `thread` 低开销；`process` 可硬终止卡死任务 |
| `LEROBOT_SCAN_EPISODE_TIMEOUT_SECONDS` | `900` | 隔离进程模式下单 episode 超时 |
| `LEROBOT_SCAN_RESOURCE_CLASS` | `cpu-video` | 写入扫描 provenance 的资源类别 |
| `LEROBOT_RAY_ADDRESS` | 空 | Ray 集群地址；只在 backend=`ray` 时使用 |
| `LEROBOT_RAY_NUM_CPUS_PER_TASK` | `1` | 每个 episode Ray 任务声明的 CPU |
| `LEROBOT_RAY_NUM_GPUS_PER_TASK` | `0` | 每个 episode Ray 任务声明的 GPU |
| `LEROBOT_RAY_CUSTOM_RESOURCES` | `{}` | Ray 自定义资源 JSON，例如 `{"video_decode":1}` |
| `LEROBOT_MAX_CHART_POINTS` | `1800` | 每个字段返回浏览器的最大时间点数 |
| `LEROBOT_FRONTEND_DIST` | `frontend/dist` | 静态前端目录 |
| `PIP_INDEX_URL` | 可选 | Python 包镜像 |
| `NPM_REGISTRY` | 可选 | npm 镜像 |

本地浏览可使用默认线程模式。单机 TB 级扫描建议设为 `process`，让坏视频或底层解码库
卡死时可以终止单个子进程，并进入现有三次重试和失败隔离账本。多机可设为 `ray`，
但所有 worker 必须能以同一路径只读访问数据，且要先在服务器验证 Ray 断连、取消和
CPFS 吞吐。网络盘建议从 4 个并发开始；视频抽帧包含随机 seek，并发过高可能更慢。

## 3. 自动扫描的内部配置

前端已经移除“数据契约”栏目。普通审核员只使用“自动检查”和“人工审核”，不需要理解
contract、schema 等内部术语。

扫描器仍需要一份内部 YAML 才能知道逐维物理范围、动作含义、相机角色以及 FK/重投影
所需资产。它由数据负责人和机器人团队在线下维护，默认按 metadata 推断并保存在：

```text
$LEROBOT_STUDIO_WORKDIR/contracts/<dataset_id>.yaml
```

不会写回原始数据目录。如果数据集根目录本来就有 `cleaning_contract.yaml`，工作台只读；
内部版本仍按 SHA256 存档，配置变化会让旧扫描和旧阈值失效。前端不再提供修改入口，
避免外包人员误填动作单位、坐标系或关节含义。需要变更时由负责人通过受控 API 或在
workdir 更新 YAML，再重新扫描。

策略动作数据最少要确认：

```yaml
dataset:
  profile: policy
  source_type: real_robot
  robot_type: g2a
  embodiment: agibot_g2
  arm_configuration: dual_arm
action:
  representation: joint
  command: absolute
  units: radian
  coordinate_frame: robot_base
  control_frequency_hz: 30
  prediction_horizon: 50
  gripper_dimensions: [left_gripper, right_gripper]
internvla:
  robot_type: g2a
  dataset_schema: g2a_dualarm_joint
  action_mode: abs
  normalization_owner: InternVLA
checks:
  atomic_segmentation:
    minimum_segment_seconds: 0.5
    context_seconds: 0.3
    velocity_minimum_quantile: 0.2
    gripper_transition_threshold: 0.5
    change_window_seconds: 0.5
    change_z_threshold: 3.0
    contact_event_fields:
      - name: left_contact_force
        field: observation.contact.left
        dimensions: [0, 1, 2]
        threshold: 2.0
        active_when: greater_equal
        minimum_active_frames: 3
  data_forward_dynamics_predictability:
    horizon_seconds: 0.1
    action_lag_seconds: 0.0
    train_fraction: 0.7
    ridge_alpha: 0.001
    permutation_trials: 32
    random_seed: 20260730
    minimum_samples: 120
    minimum_target_std: 0.0001
    minimum_action_improvement_ratio: 0.05
    minimum_shuffle_degradation_ratio: 0.05
    maximum_permutation_p_value: 0.1
```

`data_forward_dynamics_predictability` 不需要 VLA/WAM checkpoint。它对每个已经声明
`dimensions ↔ state_dimensions` 的动作组件执行：

1. 用时间前段拟合一个小型岭回归探针，时间后段留作测试，中间留出
   `action_lag + horizon` 的隔离区，避免相邻未来帧泄漏。
2. 比较只输入当前 state 与输入当前 state + 数据集记录 action 的未来 state 误差。
3. 保持同一个已拟合探针，在测试段多次打乱 action，检查正确 action 是否显著更好。
4. 报告 action 相对 state-only 的误差改善、打乱 action 后的误差恶化和置换检验
   `p` 值。

前端“自动检查”页会按动作组件直接显示“通过”或“需复核”，并在需复核时写明是
“动作没有降低预测误差”还是“正确动作没有显著优于打乱动作”。这些结论是数据关系
预筛，不会自动删除回合。

右栏使用完整高度显示机器证据；原因、备注和最终处置编辑器只在“人工审核”标签出现。
人工清单和处置编辑器位于同一个滚动区域，因此不会再把“自动检查”压缩成一条窄栏。

这项检查验证的是”数据中的 action 是否包含可泛化的未来状态信息”，不是验证完整
机器人动力学，也不是让策略模型生成 action。轨迹太短、未来状态几乎不动时返回
`not_applicable`；缺少映射或数据集专属阈值时返回 `not_configured`。
`minimum_target_std` 使用各组件的原生状态单位，因此关节、夹爪、底盘应通过
`component_overrides` 分别配置。在线性探针上失败只进入人工/机器人团队复核，不能
直接删除数据。

除网页扫描外，也可以先对单个长回合运行：

```bash
PYTHONPATH=. .venv/bin/python scripts/check_forward_dynamics_episode.py \
  /cpfs01/Embodied/datasets/AgibotWorld2026 \
  --contract /cpfs01/Embodied/audit/agibot/cleaning_contract.yaml \
  --episode-index 0 \
  --output /cpfs01/Embodied/audit/agibot/forward-dynamics-ep0.json
```

退出码 `0` 表示通过，`2` 表示需要复核、未配置或样本不适用，`1` 表示执行失败。

## 4. 生成交付包

```bash
./scripts/package_source.sh
```

脚本排除虚拟环境、`node_modules`、SQLite、导出文件、旧压缩包和测试缓存，保留构建后的前端、后端、测试、脚本与本 README。输出包自带顶层 `lerobot_cleaning_studio/` 目录。

需要把 InternVLA 联动一起传入内网时使用：

```bash
./scripts/package_transfer_bundle.sh
```

组合包包含完整 Cleaning Studio，以及 `internvla_overlay/` 下十个训练侧文件。它不是完整
InternVLA 仓库；应覆盖到团队当前使用的同一份 InternVLA-A-series 快照，并保留内网仓库自身修改。

## 5. 目录清理

当前层是唯一源码。内层同名 `lerobot_cleaning_studio/` 是旧交付包副本，不能继续开发。
先预览清理计划：

```bash
./scripts/cleanup_workspace.sh
```

确认后执行：

```bash
./scripts/cleanup_workspace.sh --apply
```

默认保留包含真实人工审核记录的 `studio-data/` 和 `studio-data-final/`。只有额外传
`--include-deps` 才会删除可重建的 `.venv` 和 `frontend/node_modules`。后续交付包统一
写入 `dist/`，不再堆在源码根目录。
