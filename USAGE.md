# 使用说明 · 审核与导出全流程

> 返回 [README](../README.md) · 同类文档：[使用说明](USAGE.md) · [部署与配置](DEPLOYMENT.md) · [开发指南](DEVELOPMENT.md)

## 1. 打开一个数据集

网页中输入包含以下文件的真实服务器目录：

```text
/path/to/dataset/meta/info.json
```

允许打开的路径必须位于 `LEROBOT_ALLOWED_ROOTS` 内。多个根目录在 Linux/macOS 上用冒号分隔：

```bash
export LEROBOT_ALLOWED_ROOTS=/cpfs01/Embodied/datasets:/another/read_only_root
```

打开数据集只建立 episode metadata 索引，不会一次读取全部 Parquet 或视频。视频通过 HTTP Range 按需播放。

### 3.1 v3 视频首次打开为什么会稍慢

LeRobot v3 会把多个 episode 拼在同一个视频 shard 中，开源数据还经常使用
AV1。浏览器直接在共享 shard 中随机 seek 时，容易出现某一路 hand view
黑屏、不同相机起始时间不一致，或播放完后无法重新开始。

本工具对 v3 使用按需浏览代理：

1. 根据 `meta/episodes/*.parquet` 中每路相机的 `from_timestamp` /
   `to_timestamp` 定位当前 episode。
2. 从共享 shard 解码当前 episode 的帧。
3. 从 0 重建视频时间戳并编码成浏览器兼容的 H.264。
4. 写入 `${LEROBOT_STUDIO_WORKDIR}/video-proxy/`，后续访问直接复用。

代理不会为了满足 H.264 的偶数宽高要求而缩小源图。遇到奇数宽度或高度时，
只在右侧或底部补 1 像素黑边，所有原始像素都保留。标题中的“源”和“播放”
分辨率可用于现场核对；两者只差 1 像素时即为这种编码补边，不是裁剪。

三路相机可以并行生成。短 episode 通常等待数秒；具体速度取决于源编码、
分辨率、episode 长度和服务器 CPU。缓存是可删除的派生文件，停服后可以清理：

```bash
rm -rf "${LEROBOT_STUDIO_WORKDIR}/video-proxy"
```

再次打开时会自动重建。大规模多人复核时应把 `LEROBOT_STUDIO_WORKDIR`
放在容量可控的高速盘，并定期清理不再使用的数据集缓存。

## 2. 支持的数据格式

### 4.1 LeRobot v2.0/v2.1

典型结构：

```text
meta/info.json
meta/episodes.jsonl
meta/tasks.jsonl
data/chunk-000/episode_000000.parquet
videos/chunk-000/<video_key>/episode_000000.mp4
```

每个 episode 使用独立 Parquet 和视频文件。工具按 `chunks_size`、`data_path` 和 `video_path` 解析，缺少模板时使用 LeRobot v2 的常见默认值。

### 4.2 LeRobot v3.x

典型结构：

```text
meta/info.json
meta/episodes/chunk-000/file-000.parquet
meta/tasks.parquet
data/chunk-000/file-000.parquet
videos/<video_key>/chunk-000/file-000.mp4
```

一个 data/video shard 可以包含多个 episode。工具读取：

- `episode_index`
- `dataset_from_index` / `dataset_to_index`
- `data/chunk_index` / `data/file_index`
- 每个视频的 `from_timestamp` / `to_timestamp`

因此不会假设“一个 v3 文件就是一个 episode”。

`features.<camera>` 的视频元信息同时兼容官方出现过的 `video_info` 和
`info` 两种写法。源文件即使是 AV1，浏览器实际播放的也是上述逐 episode
H.264 代理；自动扫描和导出仍然读取原始视频。

### 4.3 任意动作维度和拆分字段

识别规则基于字段家族，而不是固定维度：

- 动作：`action`、`action.*`、`actions.*`
- 状态：`state`、`states.*`、`observation.state`、`observation.state.*`

每个字段的维度名优先使用 `features.<key>.names`，其次解析 `field_descriptions`，最后才显示 `<field>[i]`。动作/状态字段最多允许 2048 维进入检查和曲线；普通数值字段仍限制为 128 维，以免把视觉 embedding 误当控制曲线。

左右臂等分组来自字段名和维度名的可解释规则。分组只是浏览器组织方式，不会修改原始 tensor，也不会偷偷调整维度顺序。

## 3. 推荐操作流程

### 5.1 先看结构

打开后先检查左侧“概览”：

- 版本、robot type、episode/frame 数量。
- feature 和相机 key。
- Parquet、视频存在性抽检。

结构检查通过只代表文件能找到，不代表动作语义已确认。

### 5.2 看视频和动作

- 多相机使用同一个相对 episode 时间同步。
- 播放器使用 `object-fit: contain` 展示完整画面，不做中心裁剪。三路并排时单路会缩小，
  点击相机标题右侧的放大图标可独占视频区域，再点击恢复多视角。
- 每路标题显示“源 宽×高 · 播放 宽×高 · 完整视野”；如果源图本身已被上游裁过，
  工具只能如实显示，不能恢复源文件中不存在的视野。
- 视频区最上方显示当前 prompt。进入新的 instruction segment 时会随帧更新，
  没有分段标注时显示 episode 的 task。
- 超过四路相机时，顶部可选择最多四路。
- 折线图只绘制 action；state 不参与绘图，避免双份控制量把曲线淹没。
- 曲线右侧按身体部位分组 action。
- 点击组标题可整组显示/隐藏；每个维度仍可独立勾选。
- “双臂”和“双夹爪”是常用组合，其余组合可直接勾选任意组和维度。
- “叠加”适合看相关性，“分轨”适合看每维细节。
- 播放到 episode 末尾会停在最后一帧；再次点击播放会自动回到第 0 帧，
  不需要手动拖动进度条。

### 5.3 看逐帧标签

曲线悬停不改变播放位置；单击才会跳到并锁定该帧。浮层会停在点击位置附近并自动避开画布边缘，之后不再跟随鼠标，下一次单击才同时更新帧和位置。长 instruction 会自动换行，内容超过画布高度时可在浮层内滚动。浮层显示：

- 原始帧号和相对时间。
- episode 指令和当前 instruction segment。
- `language_memory`/skill、track。
- Task Frame、Success Frame、人工接管等标签。
- 悬停时显示少量 action 值；锁定后显示该帧完整 state，action 值继续在右侧列表中查看。

右侧“当前帧”显示该采样点所有返回字段和生效标签。“指令段”中的条目可以定位到起点，剪刀按钮会直接选择该段作为导出区间。

#### 标注长轨迹中的细粒度子任务

1. 打开右侧“指令段”，点击原始指令段右侧的 `+`；也可直接在动作图或底部时间轴选区。
2. 把范围收窄到该动作真正发生的 `[起始帧, 结束帧)`，填写子任务指令和审核人后保存。
3. 已保存条目右侧的 `+` 表示“继续细分”，铅笔表示修改，垃圾桶表示删除。
4. 同一父任务下的区间不能重叠；子任务必须严格位于父任务和原始指令段内部。存在子节点时，
   必须先删子节点再删父节点，避免误删整棵标注树。

选定非全回合区间后，底部播放按钮和空格键都会只播放该区间。播放到结束帧自动暂停；
当前帧在区间外或已经到达结尾时，再次播放会从区间起点开始。点击“选择整个 episode”
即可恢复全局播放。

这套标注独立存入工作目录的 SQLite，不修改 LeRobot 原始数据，也不复用人工清洗中的
“好段 `segments`”。顶部全量清单导出会生成 `subtask_annotations.jsonl`，保留父子关系、
原始指令段编号、帧范围、审核人和更新时间，供后续生成训练用单指令 episode 或层级监督。

“已标注子任务”右上角的“导出完整回合”与底部剪刀的单片段导出不同：它会遍历所有
包含人工子任务标注的源 episode，每个源 episode 仍只生成一个完整的新 episode，原有
视频、状态、动作和全部帧都保留。父子标注会展平成无重叠 instruction segments：最深层
子指令优先，子段以外继续使用父指令或原始指令，因此不会因为标注不完全而丢帧。
可选择 v2.1 或 v3.0，输出 ZIP 包含完整 `meta/`、`data/`、`videos/` 和
`meta/subtask_provenance.jsonl`，可以作为独立派生数据集解压使用。

批量导出始终写入 `${LEROBOT_STUDIO_WORKDIR}/exports` 后再提供下载。后端明确禁止把输出
写到源数据目录或其子目录；导出 provenance 中固定记录 `source_modified: false`。源数据
即使以只读方式挂载也能正常使用。

### 5.4 自动扫描

打开一个 episode 后，工作台会自动对该 episode 启动一次扫描；同一浏览器会话中不会
因来回切换而重复扫描。已有扫描正在运行时，新打开的 episode 会等待该任务结束后再扫。
顶部“自动质检”仍保留，供负责人批量扫描当前列表或全量数据。

选择扫描范围：当前 episode、当前已加载列表或全量。可以关闭视频抽帧，只检查 metadata、Parquet 和动作。

自动扫描给出三种**建议**，但它们不是最终处置：

- `自动保留`：当前规则没有发现硬错误或软异常。
- `自动复核`：存在统计或抽样风险，需要人工判断。
- `自动排除`：存在缺文件、无法解析、时间戳错误等硬错误；人工仍可记录证据后推翻。

每个检查项另有六种明确状态：`pass`、`fail`、`needs_review`、
`not_configured`、`not_applicable`、`not_run`。后四者都不能冒充“通过”。

有帧证据的规则是可点击的：单点异常会跳到对应帧；连续异常会同时设置裁剪区间。标签会明确标注“全量规则”或“抽样证据”。

扫描任务不是浏览器内存任务。队列、每个 episode 的 attempt、错误和最新 run id
写入 `${LEROBOT_STUDIO_WORKDIR}/studio.sqlite3`；杀掉服务后重新打开同一数据集会续跑。

### 5.4.1 推荐的全量快扫和候选深扫

```bash
.venv/bin/python scripts/run_adaptive_scan.py DATASET \
  --server http://127.0.0.1:7860 \
  --audit-fraction 0.02 --audit-minimum 20 \
  --deep-video-mode sequential --deep-video-stride 1 \
  --state audits/adaptive-scan.json
```

`review/drop` 候选全部进入第二阶段，自动保留样本按时长和任务分层抽检。state 文件
记录两阶段 job ID、数据指纹、选择规则、视频扫描模式和最终状态；脚本或连接中断后
使用同一命令续跑。第二阶段默认顺序解码每路 RGB 的每一帧；可用
`--deep-video-stride N` 每隔 N 帧分析一次以降低成本。也可显式选择
`--deep-video-mode sampled --deep-video-samples 64` 做固定数量抽样，但不能把抽样
结果表述成整段视频完全检查。每次结果都会记录实际分析帧数和估算覆盖率。

CLI 将进度 JSONL 写到 stderr，stdout 只输出最终的单个 JSON，因此可安全重定向：

```bash
... > adaptive-final.json 2> adaptive-progress.jsonl
```

本机 22 MB AgiBot v2.1 真实样本的 process backend 验收中，快扫约 0.83 秒，六路
RGB、每路 2089 帧的顺序深扫约 27.3 秒；六路均达到 `2089/2089`、覆盖率 `1.0`、
零解码错误。数据声明但缺失的 `head_depth` 被独立报告为硬错误。同一 state 重跑没有
创建新任务。该结果证明功能正确，不代表 10 TB CPFS 的吞吐已经通过。

### 5.5 人工复核

人工决定和备注写入：

```text
${LEROBOT_STUDIO_WORKDIR}/studio.sqlite3
```

不会写入数据集目录。自动建议默认不覆盖人工决定。

当前页面不提供建队列、领取、仲裁或分割提示。负责人先用 episode 编号范围或外部表格
分配批次，审核员打开自己的 episode，完成六项观察题、必要的好/坏区间、原因和备注，
最后只点击一次“保存最新标记”。保存后页面会重新读取服务端最新记录。

多人的双审、抽检和分歧处理目前由负责人在线下汇总，不应向外包宣称页面已经自动完成。
仓库中保留的高级审核 API 和实验脚本不属于本轮默认界面，启用前必须另做流程验收。

播放快捷键：

- `Space`：播放/暂停。
- `↑` / `↓`：上一条/下一条 episode。
- `←` / `→`：前后移动一帧。

### 5.6 选择和导出好区间

底部浅蓝区域是当前导出范围，两个小滑块分别控制起点和终点。四个按钮依次是：

- 当前帧设为起点。
- 当前帧设为终点；结束帧采用右开区间。
- 恢复整个 episode。
- 打开片段导出。

也可以在 action 折线图上按住左键横向拖动，矩形框选一个子区间。松开后会像 Plotly 一样放大到该时间窗口、按窗口内 action 重新计算纵轴量程，同时更新导出范围并同步高亮底部双端轨道。工具栏的缩小图标或 `Esc` 可随时恢复完整时间轴，但不会清除已经选定的导出区间；短按不拖动仍表示锁定当前帧。

导出窗口可选择 v2.1 或 v3.0，并可修改片段任务指令。导出 ZIP 解压后直接包含 `meta/`、`data/` 和 `videos/`。

当前选定的剪裁区间、任务指令和导出格式会防抖“暂存”到工作目录
（`${LEROBOT_STUDIO_WORKDIR}/studio.sqlite3` 的 `staged_edits` 表），切换 episode 或
关掉页面后重新打开会自动恢复；这只写入工作目录，不影响原始数据。

批量清洗完成后，人工把 episode 标记为“通过”（处置为 keep / correct / mask /
segment）后，点右上角“导出”打开“统一导出已通过项”对话框：可选 LeRobot v2.1 /
v3.0 格式，并填写服务器输出路径。后端会把所有通过项按各自暂存的剪裁区间批量生成到
`<输出目录>/episode_XXXXXX/` 子目录（未选区间则导出完整回合），并写一份
`export_manifest.json` 记录来源与区间；每个子目录都是可单独打开的 LeRobot 数据集，
原始数据全程只读。

## 4. 片段导出到底做了什么

给定 `[start_frame, end_frame)`，导出会：

1. 裁剪该 episode 的全部 Parquet 列。
2. 把 `episode_index`、`task_index` 重排为 0。
3. 把 `frame_index` 和全局 `index` 重排为 `0..N-1`。
4. 从 0 重置 timestamp，同时保留原采样间隔。
5. 按同一时间边界解码和重编码每路视频，严格检查输出帧数。
6. 截断并平移 instruction segments、成功帧和归一化 frame tags。
7. 重算数值字段统计。
8. 写入 `meta/clip_provenance.json`，记录源数据路径、fingerprint、episode 和原始帧区间。

如果源数据的 `meta/info.json` 声明了某路视频，但对应文件在磁盘上不存在，裁剪导出不会让整段任务失败：工具只导出实际存在的视频流，从新数据集的 `features` 中删除缺失流，并在 `clip_provenance.json` 的 `omitted_video_streams` 中记录 key、原始相对路径和原因。这个行为只是对缺失媒体做显式降级，不会补帧、伪造深度或修改原始数据。

### v2.1 输出

```text
meta/info.json
meta/episodes.jsonl
meta/tasks.jsonl
meta/stats.json
meta/episodes_stats.jsonl
meta/episodes_detailed_task.jsonl
meta/clip_provenance.json
data/chunk-000/episode_000000.parquet
videos/chunk-000/<video_key>/episode_000000.mp4
```

### v3.0 输出

```text
meta/info.json
meta/episodes/chunk-000/file-000.parquet
meta/tasks.parquet
meta/stats.json
meta/episodes_detailed_task.jsonl
meta/clip_provenance.json
data/chunk-000/file-000.parquet
videos/<video_key>/chunk-000/file-000.mp4
```

普通 RGB 视频输出为 MPEG-4/MP4。depth 视频（PNG 编码的 16-bit 深度）使用流复制（stream copy）保持 PNG 原样，不重编码、不丢失深度精度；若某路视频源文件损坏无法解码，该路会被降级省略并在 `clip_provenance.json` 的 `omitted_video_streams` 中记录，不会让整段导出失败。正式训练前仍应使用目标版本的 LeRobot loader 做一次单 batch 验收，因为不同 LeRobot 发布版对 metadata 字段会继续演化。

当前一次只导出一个 episode 的一个区间。它适合人工挑选、建立小型训练集和验证数据管线，不是全量 TB 级 materialize 工具。

## 5. 自动扫描有哪些规则

| 检查 | 数据范围 | 能定位 | 可信边界 |
|---|---|---|---|
| Parquet/视频存在、能否解码 | 全量文件或抽样帧 | 文件级 | 确定性较强 |
| episode length 与 Parquet 行数 | 全量 | episode | 确定性较强 |
| timestamp 有限、递增、抖动 | 全量帧 | 首个异常帧 | 不判断控制延迟语义 |
| frame index 连续性 | 全量帧 | 首个 gap | 确定性较强 |
| action/state NaN/Inf | 全量帧和维度 | 帧和维度 | 确定性较强 |
| action 静止、重复 | 全量动作 | 最长连续区间 | 阈值需按机器人校准 |
| 速度、加速度、jerk 异常 | 全量动作 | 帧和主导维度 | 使用鲁棒尺度识别候选异常，不做训练归一化，也不等于物理越界 |
| action/state lag 和时钟漂移 | 按动作语义、按滑窗 | episode/窗口区间 | 需配置 action→state 维度和标定阈值 |
| FK/记录 EEF | 配置后抽样 | 帧 | 必须有经确认的 URDF、关节名、TCP |
| 机器人重投影 | 配置后抽样 | 帧/相机 | 必须有 FK、相机标定和视觉关键点 |
| 原子动作候选 | 全量 action + 已配置接触字段 | 边界/区间 | 规则与 ruptures 只提案；持续接触中的边界会警告，必须人工确认 |
| VLM 指令核验 | 外部 sidecar | 区间 | 至少两个独立 judge，只送人工 |
| SAM2 跟踪 | 外部 sidecar | 漂移帧 | 只生成视觉证据，不自动删除 |
| 仿真回放 | 外部原生适配器 | 回合 | state-forced 不能证明 action 可执行 |
| 黑屏、模糊、冻结 | 默认每路 7 个时间点 | 抽样帧 | 高召回筛查，不能代表整段比例 |
| instruction segment 越界/重叠 | 全量 metadata | 区间 | 双臂并行 track 可能是合法重叠 |
| takeover 存在 | metadata | 有帧标注时可定位 | 不自动判断该段应删除还是作为恢复数据保留 |

RGB 黑屏、模糊和冻结规则不会应用到声明了 `video.is_depth_map` 的视频。没有校准过的新机器人，建议先抽查 50 至 100 条，再决定是否批量应用自动建议。

## 6. 我们对两组真实数据理解到哪一层

### 8.1 InternData A1/A2D 小样本

本机已验证同一真实 episode 的 v2.1 和 v3 两种格式：

```text
/Users/equation/Documents/具身/tmp/real_dataset_smoke_20260714/interndata_v21_episode0_fixture
/Users/equation/Documents/具身/tmp/real_dataset_smoke_20260714/interndata_v30_episode0_fixture
```

已实读确认：

| 项目 | 内容 |
|---|---|
| robot type | `a2d` |
| 帧数/频率 | 591 帧，30 FPS |
| RGB 相机 | `images.rgb.hand_left`、`images.rgb.head`、`images.rgb.hand_right` |
| 动作总维度 | 20 |
| 状态总维度 | 27 |
| v2.1/v3 | 两种都能打开、画曲线和扫描 |

动作不是一个扁平 `action`，而是：

| 字段 | 维度 | metadata 名称 |
|---|---:|---|
| `actions.effector.position` | 2 | `left_gripper`, `right_gripper` |
| `actions.head.position` | 2 | `yaw`, `patch` |
| `actions.joint.position` | 14 | 左臂 7 关节 + 右臂 7 关节 |
| `actions.waist.position` | 2 | `pitch`, `lift` |

状态在对应 20 维之外，还包含：

| 字段 | 维度 | metadata 名称 |
|---|---:|---|
| `states.robot.orientation` | 4 | `x`, `y`, `z`, `w` |
| `states.robot.position` | 3 | `x`, `y`, `z` |

注意：`head.position` 的第二个名字在源 metadata 中确实写作 `patch`，工具原样显示，没有擅自改成 `pitch`。坐标系、单位、绝对/增量控制和 gripper 数值方向仍未从这个小样本得到官方确认。

修复后的 `policy` 扫描会把四组 `actions.*` 合并为 20 维动作，不再误报 `missing_action`；这两份样例当前未触发规则异常。

### 8.2 AgiBotWorld2026 小样本

已验证的本地原始 v2.1 样例：

```text
/Users/equation/Documents/具身/tmp/real_dataset_smoke_20260714/agibot_extracted/data
```

已实读确认：

| 项目 | 内容 |
|---|---|
| robot type | `g2a` |
| episode | 1 条，2089 帧 |
| 原生 action | float32，44 维 |
| 原生 state | float32，169 维 |
| 视频 | 7 路，包括左右手、top head、fisheye 和 depth |
| 标注 | episode task、instruction segment、Task Frame/Success Frame 类关键帧 |

44 维 action 的 `field_descriptions` 能拆出左右 effector、末端位姿、14 个关节、头、腰和机器人速度。曲线区只绘制 action；169 维 state 只在锁定当前帧后作为背景信息显示，避免控制曲线被状态信号淹没。

但以下问题仍未被样例充分回答：

- 每个字段的物理单位。
- action 是绝对目标、增量目标还是控制器处理后的命令。
- 世界、基座、末端坐标系定义。
- 44 维中训练时实际使用的 action mask。
- 成功帧和 episode success 的官方语义是否完全等价。

因此工作台能“正确读取和呈现 schema”，不能声称“已经知道如何把 44 维动作直接下发到真机”。

## 7. 清洗清单导出

右上角“导出”对话框里提供“清洗清单”链接，下载的是全数据集人工/自动决定清单，
不复制视频和 Parquet：

```text
dataset_contract.yaml
cleaning_manifest.jsonl
cleaning_manifest.csv
approved_episodes_by_repo.json
segments.jsonl
subtask_annotations.jsonl
validity_masks.jsonl
corrections.jsonl
leakage_group_overrides.template.jsonl
training_handoff.yaml
summary.json
```

兼容旧流程的 episode index 文本仍会保留，但正式训练应读取
`training_handoff.yaml`。对于 policy 数据，`approved_episodes_by_repo.json` 和
`cleaning_manifest.jsonl` 只表示“清洗层批准”，还没有通过无泄漏 split gate；
handoff 会明确标成 `blocked_pending_leakage_split`。填完导出包中的 scene/object
模板并生成 `cleaning_manifest.train.jsonl` 后，InternVLA 才能在严格模式训练。
不能只依赖旧版三分类文件，也不能把“清洗批准”误当“最终 train split 批准”。

这和底部剪刀的“片段数据导出”是两件事：前者给全量 materialize 或训练采样程序使用，后者直接生成一个新的单 episode LeRobot 数据集。

## 8. 哪些自动，哪些人工

| 检查 | 默认执行者 | 说明 |
|---|---|---|
| 图像、状态、动作、时间戳、指令 | 工作台自动 | 检查必需字段、文件和长度 |
| 回合长度、动作方差、静止和重复值 | 工作台自动 | 按任务分布预筛，不把异常直接解释成失败 |
| NaN/Inf、逐维物理范围、速度/加速度/Jerk | 工作台自动 | 没有逐本体阈值时明确显示“未配置”，不得显示通过 |
| 状态、动作、视频时序 | 工作台自动 | 检查时间戳、帧号、长度、时滞和方向一致性 |
| 多视角解码、黑屏、模糊、冻结、帧数和时长 | 工作台自动 | 视频内容检查可抽样，结构和时长全量检查 |
| 夹爪变化和切分边界 | 工作台自动 + 人工 | 机器定位事件，人确认接触上下文是否完整 |
| 动作语义、控制频率和预测长度 | 工作台自动核对配置 | 物理含义必须由机器人团队提供 |
| 指令段和统一导出格式 | 工作台自动 | 检查空值、越界、重叠和成功帧；导出时转换 |
| 指令与行为、动作结果是否匹配 | 人工 | 外包只判断可观察事实 |
| 坐标系、TCP、相机约定 | 工作台自动 + robot 团队 | 未确认时不能批准训练 |
| FK/末端位姿和机器人重投影 | 同事几何模块 + robot 团队 | 接入 Unify Robot Actions in Camera Frame 后才正式验收 |
| 四元组、动作细节、物理反馈和文字原则 | 人工 | 只用于世界/人类视频；Policy 数据不强制长描述 |
| VLM 长短双粒度描述 | 暂缓 | 当前不计入训练批准条件 |
| 归一化和动作重排 | InternVLA | 工作台绝不重复实现 |

审核最终处置为：

- `keep`：原样保留。
- `correct`：记录确定性修正后保留。
- `mask`：无效时间或维度不参与 loss。
- `segment`：只保留指定子区间。
- `discard`：排除。
- `pending`：尚未做最终决定，不进入批准训练清单。

网页选中的时间区间可以直接登记为掩码或有效片段。导出 ZIP 会包含契约、逐回合清单、
掩码、片段、修正历史、清洗层批准 episode、泄漏分组待填模板和 InternVLA 交接文件。
