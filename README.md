# LeRobot 数据清洗工作台

![平台截图](img/platform.png)

这是一个部署在数据服务器上的浏览器工作台，用来理解、检查、复核和裁剪 LeRobot 数据。它支持 LeRobot v2.0/v2.1 与 v3.x，不绑定 44 维动作，也不绑定某一种机器人。

原始数据始终只读。自动扫描、人工结论和导出文件只写入单独的工作目录。

> 文档地图：本文是项目总览与快速上手。详细文档请见 [使用说明](docs/USAGE.md)（审核员日常操作）、[部署与配置](docs/DEPLOYMENT.md)、[开发指南](docs/DEVELOPMENT.md)、[工作流速查](使用说明.md)。

## 1. 当前交付能做什么

- 打开 v2.0/v2.1 的逐 episode 文件，以及 v3 的共享 Parquet/video shard。
- 同步播放多相机视频；左手相机优先放左侧，头部相机放中间，右手相机放右侧。
- 视频始终使用 `contain` 保留完整视野，不裁切四边；标题显示源分辨率与实际播放分辨率，
  可将任一路单独放大并恢复多视角。画面比例与面板不一致时出现黑边是正常的完整画面适配。
- 在多视角视频上方居中显示当前 prompt；有 instruction segment 时随当前帧切换，否则显示 episode task。
- v3 共享视频按 episode 懒生成浏览器兼容的 H.264 代理，避免 AV1、随机 seek 和跨 episode 时间偏移导致部分相机黑屏或卡住。
- 自动识别 `action`、`actions.*`、`state`、`states.*`、`observation.state*`，不要求固定动作维度。
- 按动作/状态、左臂/右臂、左右夹爪、头、腰、底盘分组；整组或逐维任意组合曲线。
- 鼠标悬停曲线时显示帧号、时间、指令、子任务、关键帧、人工接管标签和当前曲线值。
- 扫描文件、时间戳、帧号、数值、动作动态和视频抽帧质量；可定位的异常会返回具体帧或区间。
- 相机角色和已有 Action 约定由字段名与 metadata 自动推断；逐本体物理阈值和几何资产
  仍由机器人团队在后端配置，普通审核员不填写“数据契约”。
- 扫描任务写入 SQLite，后端重启后可恢复；失败任务最多重试三次并隔离错误 episode。
- 提供可恢复的两阶段扫描协调器：先对全量数据做无视频快扫，再对全部异常候选和
  分层 clean 抽检做视频深扫。
- 完整扫描证据写入按数据集、profile、插件和 episode 分区的不可变
  `quality-artifacts/*.json.gz`；SQLite 仅保存摘要、路径和 SHA256。相同输入、配置、
  契约和结果重复运行时复用原 run，读取时重新校验对象完整性。
- 可将全部扫描历史展开成 `runs/checks/flags/evidence` 四张分区 Parquet；Parquet 是
  查询快照，带 SHA256，并保留 stale 历史，JSON.GZ 仍是证据真值。
- 可生成原子动作候选边界，并在“指令段”页接受、驳回或按人工框选区间修订。
- 可在长轨迹的原始 instruction segment 内框选区间并标注更细的子任务；人工子任务还能
  继续递归细分，视频顶部和逐帧浮层始终优先显示当前最细的一层指令。
- 提供统一 VLM judge runner：实际抽取多相机帧，可调用本地命令或 HTTP 服务，
  并保存模型、checkpoint、prompt、输入帧、原始响应和重试摘要。
- 选择一个好区间，同步裁剪 Parquet 与多路视频，导出为单 episode 的 LeRobot v2.1 或 v3.0 数据集。

它不能仅凭数据字段判断机器人任务是否真的成功，也不能替代机器人专家确认动作单位、坐标系和控制约定。前端不再展示未经验证的“质量总分”，只展示硬错误、待复核、未配置和未运行等可追溯证据计数。

## 2. 五分钟启动

### 2.1 依赖

> 直接跳到 2.2 启动会自动安装这些依赖和环境

| 组件 | 要求 | 什么时候需要 |
|---|---|---|
| Python | 3.10+，推荐 3.11 | 必需 |
| `uv` | 推荐安装 | 快速创建环境；没有时脚本退回 `python -m venv`；也可改用 conda，见 [部署与配置](docs/DEPLOYMENT.md) |
| conda（Miniconda/Anaconda） | 可选 | 替代 `uv`/`.venv` 建环境，见 [部署与配置](docs/DEPLOYMENT.md) |
| Node.js/npm | Node 20+ | 只有 `frontend/dist` 缺失或修改前端时需要 |
| Chrome/Edge | 较新版本 | 浏览器界面和 RGB 视频播放 |
| GPU | 核心功能不需要 | 只有运行 SAM2/VLM 外部证据生成器时需要 |

Python 依赖由 `pyproject.toml` 锁定，核心包括 FastAPI、PyArrow、PyAV 和 NumPy。交付包包含已经构建好的 `frontend/dist`，通常不需要在服务器重新安装前端依赖。

可选重模块按需安装，不会下载模型权重：

```bash
uv pip install --python .venv/bin/python -e '.[geometry,segmentation,retargeting]'
```

`geometry` 安装 Pinocchio/OpenCV，`segmentation` 安装 `ruptures`，
`retargeting` 安装 dex-retargeting。SAM2 需使用官方仓库及用户已有 checkpoint，
工作台不会自动下载。

多机扫描另安装 Ray：

```bash
uv pip install --python .venv/bin/python -e '.[cluster]'
```

v3 视频代理由 PyAV/FFmpeg 在 CPU 上转成 H.264，只在第一次打开某个
`episode + camera` 时运行，之后直接命中缓存。它不会修改或复制回原始数据集。

### 2.2 启动命令

```bash
cd frontend && npm run build && cd .. \
LEROBOT_ALLOWED_ROOTS=/cpfs01/Embodied/datasets \
LEROBOT_STUDIO_PORT=7860 \
./scripts/run_local.sh
```

也可以使用配置文件：

```bash
cp .env.example .env
# 修改 .env 中的 LEROBOT_ALLOWED_ROOTS
./scripts/run_local.sh
```

脚本会：

1. 创建 `.venv`。
2. 安装缺少的 Python 包。
3. 仅在静态前端不存在时安装 npm 依赖并构建。
4. 默认监听 `127.0.0.1:7860`。

确认服务：

```bash
curl -fsS http://127.0.0.1:7860/api/health
```

## 3. 从本机访问隔离服务器

用 SSH 转发服务器回环端口：

```bash
ssh -N -L 7860:127.0.0.1:7860 user@your-server
```

本机打开 `http://127.0.0.1:7860`。VSCode Remote SSH 的“端口转发”也可以完成同一件事。

不要直接把服务暴露到公网。当前版本没有用户登录系统，默认安全边界是服务器回环地址和 SSH。

## 4. 已知限制

1. 语义分组依赖 metadata 名称；字段名含糊时会落入“其他动作/状态”。
2. 高维动作虽然能显示，但几百条曲线同时叠加没有解释价值，应使用组开关缩小范围。
3. 物理限位、动作语义和夹爪维度依赖 `cleaning_contract.yaml`；未填写时明确显示“未配置”。
4. 网页默认视频质量检查是少量时间点抽样；批量协调器的第二阶段才默认顺序逐帧解码。
5. 工作台能消费 VLM/SAM2 sidecar，并能保存 SAM2 正负点和外接框修订；不自带或下载权重，
   真正的视频传播须在已有 checkpoint 的 GPU 环境运行，结果仍需人工确认。
6. 跨数据集大规模索引质量和对训练收益的验证仍未完成。
7. 命令行 materializer 可批量生成 segment ZIP；尚未把多个片段合并成一个共享 shard 数据集。
8. v3 浏览代理缓存尚未自动设容量上限，需要按服务器磁盘情况定期清理。

最稳妥的生产流程仍是：结构规则找确定性错误，统计规则缩小复核范围，机器人专家确认语义，再由清单或已验证片段驱动训练。

## 5. 更多文档

| 文档 | 面向 | 内容 |
|---|---|---|
| [docs/USAGE.md](docs/USAGE.md) | 数据审核员 | 打开数据集、格式支持、审核操作全流程、片段导出、扫描规则、清洗清单 |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | 部署负责人 | conda 安装、环境变量、扫描器内部配置、交付包生成、目录清理 |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | 开发者 | 代码目录、本地开发与测试、VLM 验证、按需物化、InternVLA 训练交接 |
| [使用说明.md](使用说明.md) | 数据审核员 | 「审核 → 暂存 → 通过 → 导出」四步工作流速查 |
