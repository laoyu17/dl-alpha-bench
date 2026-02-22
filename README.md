# dl-alpha-bench

金融专用、可防数据泄露的深度学习量化实验框架（非通用 DL 框架）。

## Why this project

`dl-alpha-bench` 面向量化研究实习场景，突出三件事：

1. Purged/Embargo Cross Validation（防时间泄露）
2. Walk-forward 训练验证（滚动窗口）
3. 可复现实验（YAML 配置 + seed + 实验追踪）
4. 聚宽/米筐适配层（限频 + 重试 + 错误分类）
5. 高频微观结构特征（盘口不平衡、成交强度、短窗波动等）

## Core modules

- `data`: 数据契约、连接器、企业行动接口
- `dataset`: 特征/标签/mask 对齐构建
- `cv`: PurgedKFold + Embargo + Walk-forward
- `train`: 轻量 MLP 训练与 seed 控制
- `eval`: IC / RankIC / 统计指标
- `backtest`: 轻量事件驱动回测摘要
- `exp`: 配置化实验运行与追踪
- `gui`: PyQt6 多页面研究工作台

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
pytest
```

运行一个配置化实验：

```bash
dl-alpha-bench --config configs/experiment_sample.yaml

# 高频微观结构特征示例（mock）
dl-alpha-bench --config configs/experiment_microstructure_mock.yaml

# 联机 real 配置（需要本地凭证与 SDK）
dl-alpha-bench --config configs/experiment_joinquant_real.yaml
dl-alpha-bench --config configs/experiment_ricequant_real.yaml
```

启动 GUI（需要安装可选依赖）：

```bash
pip install -e .[gui,plot]
dl-alpha-bench-gui
```

## Repo

- GitHub: https://github.com/laoyu17/dl-alpha-bench.git

## Documentation index

- `docs/01-requirements.md`
- `docs/02-architecture.md`
- `docs/03-data-contract.md`
- `docs/04-leakage-prevention.md`
- `docs/05-experiment-reproducibility.md`
- `docs/06-gui-prd.md`
- `docs/07-dev-plan.md`
- `docs/08-test-plan.md`
- `docs/09-contributing.md`
- `docs/10-release-checklist.md`

## Provider notes

- JoinQuant: 使用 `jqdatasdk`，读取 `.env` 的账号密码与限频/重试参数
- RiceQuant: 使用 `rqdatac`，支持 token 或 user/password 初始化
- 连接器在请求层内置了限频、指数退避重试和错误分类（认证、限频、临时故障）

## Runtime guards

- 默认 `runtime.fail_on_leakage=true`：若 split 出现泄露风险，会写入 blocked artifact 并中止训练
- 可通过 `runtime.validate_config_only=true` 先做配置/数据契约预检查（不训练）
- 可通过 `runtime.allow_offline_mock_fallback=true` 在在线连接器失败时降级到本地 mock 数据
- 布尔配置统一支持 `true/false/1/0`；非法取值会被视为配置错误并阻断运行

## Explainability panel

- GUI 新增 `Explainability` 面板，按因子展示：`IC/RankIC/IC_IR/RankIC_IR/分位收益贡献`
- 实验会在 `artifacts/<exp_id>/runs/<run_id>/` 输出：
  - `result.json`（包含 `feature_explainability` 字段）
  - `feature_explainability.csv`（面试展示可直接打开）
  - `config.yaml`（本次运行配置快照）
- `artifacts/<exp_id>/latest_run.txt` 记录最近一次运行的 `run_id`
