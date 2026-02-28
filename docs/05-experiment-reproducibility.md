# 05 Reproducibility

## 机制

- `seed` 控制 Python 与 NumPy
- 实验配置 YAML 全量落盘
- 配置 hash 用于实验 ID 追踪
- 每次运行输出写入 `artifacts/<exp_id>/runs/<run_id>/result.json`
- `artifacts/<exp_id>/latest_run.txt` 记录最近一次运行 ID

## 关键配置（与可复现强相关）

- `dataset.label_horizons[0]` 是主训练 horizon
- `cv.method=walk_forward` 时，使用同一 `label_horizon` 自动 purge 边界训练样本
- `dataset.apply_mask` 默认 `true`，确保训练/回测仅使用可交易样本
- 布尔字段统一严格解析：`true/false/1/0`，非法值会在 preflight 阶段报错

## 实验最小记录

- `experiment_id`
- `config_hash`
- `seed`
- `status`（`success` / `blocked`）
- `failure_reason`（blocked 时给出阻断原因）
- `fallback_used`（是否触发在线失败降级）
- `fallback_reason`（触发降级时记录原因）
- `metrics`
- `feature_explainability`（每因子 IC/RankIC/分位收益贡献）
- `feature_explainability_mode`（`oos` 或 `in_sample`，用于标记 explainability 统计口径）
- `backtest`
- `leakage_passed`

## 泄露门禁行为（新增）

- 默认 `runtime.fail_on_leakage=true`：若泄露校验失败，写入 blocked 结果并终止训练/回测
- 仅在显式设置 `runtime.fail_on_leakage=false` 时允许继续运行（结果会记录 `leakage_details`）
- `runtime.validate_config_only=true` 用于 preflight 校验，不进入训练阶段
- `runtime.allow_offline_mock_fallback=true` 时，`joinquant/ricequant` 拉取失败会降级为本地 mock 数据并继续运行（结果记录 `fallback_*` 字段）

## Explainability 复现口径（新增）

- 默认 `eval.explainability.mode=oos`，只使用 OOS 验证样本统计 explainability
- 兼容口径 `eval.explainability.mode=in_sample` 可用于历史结果对齐
- 建议在复现实验时同时固定 `seed` 与 `feature_explainability_mode`

## 推荐实践

- 关键实验固定 seed + 固定依赖版本
- 每次对外展示都附配置文件

## 联机连接器证据闭环（新增）

- 真实连接器 smoke 采用 env-gated 机制：无凭证/无 SDK 时允许 `skip`，但必须记录 `skip` 原因
- 发布前需补一份标准化联机验证记录，模板见：`docs/live-connectors-validation-template.md`
- 推荐至少记录：
  - JoinQuant：`JOINQUANT_USER/JOINQUANT_PASSWORD` 与 `jqdatasdk` 检查结论
  - RiceQuant：`RICEQUANT_TOKEN` 或 `RICEQUANT_USER/RICEQUANT_PASSWORD` 与 `rqdatac` 检查结论
  - 对应 smoke 用例结果：`pass/skip/fail`
