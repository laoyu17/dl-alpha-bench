# 03 Data Contract

## 标准列

- 必需：`symbol`, `timestamp`, `close`
- 建议：`open`, `high`, `low`, `volume`, `value`, `tradable`
- 高频可选：`bid_price1`, `ask_price1`, `bid_size1`, `ask_size1`, `trade_count`

## 约束

- `timestamp` 必须可解析为时间戳
- 数据按 `symbol, timestamp` 排序
- `tradable` 为布尔值，用于过滤停牌等不可交易样本

## mask 语义

- `mask_column` 默认使用 `tradable`
- `apply_mask` 默认为 `true`，表示训练与回测均仅保留 `mask_column == true` 的样本
- 仅在明确需要诊断不可交易样本时，将 `apply_mask` 设为 `false`

## 布尔配置解析规则（新增）

- 以下字段支持 `true/false/1/0`（布尔、数字或字符串），其他取值会被视为配置错误并阻断运行：
  - `dataset.apply_mask`
  - `dataset.strict_feature_requirements`
  - `runtime.fail_on_leakage`
  - `runtime.validate_config_only`
  - `runtime.allow_offline_mock_fallback`
  - `eval.explainability.enabled`

## 标签定义

- `label_fwd_ret_h = close(t+h)/close(t)-1`
- 支持多 horizon（默认使用第一个 horizon 训练）

## 运行时配置契约（新增）

- `runtime.fail_on_leakage`：默认 `true`，若任一 split 泄露校验失败则阻断训练
- `runtime.validate_config_only`：默认 `false`，设为 `true` 时仅做配置与数据契约检查并输出 blocked 结果
- `runtime.allow_offline_mock_fallback`：默认 `false`，仅在 `joinquant/ricequant` 拉取失败时允许降级到本地 mock 数据

## 可解释性口径契约（新增）

- `eval.explainability.mode` 支持：
  - `oos`（默认）：仅基于各 fold 验证集样本统计 `feature_explainability`
  - `in_sample`：基于完整数据集统计（兼容历史口径）
- `result.json` 会输出 `feature_explainability_mode`，用于审计与复现 explainability 口径

## 高频特征输入约束

- `obi_l1` 依赖：`bid_size1`, `ask_size1`
- `trade_intensity` 依赖：`trade_count`
- `short_vol` 依赖：`close`
- `microprice_bias` 依赖：`bid_price1`, `ask_price1`, `bid_size1`, `ask_size1`, `close`
- `ofi_l1` 依赖：`bid_size1`, `ask_size1`

## 企业行动接口

- 在线连接器会在实验中调用 `fetch_corporate_actions()` 获取 `adjust_factor`
- 通过 `CorporateActionAdjuster.apply()` 进行复权/调整（会校验 `(symbol,timestamp)` 唯一性，冲突因子将报错）
- 默认 identity，不引入隐式改动
