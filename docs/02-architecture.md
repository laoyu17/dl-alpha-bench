# 02 Architecture

## 模块边界

- `data`: 数据接入和契约
- `dataset`: 样本构建
- `cv`: 防泄露切分
- `train`: 模型训练
- `eval`: 评估指标
- `backtest`: 研究回测
- `exp`: 配置化实验编排
- `gui`: PyQt6 工作台

## 主流程

`DataConnector(universe + corporate_actions) -> DatasetBuilder(apply_mask) -> Splitter -> Trainer -> Evaluator -> Backtester -> Tracker`

运行时门禁（新增）：

`ConfigValidator -> DataContractValidator -> LeakageGuard -> (Trainer/Evaluator/Backtester)`

## 关键设计原则

- 时间索引一等公民
- 防泄露优先于模型复杂度
- Walk-forward 在 `label_horizon>1` 时自动 purge 边界训练样本
- 全流程配置化，便于复现
- 默认启用 `runtime.fail_on_leakage=true`，发现泄露风险时阻断训练并写入 blocked artifact
- 支持 `runtime.validate_config_only=true` 仅做配置/数据契约预检查
- 支持 `runtime.allow_offline_mock_fallback=true`：在线连接器失败时降级到本地 mock 回放，并在结果中标注 fallback 元数据
- 在线连接器自带限频、重试、错误分类（认证/限频/临时故障）
- 高频微观结构特征通过 `feature_generators + feature_params` 组合启用
- 配置布尔字段采用严格解析（`true/false/1/0`），非法值直接阻断运行
