# 08 Test Plan

## 单元测试

- CV 切分正确性
- 数据集标签对齐
- 配置布尔解析（`true/false/1/0` 合法，非法值阻断）
- 高频微观结构特征构建正确性
- 指标计算
- 回测核心统计
- 连接器重试/限频/错误分类
- 企业行动因子去重与冲突校验

## 集成测试

- 配置驱动实验可跑通
- 结果落盘完整
- 泄露门禁：`status=blocked` 与 `failure_reason` 落盘
- 配置预检查：`runtime.validate_config_only=true` 可返回 blocked preflight
- 在线连接器失败时的 mock 降级（`runtime.allow_offline_mock_fallback=true`）

## 回归测试

- 固定 seed 的指标稳定性
- 防泄露用例持续通过
- CLI 最小配置回归（输出 JSON + 产出 artifact）
- 联机 env-gated 烟测（JoinQuant/RiceQuant，缺凭证自动 skip）

## GUI 测试

- 无 PyQt6 时给出可读错误
- GUI 入口函数可 import
- 配置摘要刷新正确
- 指标表包含 `status/failure_reason`
