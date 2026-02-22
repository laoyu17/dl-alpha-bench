# 10 Release Checklist

- [ ] `pytest` 全通过
- [ ] 防泄露测试通过
- [ ] 泄露门禁测试通过（blocked result + failure_reason）
- [ ] 布尔配置解析用例通过（`true/false/1/0` + 非法值阻断）
- [ ] 在线失败降级策略验证通过（`runtime.allow_offline_mock_fallback` 开/关两条路径）
- [ ] 示例配置可跑通并生成 artifacts
- [ ] 联机 smoke（有凭证时）通过，或缺凭证时明确 skip 记录
- [ ] GUI 基础行为测试通过（配置摘要刷新、指标表填充）
- [ ] README 与 docs 同步更新
- [ ] `.env` 未泄露凭证
- [ ] 发布说明包含主要变更与已知限制
