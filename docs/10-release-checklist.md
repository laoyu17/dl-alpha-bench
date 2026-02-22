# 10 Release Checklist

- [ ] `pytest` 全通过
- [ ] 防泄露测试通过
- [ ] 泄露门禁测试通过（blocked result + failure_reason）
- [ ] 示例配置可跑通并生成 artifacts
- [ ] 联机 smoke（有凭证时）通过，或缺凭证时明确 skip 记录
- [ ] README 与 docs 同步更新
- [ ] `.env` 未泄露凭证
- [ ] 发布说明包含主要变更与已知限制
