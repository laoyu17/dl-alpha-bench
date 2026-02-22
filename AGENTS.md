# AGENTS.md (Project-Level)

## 项目定位

- 项目名：`dl-alpha-bench`
- 仓库地址：`https://github.com/laoyu17/dl-alpha-bench.git`
- 定位：金融专用、可防数据泄露的深度学习量化实验框架

## 分支与提交流程

- 分支模型：`main` + `feat/*` + `fix/*` + `docs/*`
- 提交规范：Conventional Commits
  - `feat:` 新功能
  - `fix:` 缺陷修复
  - `docs:` 文档改动
  - `refactor:` 重构
  - `test:` 测试
  - `chore:` 工程杂项
- 提交粒度：一次提交只做一件事

## 每次开发的强制检查

1. 单元测试：`pytest`
2. 防泄露专项：`tests/test_cv_leakage.py`
3. 最小回归：`tests/test_experiment_runner.py`
4. 连接器容错：`tests/test_connectors_resilience.py`
5. 高频特征：`tests/test_microstructure_features.py`
6. 代码质量（可选但推荐）：`ruff check src tests`

## PR 必填内容

- 变更摘要（做了什么）
- 风险说明（可能影响）
- 回归结果（执行了哪些测试）
- 文档更新列表（若接口/行为变化）

## 文档同步策略

- 任何影响 API、配置字段、数据契约的改动，必须同步更新：
  - `docs/02-architecture.md`
  - `docs/03-data-contract.md`
  - `docs/05-experiment-reproducibility.md`

## 凭证与数据合规

- 凭证仅允许放在 `.env`，严禁提交到 git
- 提供接口与模板，不在仓库中分发受限数据
- 在线接口失败时允许离线 mock 回放，以保障可复现实验
