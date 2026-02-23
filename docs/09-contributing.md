# 09 Contributing

## 快速开始

```bash
pip install -e .[dev]
pytest
```

如需在线数据源：

```bash
pip install jqdatasdk rqdatac
cp .env.example .env
```

## 贡献流程

1. 从 `main` 切分支（如 `feat/purged-cv`）
2. 完成改动并补测试
3. 按 Conventional Commits 提交
4. 发起 PR，填写模板

## 代码风格

- 保持函数单一职责
- 配置优先，避免硬编码
- 所有时间逻辑明确时区和排序

## 文档与演示素材同步

- 若改动 GUI 页面结构或展示字段，需要同步更新 README 的 `UI Demo` 区域
- README 演示素材统一由脚本生成：

```bash
python scripts/generate_readme_ui_assets.py
```

- 生成结果位于 `docs/assets/readme/`，提交 PR 前请确认图片路径可用、无敏感信息
