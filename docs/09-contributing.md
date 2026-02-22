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
