# Live Connectors Validation Record Template

> 用途：记录 JoinQuant / RiceQuant 真实联机 smoke 结果，补齐“功能具备 + 环境证据”闭环。
>
> 说明：该模板支持 `pass` / `skip` / `fail` 三态；`skip` 必须注明具体原因（例如缺凭证或缺 SDK）。

## 1) 基本信息

- 执行日期（UTC）：`YYYY-MM-DDTHH:mm:ssZ`
- 执行人：`<name>`
- 分支 / 提交：`<branch> @ <commit_sha>`
- 运行环境：`<OS + Python 版本>`

## 2) 执行命令

```bash
pytest -q tests/test_live_connectors_smoke.py
```

如仅验证单个连接器，可拆分执行：

```bash
pytest -q tests/test_live_connectors_smoke.py -k joinquant
pytest -q tests/test_live_connectors_smoke.py -k ricequant
```

## 3) 环境检查（不记录敏感值）

| 项目 | 状态 | 备注 |
|---|---|---|
| `jqdatasdk` 已安装 | yes/no | `python -c "import jqdatasdk"` |
| `rqdatac` 已安装 | yes/no | `python -c "import rqdatac"` |
| `JOINQUANT_USER` 已配置 | yes/no | 仅记录存在性 |
| `JOINQUANT_PASSWORD` 已配置 | yes/no | 仅记录存在性 |
| `RICEQUANT_TOKEN` 已配置 | yes/no | 仅记录存在性 |
| `RICEQUANT_USER` 已配置 | yes/no | 仅记录存在性 |
| `RICEQUANT_PASSWORD` 已配置 | yes/no | 仅记录存在性 |

## 4) 结果记录

| Provider | 测试用例 | 结果 (`pass/skip/fail`) | 证据（日志/关键信息） | skip/fail 原因 |
|---|---|---|---|---|
| JoinQuant | `test_joinquant_live_smoke_if_credentials_present` |  |  |  |
| RiceQuant | `test_ricequant_live_smoke_if_credentials_present` |  |  |  |

## 5) 结论

- 联机能力状态：`fully_verified / partially_verified / not_verified`
- 发布口径建议：
  - 若任一 provider 为 `skip`：标记“代码能力已具备，联机证据受环境约束待补”
  - 若出现 `fail`：需先修复后再发布
