# 04 Leakage Prevention

## Purged/Embargo

- Purged：验证窗口前 `label_horizon-1` 时间步不进入训练
- Embargo：验证窗口后 `embargo` 时间步不进入训练

## Walk-forward

- 仅用过去窗口训练，未来窗口验证
- 禁止未来样本进入训练集

## 自动校验

- 每个 split 执行 `validate_no_time_overlap`
- 若出现 forbidden zone 冲突，记录到 `leakage_details`
