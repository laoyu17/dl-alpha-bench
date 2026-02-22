from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from dl_alpha_bench.cv import LeakageReport
from dl_alpha_bench.dataset import DatasetBuilder as BaseDatasetBuilder
from dl_alpha_bench.exp import ConfigValidationError, ExperimentRunner, LeakageGuardError
from dl_alpha_bench.exp import runner as runner_module
from dl_alpha_bench.exp.tracker import ExperimentTracker
from dl_alpha_bench.utils.config import load_yaml


def test_experiment_runner_outputs_artifact(tmp_path: Path) -> None:
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-exp"
    cfg["data"]["periods"] = 120

    runner = ExperimentRunner()
    result = runner.run(cfg)

    result_path = Path(result.artifact_dir) / "result.json"
    explain_path = Path(result.artifact_dir) / "feature_explainability.csv"
    assert result_path.exists()
    assert explain_path.exists()
    assert "ic_mean" in result.metrics
    assert result.feature_explainability
    assert result.backtest["n_periods"] > 0


def test_runner_reproducible_with_same_seed() -> None:
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-exp-repro-1"
    cfg["data"]["periods"] = 120
    cfg["seed"] = 999

    runner = ExperimentRunner()
    r1 = runner.run(cfg)
    cfg["experiment_id"] = "pytest-exp-repro-2"
    r2 = runner.run(cfg)

    assert abs(r1.metrics["ic_mean"] - r2.metrics["ic_mean"]) < 1e-12
    assert abs(r1.backtest["sharpe"] - r2.backtest["sharpe"]) < 1e-12


def test_runner_fetches_corporate_actions_for_online_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJoinQuantConnector:
        def __init__(self, **kwargs: Any):
            _ = kwargs

        def fetch_universe(
            self,
            start: str,
            end: str,
            freq: str,
            symbols: list[str],
            fields: list[str] | None = None,
        ) -> pd.DataFrame:
            _ = (freq, fields)
            ts = pd.date_range(start=start, end=end, freq="D")
            rows: list[dict[str, Any]] = []
            for s_idx, symbol in enumerate(symbols):
                for i, stamp in enumerate(ts):
                    close = 100 + s_idx + i * 0.1
                    rows.append(
                        {
                            "symbol": symbol,
                            "timestamp": stamp,
                            "open": close,
                            "high": close * 1.01,
                            "low": close * 0.99,
                            "close": close,
                            "volume": 1000 + i,
                            "value": close * (1000 + i),
                            "tradable": True,
                        }
                    )
            return pd.DataFrame(rows)

        def fetch_corporate_actions(self, start: str, end: str, symbols: list[str]) -> pd.DataFrame:
            ts = pd.date_range(start=start, end=end, freq="D")
            return pd.DataFrame(
                {
                    "symbol": [symbols[0]],
                    "timestamp": [ts[10]],
                    "adjust_factor": [1.05],
                }
            )

    class CapturingDatasetBuilder(BaseDatasetBuilder):
        last_actions: pd.DataFrame | None = None

        def build(self, *args: Any, **kwargs: Any):
            actions = kwargs.get("actions")
            if actions is None and len(args) >= 4:
                actions = args[3]
            type(self).last_actions = actions
            return super().build(*args, **kwargs)

    monkeypatch.setattr(runner_module, "JoinQuantConnector", FakeJoinQuantConnector)
    monkeypatch.setattr(runner_module, "DatasetBuilder", CapturingDatasetBuilder)

    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-exp-actions"
    cfg["data"]["source"] = "joinquant"
    cfg["data"]["start"] = "2024-01-01"
    cfg["data"]["end"] = "2024-04-30"
    cfg["data"]["symbols"] = ["000001.XSHE", "000333.XSHE", "600519.XSHG"]

    result = ExperimentRunner().run(cfg)

    assert result.leakage_passed
    assert CapturingDatasetBuilder.last_actions is not None
    assert not CapturingDatasetBuilder.last_actions.empty


def test_runner_blocks_and_records_when_leakage_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-exp-leakage-block"
    cfg["data"]["periods"] = 120

    def _always_leak(*args: Any, **kwargs: Any):
        _ = (args, kwargs)
        return LeakageReport(passed=False, details=["synthetic leakage"])

    monkeypatch.setattr(runner_module, "validate_no_time_overlap", _always_leak)

    runner = ExperimentRunner(tracker=ExperimentTracker(root=tmp_path))
    with pytest.raises(LeakageGuardError):
        runner.run(cfg)

    result_path = tmp_path / cfg["experiment_id"] / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["metrics"] == {}
    assert payload["backtest"] == {}
    assert payload["leakage_passed"] is False
    assert payload["failure_reason"].startswith("leakage_guard:")


def test_runner_can_continue_with_fail_on_leakage_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-exp-leakage-allow"
    cfg["data"]["periods"] = 120
    cfg["runtime"] = {"fail_on_leakage": False}

    def _always_leak(*args: Any, **kwargs: Any):
        _ = (args, kwargs)
        return LeakageReport(passed=False, details=["synthetic leakage"])

    monkeypatch.setattr(runner_module, "validate_no_time_overlap", _always_leak)

    result = ExperimentRunner().run(cfg)
    assert result.status == "success"
    assert result.leakage_passed is False
    assert "synthetic leakage" in result.leakage_details


def test_runner_validate_config_only_returns_blocked_result(tmp_path: Path) -> None:
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-exp-validate-only"
    cfg["data"]["periods"] = 120
    cfg["runtime"] = {"validate_config_only": True}

    runner = ExperimentRunner(tracker=ExperimentTracker(root=tmp_path))
    result = runner.run(cfg)
    assert result.status == "blocked"
    assert result.failure_reason == "validate_config_only"
    assert result.metrics == {}
    assert result.backtest == {}


def test_runner_rejects_invalid_label_horizon() -> None:
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["dataset"]["label_horizons"] = []
    with pytest.raises(ConfigValidationError, match="label_horizons"):
        ExperimentRunner().run(cfg)


def test_runner_rejects_csv_missing_timestamp(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"symbol": ["A"], "close": [1.0]}).to_csv(bad_csv, index=False)
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-exp-bad-csv"
    cfg["data"] = {"source": "csv", "path": str(bad_csv)}

    with pytest.raises(ConfigValidationError, match="timestamp"):
        ExperimentRunner().run(cfg)
