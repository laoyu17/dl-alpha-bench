from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from dl_alpha_bench.cv import LeakageReport
from dl_alpha_bench.data import DataConnectorError
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

    result_candidates = sorted((tmp_path / cfg["experiment_id"] / "runs").glob("*/result.json"))
    assert result_candidates
    result_path = result_candidates[-1]
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


def test_runner_accepts_string_false_for_fail_on_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-exp-leakage-allow-str"
    cfg["data"]["periods"] = 120
    cfg["runtime"] = {"fail_on_leakage": "false"}

    def _always_leak(*args: Any, **kwargs: Any):
        _ = (args, kwargs)
        return LeakageReport(passed=False, details=["synthetic leakage"])

    monkeypatch.setattr(runner_module, "validate_no_time_overlap", _always_leak)

    result = ExperimentRunner().run(cfg)
    assert result.status == "success"
    assert result.leakage_passed is False


def test_runner_rejects_invalid_boolean_fields() -> None:
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["runtime"] = {"fail_on_leakage": "not-a-bool"}
    with pytest.raises(ConfigValidationError, match="runtime.fail_on_leakage"):
        ExperimentRunner().run(cfg)

    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["runtime"] = {"allow_offline_mock_fallback": "not-a-bool"}
    with pytest.raises(ConfigValidationError, match="runtime.allow_offline_mock_fallback"):
        ExperimentRunner().run(cfg)

    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["dataset"]["apply_mask"] = "not-a-bool"
    with pytest.raises(ConfigValidationError, match="dataset.apply_mask"):
        ExperimentRunner().run(cfg)


def test_runner_fallback_to_mock_when_online_connector_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingJoinQuantConnector:
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
            _ = (start, end, freq, symbols, fields)
            raise DataConnectorError("simulated provider outage")

        def fetch_corporate_actions(
            self, start: str, end: str, symbols: list[str]
        ) -> pd.DataFrame:
            _ = (start, end, symbols)
            return pd.DataFrame()

    monkeypatch.setattr(runner_module, "JoinQuantConnector", _FailingJoinQuantConnector)

    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-exp-fallback"
    cfg["data"]["source"] = "joinquant"
    cfg["data"]["symbols"] = ["000001.XSHE"]
    cfg["runtime"] = {"allow_offline_mock_fallback": True}

    runner = ExperimentRunner(tracker=ExperimentTracker(root=tmp_path))
    result = runner.run(cfg)

    assert result.status == "success"
    assert result.fallback_used is True
    assert "joinquant connector failure" in str(result.fallback_reason)
    payload = json.loads((Path(result.artifact_dir) / "result.json").read_text(encoding="utf-8"))
    assert payload["fallback_used"] is True


def test_runner_online_connector_failure_without_fallback_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingJoinQuantConnector:
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
            _ = (start, end, freq, symbols, fields)
            raise DataConnectorError("simulated provider outage")

        def fetch_corporate_actions(
            self, start: str, end: str, symbols: list[str]
        ) -> pd.DataFrame:
            _ = (start, end, symbols)
            return pd.DataFrame()

    monkeypatch.setattr(runner_module, "JoinQuantConnector", _FailingJoinQuantConnector)

    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-exp-no-fallback"
    cfg["data"]["source"] = "joinquant"
    cfg["data"]["symbols"] = ["000001.XSHE"]
    cfg["runtime"] = {"allow_offline_mock_fallback": False}

    with pytest.raises(DataConnectorError, match="simulated provider outage"):
        ExperimentRunner().run(cfg)


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


def test_runner_rejects_csv_unsorted_rows(tmp_path: Path) -> None:
    unsorted_csv = tmp_path / "unsorted.csv"
    pd.DataFrame(
        [
            {"symbol": "A", "timestamp": "2024-01-02", "close": 10.0, "tradable": True},
            {"symbol": "A", "timestamp": "2024-01-01", "close": 9.9, "tradable": True},
        ]
    ).to_csv(unsorted_csv, index=False)
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-exp-unsorted-csv"
    cfg["data"] = {"source": "csv", "path": str(unsorted_csv)}
    cfg["dataset"]["feature_generators"] = ["ret_1"]
    cfg["dataset"]["feature_columns"] = ["feat_ret_1"]
    cfg["dataset"]["label_horizons"] = [1]

    with pytest.raises(ConfigValidationError, match="sorted by symbol, timestamp"):
        ExperimentRunner().run(cfg)


def test_runner_rejects_invalid_tradable_values(tmp_path: Path) -> None:
    bad_tradable_csv = tmp_path / "bad-tradable.csv"
    pd.DataFrame(
        [
            {"symbol": "A", "timestamp": "2024-01-01", "close": 10.0, "tradable": "yes"},
            {"symbol": "A", "timestamp": "2024-01-02", "close": 10.1, "tradable": "no"},
        ]
    ).to_csv(bad_tradable_csv, index=False)
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-exp-bad-tradable-csv"
    cfg["data"] = {"source": "csv", "path": str(bad_tradable_csv)}
    cfg["dataset"]["feature_generators"] = ["ret_1"]
    cfg["dataset"]["feature_columns"] = ["feat_ret_1"]
    cfg["dataset"]["label_horizons"] = [1]

    with pytest.raises(ConfigValidationError, match="tradable column must be boolean-like"):
        ExperimentRunner().run(cfg)


def test_tracker_creates_unique_run_directories_for_same_experiment(tmp_path: Path) -> None:
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-exp-multi-run"
    cfg["data"]["periods"] = 120

    runner = ExperimentRunner(tracker=ExperimentTracker(root=tmp_path))
    first = runner.run(cfg)
    second = runner.run(cfg)

    assert first.artifact_dir != second.artifact_dir
    result_paths = sorted((tmp_path / cfg["experiment_id"] / "runs").glob("*/result.json"))
    assert len(result_paths) == 2


def test_runner_emits_progress_messages() -> None:
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-exp-progress"
    cfg["data"]["periods"] = 120

    messages: list[str] = []
    runner = ExperimentRunner(progress_callback=messages.append)
    result = runner.run(cfg)

    assert result.status == "success"
    assert any("config validated" in msg for msg in messages)
    assert any("building dataset bundle" in msg for msg in messages)
    assert any("running backtest" in msg for msg in messages)
    assert messages[-1] == "result saved"
