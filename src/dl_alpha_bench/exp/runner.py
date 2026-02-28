"""Config-driven experiment runner."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from dl_alpha_bench.backtest import EventBacktester
from dl_alpha_bench.cv import PurgedKFoldSplitter, WalkForwardSplitter, validate_no_time_overlap
from dl_alpha_bench.data import (
    DataConnectorError,
    JoinQuantConnector,
    LocalMockConnector,
    RiceQuantConnector,
    validate_data_contract,
)
from dl_alpha_bench.dataset import DatasetBuilder
from dl_alpha_bench.eval import summarize_feature_explainability, summarize_fold_metrics
from dl_alpha_bench.train import TrainConfig, Trainer
from dl_alpha_bench.utils.config import parse_config_bool, stable_hash
from dl_alpha_bench.utils.seed import set_global_seed

from .errors import ConfigValidationError, LeakageGuardError
from .tracker import ExperimentResult, ExperimentTracker

_ALLOWED_DATA_SOURCES = {"csv", "mock", "joinquant", "ricequant"}
_ALLOWED_CV_METHODS = {"purged_kfold", "walk_forward"}
_ALLOWED_EXPLAINABILITY_MODES = {"oos", "in_sample"}


class ExperimentRunner:
    def __init__(
        self,
        tracker: ExperimentTracker | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        artifact_root = os.getenv("DL_ALPHA_BENCH_ARTIFACT_DIR", "artifacts")
        self.tracker = tracker or ExperimentTracker(artifact_root)
        self._progress_callback = progress_callback

    def run(self, config: dict[str, Any]) -> ExperimentResult:
        self._validate_config(config)
        self._emit("config validated")

        seed = int(config.get("seed", 42))
        set_global_seed(seed)
        config_hash = stable_hash(config)
        experiment_id = config.get("experiment_id", f"exp-{config_hash}")

        runtime_cfg = config.get("runtime", {})
        fail_on_leakage = self._parse_bool(
            runtime_cfg.get("fail_on_leakage"),
            field_name="runtime.fail_on_leakage",
            default=True,
        )
        validate_config_only = self._parse_bool(
            runtime_cfg.get("validate_config_only"),
            field_name="runtime.validate_config_only",
            default=False,
        )
        allow_offline_mock_fallback = self._parse_bool(
            runtime_cfg.get("allow_offline_mock_fallback"),
            field_name="runtime.allow_offline_mock_fallback",
            default=False,
        )
        eval_cfg = config.get("eval") or {}
        explain_cfg = eval_cfg.get("explainability") or {}
        explain_enabled = self._parse_bool(
            explain_cfg.get("enabled"),
            field_name="eval.explainability.enabled",
            default=True,
        )
        explain_mode = self._parse_explainability_mode(
            explain_cfg.get("mode"),
            field_name="eval.explainability.mode",
            default="oos",
        )
        source = str(config.get("data", {}).get("source", "mock"))
        fallback_used = False
        fallback_reason: str | None = None

        run_dir = self.tracker.start(experiment_id)
        self.tracker.log_config(run_dir, config)
        self._emit(f"artifact directory prepared: {run_dir}")

        self._emit("loading market data")
        loaded_source = source
        try:
            data_frame, actions = self._load_data_and_actions(config)
        except DataConnectorError as exc:
            if source in {"joinquant", "ricequant"} and allow_offline_mock_fallback:
                fallback_used = True
                fallback_reason = f"{source} connector failure: {exc}"
                loaded_source = "mock"
                self._emit(f"{fallback_reason}; fallback to local mock data")
                data_frame, actions = self._load_mock_data_and_actions(config)
            else:
                raise

        self._validate_loaded_data(frame=data_frame, source=loaded_source)
        data_status = f"data contract validated: {len(data_frame)} rows from {loaded_source}"
        if fallback_used:
            data_status = f"{data_status} (fallback enabled)"
        self._emit(data_status)

        if validate_config_only:
            result = self._make_blocked_result(
                experiment_id=experiment_id,
                config_hash=config_hash,
                seed=seed,
                artifact_dir=run_dir,
                leakage_passed=True,
                leakage_details=[],
                failure_reason="validate_config_only",
                feature_explainability_mode=explain_mode,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
            self.tracker.log_result(run_dir, result)
            self._emit("validate_config_only enabled, run blocked before training")
            return result

        dataset_cfg = config.get("dataset", {})
        self._emit("building dataset bundle")
        try:
            dataset = DatasetBuilder().build(data_frame, dataset_cfg, actions=actions)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigValidationError(f"dataset config is invalid: {exc}") from exc

        if dataset.frame.empty:
            raise ConfigValidationError(
                "dataset is empty after feature/label alignment and mask filtering"
            )
        if not dataset.feature_columns:
            raise ConfigValidationError("dataset produced no usable feature columns")
        if not dataset.label_columns:
            raise ConfigValidationError("dataset produced no usable label columns")
        self._emit(
            f"dataset ready: rows={len(dataset.frame)}, "
            f"features={len(dataset.feature_columns)}, labels={len(dataset.label_columns)}"
        )

        split_kind = config.get("cv", {}).get("method", "purged_kfold")
        split_cfg = config.get("cv", {})
        label_horizon = int(dataset_cfg.get("label_horizons", [1])[0])
        self._emit(f"building cv splits ({split_kind})")
        if split_kind == "walk_forward":
            splitter = WalkForwardSplitter(
                train_window=int(split_cfg.get("train_window", 120)),
                valid_window=int(split_cfg.get("valid_window", 20)),
                step=int(split_cfg.get("step", split_cfg.get("valid_window", 20))),
                label_horizon=label_horizon,
            )
            splits = splitter.split(dataset.frame["timestamp"])
            embargo = 0
        else:
            embargo = int(split_cfg.get("embargo", 0))
            splitter = PurgedKFoldSplitter(
                n_splits=int(split_cfg.get("n_splits", 5)),
                label_horizon=label_horizon,
                embargo=embargo,
            )
            splits = splitter.split(dataset.frame["timestamp"])

        if not splits:
            raise ConfigValidationError("no valid CV split generated")
        self._emit(f"generated {len(splits)} splits")

        leakage_details: list[str] = []
        leakage_passed = True
        for split in splits:
            report = validate_no_time_overlap(
                dataset.frame["timestamp"],
                split,
                label_horizon,
                embargo,
            )
            if not report.passed:
                leakage_passed = False
                leakage_details.extend(report.details)

        leakage_details = sorted(set(leakage_details))
        self._emit(
            "leakage validation passed"
            if leakage_passed
            else f"leakage validation failed ({len(leakage_details)} issue(s))"
        )
        if not leakage_passed and fail_on_leakage:
            failure_reason = self._build_leakage_failure_reason(leakage_details)
            blocked_result = self._make_blocked_result(
                experiment_id=experiment_id,
                config_hash=config_hash,
                seed=seed,
                artifact_dir=run_dir,
                leakage_passed=False,
                leakage_details=leakage_details,
                failure_reason=failure_reason,
                feature_explainability_mode=explain_mode,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
            self.tracker.log_result(run_dir, blocked_result)
            self._emit(f"run blocked by leakage guard: {failure_reason}")
            raise LeakageGuardError(f"{failure_reason}; artifact={run_dir}")

        train_cfg = config.get("train", {})
        trainer = Trainer(
            TrainConfig(
                seed=seed,
                hidden_dim=int(train_cfg.get("hidden_dim", 32)),
                lr=float(train_cfg.get("lr", 1e-2)),
                epochs=int(train_cfg.get("epochs", 50)),
                l2=float(train_cfg.get("l2", 0.0)),
            )
        )

        self._emit("training model across cv folds")
        x = dataset.features()
        y = dataset.labels()[:, 0]
        fold_results = trainer.fit_cv(x, y, splits)
        metric_summary = summarize_fold_metrics(fold_results)
        self._emit("training complete")

        pred_col = "pred_score"
        valid_frame = dataset.frame.copy()
        valid_frame[pred_col] = np.nan
        for fold, split in zip(fold_results, splits):
            valid_frame.loc[split.valid_idx, pred_col] = fold.valid_pred
        oos_frame = valid_frame.dropna(subset=[pred_col])

        feature_explainability: list[dict[str, Any]] = []
        if explain_enabled:
            explain_frame = oos_frame if explain_mode == "oos" else dataset.frame
            self._emit(
                f"computing feature explainability (mode={explain_mode}, rows={len(explain_frame)})"
            )
            feature_explainability = summarize_feature_explainability(
                frame=explain_frame,
                feature_columns=dataset.feature_columns,
                label_column=dataset.label_columns[0],
                timestamp_column="timestamp",
                n_quantiles=int(explain_cfg.get("n_quantiles", 5)),
            )
            top_k = explain_cfg.get("top_k")
            if top_k is not None:
                feature_explainability = feature_explainability[: max(1, int(top_k))]
            self.tracker.log_explainability(run_dir, feature_explainability)
            self._emit(
                f"explainability saved ({len(feature_explainability)} feature rows, mode={explain_mode})"
            )

        back_cfg = config.get("backtest", {})
        backtester = EventBacktester(
            top_frac=float(back_cfg.get("top_frac", 0.2)),
            trading_cost=float(back_cfg.get("trading_cost", 0.0005)),
        )
        self._emit("running backtest")
        back_summary = backtester.run(
            oos_frame,
            pred_col,
            dataset.label_columns[0],
        )

        metrics = {
            "ic_mean": metric_summary.ic_mean,
            "rank_ic_mean": metric_summary.rank_ic_mean,
            "rmse_mean": metric_summary.rmse_mean,
        }
        result = ExperimentResult(
            experiment_id=experiment_id,
            config_hash=config_hash,
            seed=seed,
            metrics=metrics,
            feature_explainability=feature_explainability,
            feature_explainability_mode=explain_mode,
            leakage_passed=leakage_passed,
            leakage_details=leakage_details,
            backtest={k: float(v) for k, v in asdict(back_summary).items()},
            artifact_dir=str(run_dir),
            status="success",
            failure_reason=None,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )
        self.tracker.log_result(run_dir, result)
        self._emit("result saved")
        return result

    def _validate_config(self, config: dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise ConfigValidationError("config must be a dict")

        data_cfg = config.get("data", {})
        if not isinstance(data_cfg, dict):
            raise ConfigValidationError("config.data must be a dict")
        source = str(data_cfg.get("source", "mock"))
        if source not in _ALLOWED_DATA_SOURCES:
            allowed = ", ".join(sorted(_ALLOWED_DATA_SOURCES))
            raise ConfigValidationError(f"data.source must be one of: {allowed}")

        if source == "csv":
            path = data_cfg.get("path")
            if not path:
                raise ConfigValidationError("data.path is required when data.source=csv")
            if not Path(path).exists():
                raise ConfigValidationError(f"csv path does not exist: {path}")

        if source in {"joinquant", "ricequant"}:
            symbols = data_cfg.get("symbols")
            if not isinstance(symbols, list) or not symbols:
                raise ConfigValidationError(
                    f"data.symbols must be a non-empty list when data.source={source}"
                )

        dataset_cfg = config.get("dataset", {})
        if not isinstance(dataset_cfg, dict):
            raise ConfigValidationError("config.dataset must be a dict")
        horizons = dataset_cfg.get("label_horizons", [1])
        if not isinstance(horizons, list) or not horizons:
            raise ConfigValidationError("dataset.label_horizons must be a non-empty list")
        for horizon in horizons:
            try:
                value = int(horizon)
            except (TypeError, ValueError) as exc:
                raise ConfigValidationError(
                    f"dataset.label_horizons contains invalid value: {horizon}"
                ) from exc
            if value <= 0:
                raise ConfigValidationError("dataset.label_horizons values must be positive")

        feature_columns = dataset_cfg.get("feature_columns")
        if feature_columns is not None:
            if not isinstance(feature_columns, list) or not feature_columns:
                raise ConfigValidationError(
                    "dataset.feature_columns must be a non-empty list when provided"
                )
            if any(not isinstance(col, str) or not col.strip() for col in feature_columns):
                raise ConfigValidationError(
                    "dataset.feature_columns must contain non-empty strings"
                )

        cv_cfg = config.get("cv", {})
        if not isinstance(cv_cfg, dict):
            raise ConfigValidationError("config.cv must be a dict")

        cv_method = cv_cfg.get("method", "purged_kfold")
        if cv_method not in _ALLOWED_CV_METHODS:
            allowed = ", ".join(sorted(_ALLOWED_CV_METHODS))
            raise ConfigValidationError(f"cv.method must be one of: {allowed}")

        if cv_method == "purged_kfold":
            try:
                n_splits = int(cv_cfg.get("n_splits", 5))
                embargo = int(cv_cfg.get("embargo", 0))
            except (TypeError, ValueError) as exc:
                raise ConfigValidationError(
                    "cv.n_splits and cv.embargo must be integers for purged_kfold"
                ) from exc
            if n_splits < 2:
                raise ConfigValidationError("cv.n_splits must be >= 2 for purged_kfold")
            if embargo < 0:
                raise ConfigValidationError("cv.embargo must be >= 0")
        else:
            try:
                train_window = int(cv_cfg.get("train_window", 120))
                valid_window = int(cv_cfg.get("valid_window", 20))
                step = int(cv_cfg.get("step", valid_window))
            except (TypeError, ValueError) as exc:
                raise ConfigValidationError(
                    "walk_forward parameters must be integers"
                ) from exc
            if train_window <= 0 or valid_window <= 0 or step <= 0:
                raise ConfigValidationError(
                    "walk_forward requires train_window/valid_window/step to be positive"
                )

        runtime_cfg = config.get("runtime")
        if runtime_cfg is None:
            runtime_cfg = {}
        elif not isinstance(runtime_cfg, dict):
            raise ConfigValidationError("config.runtime must be a dict when provided")
        self._parse_bool(
            runtime_cfg.get("fail_on_leakage"),
            field_name="runtime.fail_on_leakage",
            default=True,
        )
        self._parse_bool(
            runtime_cfg.get("validate_config_only"),
            field_name="runtime.validate_config_only",
            default=False,
        )
        self._parse_bool(
            runtime_cfg.get("allow_offline_mock_fallback"),
            field_name="runtime.allow_offline_mock_fallback",
            default=False,
        )

        self._parse_bool(
            dataset_cfg.get("apply_mask"),
            field_name="dataset.apply_mask",
            default=True,
        )
        self._parse_bool(
            dataset_cfg.get("strict_feature_requirements"),
            field_name="dataset.strict_feature_requirements",
            default=True,
        )

        eval_cfg = config.get("eval")
        if eval_cfg is None:
            eval_cfg = {}
        elif not isinstance(eval_cfg, dict):
            raise ConfigValidationError("config.eval must be a dict when provided")
        explain_cfg = eval_cfg.get("explainability")
        if explain_cfg is None:
            explain_cfg = {}
        elif not isinstance(explain_cfg, dict):
            raise ConfigValidationError("config.eval.explainability must be a dict when provided")
        self._parse_bool(
            explain_cfg.get("enabled"),
            field_name="eval.explainability.enabled",
            default=True,
        )
        self._parse_explainability_mode(
            explain_cfg.get("mode"),
            field_name="eval.explainability.mode",
            default="oos",
        )

    def _validate_loaded_data(self, frame: pd.DataFrame, source: str) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise ConfigValidationError(f"{source} returned non-DataFrame payload")
        if frame.empty:
            raise ConfigValidationError(f"{source} returned empty dataset")
        try:
            validate_data_contract(frame)
        except ValueError as exc:
            raise ConfigValidationError(str(exc)) from exc

    def _emit(self, message: str) -> None:
        if self._progress_callback is None:
            return
        self._progress_callback(message)

    def _parse_bool(self, value: Any, *, field_name: str, default: bool) -> bool:
        try:
            return parse_config_bool(
                value,
                default=default,
                field_name=field_name,
            )
        except ValueError as exc:
            raise ConfigValidationError(str(exc)) from exc

    def _parse_explainability_mode(
        self,
        value: Any,
        *,
        field_name: str,
        default: str,
    ) -> str:
        if value is None:
            return default
        if not isinstance(value, str):
            allowed = ", ".join(sorted(_ALLOWED_EXPLAINABILITY_MODES))
            raise ConfigValidationError(f"{field_name} must be one of: {allowed}")
        mode = value.strip().lower()
        if mode not in _ALLOWED_EXPLAINABILITY_MODES:
            allowed = ", ".join(sorted(_ALLOWED_EXPLAINABILITY_MODES))
            raise ConfigValidationError(f"{field_name} must be one of: {allowed}")
        return mode

    def _make_blocked_result(
        self,
        experiment_id: str,
        config_hash: str,
        seed: int,
        artifact_dir: Path,
        leakage_passed: bool,
        leakage_details: list[str],
        failure_reason: str,
        feature_explainability_mode: str,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=experiment_id,
            config_hash=config_hash,
            seed=seed,
            metrics={},
            feature_explainability=[],
            feature_explainability_mode=feature_explainability_mode,
            leakage_passed=leakage_passed,
            leakage_details=leakage_details,
            backtest={},
            artifact_dir=str(artifact_dir),
            status="blocked",
            failure_reason=failure_reason,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    def _build_leakage_failure_reason(self, details: list[str]) -> str:
        if not details:
            return "leakage_guard: leakage validation failed"
        merged = "; ".join(details)
        return f"leakage_guard: {merged}"

    def _load_data_and_actions(
        self,
        config: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        data_cfg = config.get("data", {})
        source = data_cfg.get("source", "mock")
        fields = data_cfg.get("fields")
        symbols = data_cfg.get("symbols", [])
        start = data_cfg.get("start", "2020-01-01")
        end = data_cfg.get("end", "2022-01-01")
        freq = data_cfg.get("freq", "1d")

        if source == "csv":
            path_value = data_cfg.get("path")
            if not path_value:
                raise ConfigValidationError("data.path is required when data.source=csv")
            path = Path(path_value)
            try:
                frame = pd.read_csv(path)
            except Exception as exc:  # noqa: BLE001
                raise ConfigValidationError(f"failed to read csv data from {path}: {exc}") from exc
            if "timestamp" not in frame.columns:
                raise ConfigValidationError("csv input must include `timestamp` column")
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            return frame, None

        if source == "joinquant":
            connector = JoinQuantConnector(
                user_env=data_cfg.get("user_env", "JOINQUANT_USER"),
                pwd_env=data_cfg.get("password_env", "JOINQUANT_PASSWORD"),
                min_interval_sec=data_cfg.get("min_interval_sec"),
                max_retries=data_cfg.get("max_retries"),
                base_delay_sec=data_cfg.get("base_delay_sec"),
                max_delay_sec=data_cfg.get("max_delay_sec"),
            )
            frame = connector.fetch_universe(
                start=start,
                end=end,
                freq=freq,
                symbols=symbols,
                fields=fields,
            )
            actions = connector.fetch_corporate_actions(start=start, end=end, symbols=symbols)
            return frame, actions

        if source == "ricequant":
            connector = RiceQuantConnector(
                token_env=data_cfg.get("token_env", "RICEQUANT_TOKEN"),
                user_env=data_cfg.get("user_env", "RICEQUANT_USER"),
                pwd_env=data_cfg.get("password_env", "RICEQUANT_PASSWORD"),
                min_interval_sec=data_cfg.get("min_interval_sec"),
                max_retries=data_cfg.get("max_retries"),
                base_delay_sec=data_cfg.get("base_delay_sec"),
                max_delay_sec=data_cfg.get("max_delay_sec"),
            )
            frame = connector.fetch_universe(
                start=start,
                end=end,
                freq=freq,
                symbols=symbols,
                fields=fields,
            )
            actions = connector.fetch_corporate_actions(start=start, end=end, symbols=symbols)
            return frame, actions

        # Default synthetic demo path for deterministic CI and local reproducibility.
        return self._load_mock_data_and_actions(config)

    def _load_mock_data_and_actions(
        self,
        config: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        data_cfg = config.get("data", {})
        fields = data_cfg.get("fields")
        symbols = data_cfg.get("symbols", [])
        start = data_cfg.get("start", "2020-01-01")
        end = data_cfg.get("end", "2022-01-01")
        freq = data_cfg.get("freq", "1d")
        frame = self._make_synthetic_data(
            start=start,
            periods=int(data_cfg.get("periods", 260)),
            symbols=symbols or ["AAA", "BBB", "CCC", "DDD", "EEE"],
            freq=freq,
            seed=int(data_cfg.get("seed", 123)),
        )
        connector = LocalMockConnector(frame)
        out = connector.fetch_universe(
            start=start,
            end=end,
            freq=freq,
            symbols=symbols,
            fields=fields,
        )
        return out, None

    def _make_synthetic_data(
        self,
        start: str,
        periods: int,
        symbols: list[str],
        freq: str,
        seed: int,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        freq_lower = freq.lower()
        if freq_lower.endswith("d"):
            pandas_freq = "D"
        elif freq_lower.endswith("h"):
            pandas_freq = "H"
        elif freq_lower.endswith("m") or "min" in freq_lower:
            pandas_freq = "min"
        else:
            pandas_freq = "D"
        ts = pd.date_range(start=start, periods=periods, freq=pandas_freq)

        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            base = 100 + rng.normal(scale=3)
            shocks = rng.normal(0, 0.01, size=len(ts))
            trend = np.cumsum(shocks)
            close = base * np.exp(trend)
            high = close * (1 + rng.uniform(0, 0.01, size=len(ts)))
            low = close * (1 - rng.uniform(0, 0.01, size=len(ts)))
            open_ = close * (1 + rng.normal(0, 0.002, size=len(ts)))
            volume = rng.integers(1_000, 10_000, size=len(ts))
            trade_count = rng.integers(10, 300, size=len(ts))
            bid_size1 = rng.integers(50, 4_000, size=len(ts))
            ask_size1 = rng.integers(50, 4_000, size=len(ts))
            spread_bps = rng.uniform(1e-4, 8e-4, size=len(ts))
            bid_price1 = close * (1 - spread_bps / 2)
            ask_price1 = close * (1 + spread_bps / 2)
            for i, stamp in enumerate(ts):
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": stamp,
                        "open": float(open_[i]),
                        "high": float(high[i]),
                        "low": float(low[i]),
                        "close": float(close[i]),
                        "volume": float(volume[i]),
                        "value": float(close[i] * volume[i]),
                        "bid_price1": float(bid_price1[i]),
                        "ask_price1": float(ask_price1[i]),
                        "bid_size1": float(bid_size1[i]),
                        "ask_size1": float(ask_size1[i]),
                        "trade_count": float(trade_count[i]),
                        "tradable": True,
                    }
                )
        return pd.DataFrame(rows)
