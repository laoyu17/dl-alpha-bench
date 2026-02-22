from __future__ import annotations

import importlib.util
import os

import pytest

from dl_alpha_bench.exp import ExperimentRunner
from dl_alpha_bench.utils.config import load_yaml


def _skip_if_missing_sdk(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        pytest.skip(f"{module_name} is not installed")


def _build_live_cfg(template_path: str, experiment_id: str) -> dict:
    cfg = load_yaml(template_path)
    cfg["experiment_id"] = experiment_id
    cfg["data"]["start"] = "2024-01-02"
    cfg["data"]["end"] = "2024-01-31"
    cfg["data"]["freq"] = "1d"
    cfg["data"]["symbols"] = cfg["data"]["symbols"][:1]
    cfg["dataset"]["label_horizons"] = [1]
    cfg["cv"]["method"] = "purged_kfold"
    cfg["cv"]["n_splits"] = 2
    cfg["cv"]["embargo"] = 0
    cfg["train"]["epochs"] = 5
    cfg["runtime"] = {"fail_on_leakage": True}
    return cfg


def test_joinquant_live_smoke_if_credentials_present() -> None:
    if not (os.getenv("JOINQUANT_USER") and os.getenv("JOINQUANT_PASSWORD")):
        pytest.skip("JOINQUANT_USER/JOINQUANT_PASSWORD not configured")
    _skip_if_missing_sdk("jqdatasdk")

    cfg = _build_live_cfg(
        template_path="configs/experiment_joinquant_template.yaml",
        experiment_id="pytest-live-joinquant-smoke",
    )
    result = ExperimentRunner().run(cfg)
    assert result.status == "success"
    assert "ic_mean" in result.metrics


def test_ricequant_live_smoke_if_credentials_present() -> None:
    has_token = bool(os.getenv("RICEQUANT_TOKEN"))
    has_user_pwd = bool(os.getenv("RICEQUANT_USER") and os.getenv("RICEQUANT_PASSWORD"))
    if not (has_token or has_user_pwd):
        pytest.skip("RICEQUANT_TOKEN or RICEQUANT_USER/RICEQUANT_PASSWORD not configured")
    _skip_if_missing_sdk("rqdatac")

    cfg = _build_live_cfg(
        template_path="configs/experiment_ricequant_template.yaml",
        experiment_id="pytest-live-ricequant-smoke",
    )
    result = ExperimentRunner().run(cfg)
    assert result.status == "success"
    assert "ic_mean" in result.metrics
