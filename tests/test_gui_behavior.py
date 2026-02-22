from __future__ import annotations

from pathlib import Path

import pytest

from dl_alpha_bench.utils.config import dump_yaml, load_yaml


def _make_window(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("PyQt6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from dl_alpha_bench.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    return app, window


def test_gui_refresh_config_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _, window = _make_window(monkeypatch)
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg_path = tmp_path / "gui-config.yaml"
    dump_yaml(cfg, cfg_path)

    try:
        window.config_path.setText(str(cfg_path))
        window.refresh_config_summary()
        text = window.dataset_summary.toPlainText()
        assert f"config: {cfg_path}" in text
        assert "data.source: mock" in text
        assert "cv.method: purged_kfold" in text
    finally:
        window.close()


def test_gui_fill_metric_table_contains_status_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, window = _make_window(monkeypatch)
    payload = {
        "metrics": {"ic_mean": 0.123},
        "backtest": {"sharpe": 1.5},
        "status": "blocked",
        "failure_reason": "validate_config_only",
    }

    try:
        window._fill_metric_table(payload)
        keys = {
            window.metrics_table.item(row, 0).text()
            for row in range(window.metrics_table.rowCount())
        }
        assert {"ic_mean", "sharpe", "status", "failure_reason"}.issubset(keys)
    finally:
        window.close()
