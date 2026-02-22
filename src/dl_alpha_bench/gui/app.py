"""PyQt6 GUI for managing quant DL experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dl_alpha_bench.exp import ExperimentRunner
from dl_alpha_bench.utils.config import load_yaml

try:
    from PyQt6.QtCore import QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyQt6 is not installed. Run: pip install -e .[gui,plot]"
    ) from exc


class ExperimentWorker(QThread):
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path

    def run(self) -> None:  # noqa: D401
        try:
            cfg = load_yaml(self.config_path)
            res = ExperimentRunner().run(cfg)
            self.finished_ok.emit(res.__dict__)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("dl-alpha-bench Workstation")
        self.resize(1280, 820)

        self.config_path = QLineEdit("configs/experiment_sample.yaml")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.dataset_summary = QPlainTextEdit()
        self.dataset_summary.setReadOnly(True)
        self.status_label = QLabel("Status: idle")
        self.failure_label = QLabel("Failure: -")

        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])

        self.explain_table = QTableWidget(0, 8)
        self.explain_table.setHorizontalHeaderLabels(
            [
                "Feature",
                "IC",
                "RankIC",
                "IC_IR",
                "RankIC_IR",
                "QSpread",
                "QPosRatio",
                "Obs",
            ]
        )

        self.compare_table = QTableWidget(0, 3)
        self.compare_table.setHorizontalHeaderLabels(["Experiment", "IC", "Sharpe"])

        tabs = QTabWidget()
        tabs.addTab(self._build_data_page(), "Data")
        tabs.addTab(self._build_dataset_page(), "Dataset")
        tabs.addTab(self._build_experiment_page(), "Experiment")
        tabs.addTab(self._build_monitor_page(), "Monitor")
        tabs.addTab(self._build_backtest_page(), "Backtest")
        tabs.addTab(self._build_explainability_page(), "Explainability")
        tabs.addTab(self._build_compare_page(), "Compare")

        self.setCentralWidget(tabs)
        self.worker: ExperimentWorker | None = None
        self.refresh_config_summary()
        self.refresh_compare_table()

    def _build_data_page(self) -> QWidget:
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.addWidget(QLabel("Source"), 0, 0)
        layout.addWidget(QLabel("joinquant / ricequant / csv / mock"), 0, 1)
        layout.addWidget(QLabel("Credential policy"), 1, 0)
        layout.addWidget(QLabel("Use local .env only, never commit secrets"), 1, 1)
        layout.addWidget(QLabel("Corporate actions"), 2, 0)
        layout.addWidget(QLabel("Adjuster hook is enabled in DatasetBuilder"), 2, 1)
        return widget

    def _build_dataset_page(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("Dataset summary from current config"))
        layout.addWidget(self.dataset_summary)
        return widget

    def _build_experiment_page(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Config"))
        path_layout.addWidget(self.config_path)
        btn_choose = QPushButton("Browse")
        btn_choose.clicked.connect(self.choose_config)
        path_layout.addWidget(btn_choose)

        btn_run = QPushButton("Run Experiment")
        btn_run.clicked.connect(self.run_experiment)

        layout.addLayout(path_layout)
        layout.addWidget(btn_run)
        layout.addWidget(QLabel("Tip: start from configs/experiment_sample.yaml"))
        return widget

    def _build_monitor_page(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.status_label)
        layout.addWidget(self.failure_label)
        layout.addWidget(QLabel("Run logs"))
        layout.addWidget(self.log)
        return widget

    def _build_backtest_page(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("Backtest summary"))
        layout.addWidget(self.metrics_table)
        return widget

    def _build_explainability_page(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("High-frequency factor explainability"))
        btn_load_latest = QPushButton("Load Latest Artifact")
        btn_load_latest.clicked.connect(self.load_latest_explainability)
        layout.addWidget(btn_load_latest)
        layout.addWidget(self.explain_table)
        return widget

    def _build_compare_page(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh_compare_table)
        layout.addWidget(btn_refresh)
        layout.addWidget(self.compare_table)
        return widget

    def choose_config(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select config",
            str(Path.cwd()),
            "YAML files (*.yaml *.yml)",
        )
        if selected:
            self.config_path.setText(selected)
            self.refresh_config_summary()

    def run_experiment(self) -> None:
        path = self.config_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Missing config", "Please provide a config path")
            return

        self.refresh_config_summary()
        self.log.appendPlainText(f"[RUN] {path}")
        self.worker = ExperimentWorker(path)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_finished(self, payload: dict) -> None:
        status = str(payload.get("status", "success"))
        failure = payload.get("failure_reason")
        self.status_label.setText(f"Status: {status}")
        self.failure_label.setText(f"Failure: {failure or '-'}")
        self.log.appendPlainText("[DONE] Experiment completed")
        self.log.appendPlainText(json.dumps(payload, indent=2, ensure_ascii=False))
        self._fill_metric_table(payload)
        self._fill_explainability_table(payload)
        self.refresh_compare_table()

    def _on_failed(self, message: str) -> None:
        self.status_label.setText("Status: failed")
        self.failure_label.setText(f"Failure: {message}")
        self.log.appendPlainText(f"[ERROR] {message}")
        QMessageBox.critical(self, "Experiment failed", message)

    def _fill_metric_table(self, payload: dict) -> None:
        merged = {}
        merged.update(payload.get("metrics", {}))
        merged.update(payload.get("backtest", {}))
        merged["status"] = payload.get("status", "success")
        if payload.get("failure_reason"):
            merged["failure_reason"] = payload.get("failure_reason")

        self.metrics_table.setRowCount(len(merged))
        for row, (key, val) in enumerate(merged.items()):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(str(key)))
            if isinstance(val, (int, float)):
                val_text = f"{val:.6f}"
            else:
                val_text = str(val)
            self.metrics_table.setItem(row, 1, QTableWidgetItem(val_text))

    def _fill_explainability_table(self, payload: dict) -> None:
        rows = payload.get("feature_explainability", [])
        if not isinstance(rows, list):
            rows = []

        self.explain_table.setRowCount(len(rows))
        for row_id, row in enumerate(rows):
            feature = str(row.get("feature", ""))
            ic_val = self._format_float(row.get("ic_mean"))
            rank_ic = self._format_float(row.get("rank_ic_mean"))
            ic_ir = self._format_float(row.get("ic_ir"))
            rank_ic_ir = self._format_float(row.get("rank_ic_ir"))
            q_spread = self._format_float(row.get("quantile_spread_mean"))
            q_pos = self._format_float(row.get("quantile_positive_ratio"))
            obs = str(row.get("n_obs", ""))

            self.explain_table.setItem(row_id, 0, QTableWidgetItem(feature))
            self.explain_table.setItem(row_id, 1, QTableWidgetItem(ic_val))
            self.explain_table.setItem(row_id, 2, QTableWidgetItem(rank_ic))
            self.explain_table.setItem(row_id, 3, QTableWidgetItem(ic_ir))
            self.explain_table.setItem(row_id, 4, QTableWidgetItem(rank_ic_ir))
            self.explain_table.setItem(row_id, 5, QTableWidgetItem(q_spread))
            self.explain_table.setItem(row_id, 6, QTableWidgetItem(q_pos))
            self.explain_table.setItem(row_id, 7, QTableWidgetItem(obs))

    def load_latest_explainability(self) -> None:
        art = Path("artifacts")
        latest_path: Path | None = None
        latest_ts = -1.0
        if not art.exists():
            return
        for result_path in art.glob("*/result.json"):
            try:
                mtime = result_path.stat().st_mtime
            except OSError:
                continue
            if mtime > latest_ts:
                latest_ts = mtime
                latest_path = result_path
        if latest_path is None:
            return
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
            self._fill_explainability_table(payload)
            self.log.appendPlainText(f"[LOAD] explainability from {latest_path}")
        except Exception as exc:  # pragma: no cover
            self.log.appendPlainText(f"[WARN] load explainability failed: {exc}")

    def _format_float(self, value: object) -> str:
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return ""

    def _safe_float(self, value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def refresh_config_summary(self) -> None:
        path = self.config_path.text().strip()
        if not path:
            self.dataset_summary.setPlainText("No config selected.")
            return
        try:
            cfg = load_yaml(path)
        except Exception as exc:  # pragma: no cover
            self.dataset_summary.setPlainText(f"Failed to load config: {exc}")
            return

        data_cfg = cfg.get("data", {})
        ds_cfg = cfg.get("dataset", {})
        cv_cfg = cfg.get("cv", {})
        lines = [
            f"config: {path}",
            f"data.source: {data_cfg.get('source', 'mock')}",
            f"data.freq: {data_cfg.get('freq', '1d')}",
            f"data.symbols: {len(data_cfg.get('symbols', []))}",
            f"feature_generators: {', '.join(ds_cfg.get('feature_generators', []))}",
            f"feature_columns: {', '.join(ds_cfg.get('feature_columns', []))}",
            f"label_horizons: {ds_cfg.get('label_horizons', [1])}",
            f"mask_column: {ds_cfg.get('mask_column', 'tradable')}",
            f"cv.method: {cv_cfg.get('method', 'purged_kfold')}",
        ]
        self.dataset_summary.setPlainText("\\n".join(lines))

    def refresh_compare_table(self) -> None:
        art = Path("artifacts")
        rows: list[tuple[str, float, float]] = []
        if art.exists():
            for result_path in art.glob("*/result.json"):
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    metrics = payload.get("metrics") or {}
                    backtest = payload.get("backtest") or {}
                    rows.append(
                        (
                            payload.get("experiment_id", result_path.parent.name),
                            self._safe_float(metrics.get("ic_mean"), 0.0),
                            self._safe_float(backtest.get("sharpe"), 0.0),
                        )
                    )
                except Exception:
                    continue

        rows.sort(key=lambda r: r[1], reverse=True)
        self.compare_table.setRowCount(len(rows))
        for i, (exp_id, ic_val, sharpe) in enumerate(rows):
            self.compare_table.setItem(i, 0, QTableWidgetItem(exp_id))
            self.compare_table.setItem(i, 1, QTableWidgetItem(f"{ic_val:.6f}"))
            self.compare_table.setItem(i, 2, QTableWidgetItem(f"{sharpe:.6f}"))


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(
        """
        QWidget { background-color: #11141a; color: #d8dee9; font-size: 13px; }
        QLineEdit, QPlainTextEdit, QTableWidget {
            background-color: #1d2330;
            border: 1px solid #33415c;
        }
        QPushButton { background-color: #2e3a59; border: 1px solid #3f5279; padding: 6px 10px; }
        QPushButton:hover { background-color: #3a4a70; }
        QHeaderView::section { background-color: #1a2030; color: #d8dee9; }
        """
    )
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
