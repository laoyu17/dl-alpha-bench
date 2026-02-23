#!/usr/bin/env python3
"""Generate README UI demo screenshots and GIF from the PyQt workstation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from PIL import Image
from PyQt6.QtWidgets import QApplication, QTabWidget

from dl_alpha_bench.gui.app import MainWindow

DEFAULT_RESULT_CANDIDATES = [
    Path("artifacts/microstructure-mock-exp/result.json"),
    Path("artifacts/sample-mock-exp/result.json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Optional result.json path used to fill metrics/explainability tables",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/readme"),
        help="Directory for exported README assets",
    )
    parser.add_argument(
        "--config-path",
        default="configs/experiment_microstructure_mock.yaml",
        help="Config path shown in the Experiment tab",
    )
    parser.add_argument(
        "--gif-width",
        type=int,
        default=960,
        help="GIF output width in pixels",
    )
    return parser.parse_args()


def load_payload(explicit_path: Path | None) -> dict:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.extend(DEFAULT_RESULT_CANDIDATES)

    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

    fallback = sorted(Path("artifacts").glob("*/runs/*/result.json"))
    if fallback:
        return json.loads(fallback[-1].read_text(encoding="utf-8"))

    raise FileNotFoundError("No result.json found under artifacts/")


def apply_style(app: QApplication) -> None:
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


def capture_tab(window: MainWindow, tabs: QTabWidget, index: int, out: Path) -> None:
    tabs.setCurrentIndex(index)
    QApplication.processEvents()
    if not window.grab().save(str(out)):
        raise RuntimeError(f"Failed to save screenshot: {out}")


def build_gif(paths: Iterable[Path], out_path: Path, width: int) -> None:
    frames = []
    for path in paths:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        new_h = int(rgb.height * (width / rgb.width))
        resized = rgb.resize((width, new_h), Image.Resampling.LANCZOS)
        frames.append(resized)

    durations = [1300, 1200, 1200, 1200, 1300]
    palette_frames = [
        frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for frame in frames
    ]
    palette_frames[0].save(
        out_path,
        save_all=True,
        append_images=palette_frames[1:],
        loop=0,
        duration=durations,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication([])
    apply_style(app)

    window = MainWindow()
    window.show()
    QApplication.processEvents()

    window.config_path.setText(args.config_path)
    window.refresh_config_summary()

    payload = load_payload(args.result_json)
    window._on_worker_log(f"loading config: {args.config_path}")
    window._on_worker_log("build dataset: 18000 rows, 7 features")
    window._on_worker_log("run complete, writing artifacts")
    window._on_finished(payload)
    window.load_latest_explainability()

    tabs = window.findChild(QTabWidget)
    if tabs is None:
        raise RuntimeError("QTabWidget not found")

    png_paths = [
        output_dir / "ui-01-experiment.png",
        output_dir / "ui-02-monitor.png",
        output_dir / "ui-03-backtest.png",
        output_dir / "ui-04-explainability.png",
        output_dir / "ui-05-compare.png",
    ]

    capture_tab(window, tabs, 2, png_paths[0])
    capture_tab(window, tabs, 3, png_paths[1])
    capture_tab(window, tabs, 4, png_paths[2])
    capture_tab(window, tabs, 5, png_paths[3])
    capture_tab(window, tabs, 6, png_paths[4])

    build_gif(png_paths, output_dir / "ui-demo.gif", width=args.gif_width)

    for path in [*png_paths, output_dir / "ui-demo.gif"]:
        print(f"saved {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
