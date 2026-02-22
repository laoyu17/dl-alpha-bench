"""Simple experiment tracking with YAML + JSON artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dl_alpha_bench.utils.config import dump_yaml
from dl_alpha_bench.utils.time_utils import utc_now_iso


@dataclass
class ExperimentResult:
    experiment_id: str
    config_hash: str
    seed: int
    metrics: dict[str, float]
    feature_explainability: list[dict[str, Any]]
    leakage_passed: bool
    leakage_details: list[str]
    backtest: dict[str, float]
    artifact_dir: str
    status: str = "success"
    failure_reason: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None


class ExperimentTracker:
    def __init__(self, root: str | Path = "artifacts"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def start(self, experiment_id: str) -> Path:
        run_id = self._make_run_id()
        run_dir = self.root / experiment_id / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        latest_file = self.root / experiment_id / "latest_run.txt"
        latest_file.write_text(run_id, encoding="utf-8")
        return run_dir

    def log_config(self, run_dir: Path, config: dict[str, Any]) -> None:
        dump_yaml(config, run_dir / "config.yaml")

    def log_explainability(self, run_dir: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with (run_dir / "feature_explainability.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def log_result(self, run_dir: Path, result: ExperimentResult) -> None:
        payload = asdict(result)
        payload["created_at"] = utc_now_iso()
        with (run_dir / "result.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def _make_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        suffix = uuid4().hex[:8]
        return f"{timestamp}-{suffix}"
