from __future__ import annotations

import json
import sys
from pathlib import Path

from dl_alpha_bench import cli
from dl_alpha_bench.utils.config import dump_yaml, load_yaml


def test_cli_main_outputs_json_payload(tmp_path, monkeypatch) -> None:
    cfg = load_yaml("configs/experiment_sample.yaml")
    cfg["experiment_id"] = "pytest-cli-exp"
    cfg["data"]["periods"] = 80
    cfg_path = tmp_path / "cli-config.yaml"
    dump_yaml(cfg, cfg_path)
    monkeypatch.setenv("DL_ALPHA_BENCH_ARTIFACT_DIR", str(tmp_path))

    old_argv = sys.argv[:]
    try:
        sys.argv = ["dl-alpha-bench", "--config", str(cfg_path)]
        cli.main()
    finally:
        sys.argv = old_argv

    result_candidates = sorted((tmp_path / cfg["experiment_id"] / "runs").glob("*/result.json"))
    assert result_candidates
    result_path = Path(result_candidates[-1])
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "pytest-cli-exp"
    assert payload["status"] == "success"
