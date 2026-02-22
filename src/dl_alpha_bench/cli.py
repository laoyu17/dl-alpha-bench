"""Command line interface for dl-alpha-bench."""

from __future__ import annotations

import argparse
import json

from dl_alpha_bench.exp import ConfigValidationError, ExperimentRunner, LeakageGuardError
from dl_alpha_bench.utils.config import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leakage-safe DL alpha benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_yaml(args.config)
    try:
        result = ExperimentRunner().run(config)
    except (ConfigValidationError, LeakageGuardError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "failure_reason": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2) from exc
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
