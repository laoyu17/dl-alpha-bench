"""Model training over CV splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from dl_alpha_bench.cv import Split
from dl_alpha_bench.models import MLPConfig, NumpyMLPRegressor, Regressor
from dl_alpha_bench.utils.seed import set_global_seed


@dataclass
class TrainConfig:
    seed: int = 42
    hidden_dim: int = 32
    lr: float = 1e-2
    epochs: int = 100
    l2: float = 0.0


@dataclass
class FoldTrainResult:
    split_id: int
    train_loss_last: float
    valid_pred: np.ndarray
    valid_y: np.ndarray


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        model_factory: Callable[[int], Regressor] | None = None,
    ):
        self.config = config
        self.model_factory = model_factory or self._default_model_factory

    def fit_cv(
        self,
        x: np.ndarray,
        y: np.ndarray,
        splits: list[Split],
    ) -> list[FoldTrainResult]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        out: list[FoldTrainResult] = []

        for i, split in enumerate(splits):
            set_global_seed(self.config.seed + i)
            model = self.model_factory(self.config.seed + i)
            train_x = x[split.train_idx]
            train_y = y[split.train_idx]
            valid_x = x[split.valid_idx]
            valid_y = y[split.valid_idx]

            history = model.fit(train_x, train_y)
            pred = model.predict(valid_x)
            out.append(
                FoldTrainResult(
                    split_id=i,
                    train_loss_last=float(history[-1]) if history else float("nan"),
                    valid_pred=pred,
                    valid_y=valid_y,
                )
            )
        return out

    def _default_model_factory(self, seed: int) -> Regressor:
        conf = MLPConfig(
            hidden_dim=self.config.hidden_dim,
            lr=self.config.lr,
            epochs=self.config.epochs,
            seed=seed,
            l2=self.config.l2,
        )
        return NumpyMLPRegressor(conf)
