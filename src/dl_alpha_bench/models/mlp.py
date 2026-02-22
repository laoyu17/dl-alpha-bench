"""A tiny numpy MLP regressor for deterministic experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Regressor


@dataclass
class MLPConfig:
    hidden_dim: int = 32
    lr: float = 1e-2
    epochs: int = 100
    seed: int = 42
    l2: float = 0.0


class NumpyMLPRegressor(Regressor):
    def __init__(self, config: MLPConfig | None = None):
        self.config = config or MLPConfig()
        self._params: dict[str, np.ndarray] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> list[float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n, d = x.shape
        rng = np.random.default_rng(self.config.seed)

        w1 = rng.normal(scale=0.1, size=(d, self.config.hidden_dim))
        b1 = np.zeros((1, self.config.hidden_dim))
        w2 = rng.normal(scale=0.1, size=(self.config.hidden_dim, 1))
        b2 = np.zeros((1, 1))

        losses: list[float] = []
        for _ in range(self.config.epochs):
            h_pre = x @ w1 + b1
            h = np.maximum(h_pre, 0.0)
            pred = h @ w2 + b2

            err = pred - y
            mse = float(np.mean(err**2))
            reg = self.config.l2 * (float(np.sum(w1**2)) + float(np.sum(w2**2)))
            loss = mse + reg
            losses.append(loss)

            grad_pred = 2.0 * err / n
            grad_w2 = h.T @ grad_pred + 2 * self.config.l2 * w2
            grad_b2 = np.sum(grad_pred, axis=0, keepdims=True)

            grad_h = grad_pred @ w2.T
            grad_h_pre = grad_h * (h_pre > 0)
            grad_w1 = x.T @ grad_h_pre + 2 * self.config.l2 * w1
            grad_b1 = np.sum(grad_h_pre, axis=0, keepdims=True)

            w2 -= self.config.lr * grad_w2
            b2 -= self.config.lr * grad_b2
            w1 -= self.config.lr * grad_w1
            b1 -= self.config.lr * grad_b1

        self._params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
        return losses

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._params:
            raise RuntimeError("model is not fitted")
        x = np.asarray(x, dtype=float)
        w1 = self._params["w1"]
        b1 = self._params["b1"]
        w2 = self._params["w2"]
        b2 = self._params["b2"]
        h = np.maximum(x @ w1 + b1, 0.0)
        pred = h @ w2 + b2
        return pred.ravel()
