"""Evaluation metrics for alpha models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dl_alpha_bench.train import FoldTrainResult


@dataclass
class MetricSummary:
    ic_mean: float
    rank_ic_mean: float
    rmse_mean: float



def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    if std_a == 0.0 or std_b == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])



def information_coefficient(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return _safe_corr(pred, truth)



def rank_information_coefficient(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    pred_rank = np.argsort(np.argsort(pred))
    truth_rank = np.argsort(np.argsort(truth))
    return _safe_corr(pred_rank, truth_rank)



def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))



def summarize_fold_metrics(folds: list[FoldTrainResult]) -> MetricSummary:
    ics = [information_coefficient(fr.valid_pred, fr.valid_y) for fr in folds]
    rank_ics = [rank_information_coefficient(fr.valid_pred, fr.valid_y) for fr in folds]
    rmses = [rmse(fr.valid_pred, fr.valid_y) for fr in folds]
    return MetricSummary(
        ic_mean=float(np.nanmean(ics)) if ics else float("nan"),
        rank_ic_mean=float(np.nanmean(rank_ics)) if rank_ics else float("nan"),
        rmse_mean=float(np.nanmean(rmses)) if rmses else float("nan"),
    )
