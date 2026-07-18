"""Choose a confidence threshold using dev-only walk-forward folds.

The full-frequency backtest (every row traded) showed the per-trade edge is
far smaller than round-trip costs. This searches, strictly on dev data,
whether restricting trades to only the model's highest-magnitude
predictions -- fewer trades, presumably each with a larger average edge --
can turn that into a net-positive result. The holdout is never touched
here: selection happens entirely on dev folds, so a caller can still score
the single chosen fraction on the holdout exactly once afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.training import ModelTrainer, PurgedWalkForwardSplit

from .backtest import directional_backtest


@dataclass(frozen=True)
class ThresholdCandidateResult:
    """Aggregated dev-fold outcome for one confidence threshold.

    Attributes:
        keep_fraction: Fraction of each fold's rows kept as trades: only
            rows whose ``|prediction|`` ranks in the top ``keep_fraction``
            of that fold's validation predictions are traded; the rest are
            treated as flat (no position, no cost). ``1.0`` means every row
            is traded -- the unfiltered baseline.
        mean_trades_per_fold: Average number of trades taken per fold.
        mean_net_return: Mean, across folds, of each fold's net-of-cost
            mean per-trade return.
        std_net_return: Standard deviation across folds of that mean --
            lower means the result held up consistently across folds
            (different market regimes), not just one lucky fold.
    """

    keep_fraction: float
    mean_trades_per_fold: float
    mean_net_return: float
    std_net_return: float


def select_confidence_threshold(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    splitter: PurgedWalkForwardSplit,
    trainer_factory: Callable[[], ModelTrainer],
    keep_fractions: Sequence[float],
    cost_per_trade: float,
    min_trades_per_fold: int,
) -> Tuple[float, List[ThresholdCandidateResult]]:
    """Pick the keep_fraction with the best net return, using dev folds only.

    For every fold, a fresh model is fit on the fold's training rows and
    scored on its validation rows. For each candidate ``keep_fraction``,
    only the validation rows whose predicted magnitude ranks in that top
    fraction are treated as trades, scored with ``directional_backtest``.
    Every candidate's results are aggregated across every fold before any
    selection happens, so the decision reflects the full picture at once
    rather than being adjusted fold by fold.

    Args:
        X_dev: Dev-set feature matrix. The caller must already have
            excluded the holdout (e.g. via ``PurgedWalkForwardSplit.
            test_split``) before passing data here.
        y_dev: Target aligned with ``X_dev``.
        splitter: Produces the walk-forward folds within ``X_dev``.
        trainer_factory: Produces a fresh, unfitted model per fold -- reused
            model state across folds would leak information the same way
            reusing one across a plain walk-forward evaluation would.
        keep_fractions: Candidate fractions to evaluate, e.g. ``(1.0, 0.5,
            0.25, 0.1, 0.05)``. Include ``1.0`` (no filtering) as the
            baseline to compare against.
        cost_per_trade: Round-trip trading cost as a fraction, e.g. 0.002.
        min_trades_per_fold: Candidates averaging fewer trades per fold
            than this are excluded from selection: too few trades make the
            mean/std across folds unreliable no matter how good they look.

    Returns:
        A tuple ``(chosen_keep_fraction, candidate_results)``: the selected
        fraction (highest mean net return among candidates meeting
        ``min_trades_per_fold``) and the full aggregated table for every
        candidate, for transparency.

    Raises:
        ValueError: If no candidate meets ``min_trades_per_fold``.
    """
    fold_returns: Dict[float, List[float]] = {fraction: [] for fraction in keep_fractions}
    fold_trades: Dict[float, List[int]] = {fraction: [] for fraction in keep_fractions}

    for train_pos, val_pos in splitter.split(X_dev.index):
        X_train, y_train = X_dev.iloc[train_pos], y_dev.iloc[train_pos]
        X_val, y_val = X_dev.iloc[val_pos], y_dev.iloc[val_pos]

        trainer = trainer_factory()
        trainer.fit(X_train, y_train)
        predictions = trainer.predict(X_val)
        magnitude = predictions.abs()

        for fraction in keep_fractions:
            cutoff = magnitude.quantile(1.0 - fraction)
            mask = magnitude >= cutoff
            filtered_predictions = predictions.where(mask, 0.0)
            result = directional_backtest(y_val, filtered_predictions, cost_per_trade)
            fold_returns[fraction].append(result.net_mean_return)
            fold_trades[fraction].append(result.n_trades)

    candidate_results = [
        ThresholdCandidateResult(
            keep_fraction=fraction,
            mean_trades_per_fold=float(np.mean(fold_trades[fraction])),
            mean_net_return=float(np.mean(fold_returns[fraction])),
            std_net_return=float(np.std(fold_returns[fraction])),
        )
        for fraction in keep_fractions
    ]

    eligible = [c for c in candidate_results if c.mean_trades_per_fold >= min_trades_per_fold]
    if not eligible:
        raise ValueError(
            f"no keep_fraction produced at least {min_trades_per_fold} trades "
            "per fold on average"
        )

    best = max(eligible, key=lambda c: c.mean_net_return)
    return best.keep_fraction, candidate_results


__all__ = ["ThresholdCandidateResult", "select_confidence_threshold"]
