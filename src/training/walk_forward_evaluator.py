"""Orchestrate purged walk-forward cross-validation over a ModelTrainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.training.base_trainer import ModelTrainer
from src.training.metrics import directional_accuracy
from src.training.time_series_split import PurgedWalkForwardSplit


@dataclass(frozen=True)
class FoldResult:
    """Metrics produced by one walk-forward fold.

    Attributes:
        fold: One-indexed fold number, in chronological order.
        train_size: Number of rows the model was trained on.
        val_size: Number of rows the model was evaluated on.
        mae: Mean absolute error of predicted vs. true log-return.
        rmse: Root mean squared error of predicted vs. true log-return.
        directional_accuracy: Fraction of rows where the predicted sign
            matched the true sign.
    """

    fold: int
    train_size: int
    val_size: int
    mae: float
    rmse: float
    directional_accuracy: float


class WalkForwardEvaluator:
    """Fit and score a fresh model per fold of a PurgedWalkForwardSplit.

    A new, untrained model is requested from ``trainer_factory`` for every
    fold: reusing one model across folds would let it accumulate state from
    validation windows it should never have trained on, silently
    reintroducing the leakage the splitter is designed to prevent.

    Args:
        splitter: Produces embargoed, expanding-window train/validation
            folds from a chronological index.
        trainer_factory: Called once per fold to produce a fresh, unfitted
            ``ModelTrainer``.
    """

    def __init__(
        self,
        splitter: PurgedWalkForwardSplit,
        trainer_factory: Callable[[], ModelTrainer],
    ) -> None:
        self._splitter = splitter
        self._trainer_factory = trainer_factory

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> List[FoldResult]:
        """Run every walk-forward fold and collect its metrics.

        Args:
            X: Feature matrix, chronologically sorted.
            y: Target aligned with ``X`` by index.

        Returns:
            One ``FoldResult`` per fold, in chronological order.

        Raises:
            ValueError: If ``X`` and ``y`` do not share the same index, or
                either is empty.
        """
        if X.empty or y.empty:
            raise ValueError("X and y must not be empty")
        if not X.index.equals(y.index):
            raise ValueError("X and y must share the same index")

        results: List[FoldResult] = []
        for fold, (train_pos, val_pos) in enumerate(self._splitter.split(X.index), start=1):
            X_train, y_train = X.iloc[train_pos], y.iloc[train_pos]
            X_val, y_val = X.iloc[val_pos], y.iloc[val_pos]

            trainer = self._trainer_factory()
            trainer.fit(X_train, y_train)
            predictions = trainer.predict(X_val)

            results.append(
                FoldResult(
                    fold=fold,
                    train_size=len(X_train),
                    val_size=len(X_val),
                    mae=mean_absolute_error(y_val, predictions),
                    rmse=mean_squared_error(y_val, predictions) ** 0.5,
                    directional_accuracy=directional_accuracy(y_val, predictions),
                )
            )

        return results


__all__ = ["FoldResult", "WalkForwardEvaluator"]
