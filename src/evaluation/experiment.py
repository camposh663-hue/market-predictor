"""Immutable descriptions of one experiment and its dev-set validation result."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping, Tuple

from src.domain import TimeFrame
from src.training import FoldResult


@dataclass(frozen=True)
class ExperimentConfig:
    """One point in the (model x hyperparameters x timeframe x horizon) grid.

    Attributes:
        model_name: Key into the model registry's trainer factories and
            hyperparameter grids, e.g. ``"random_forest"`` or ``"xgboost"``.
        timeframe: Bar aggregation interval to build the dataset from.
        horizon: Prediction horizon, must be an exact multiple of
            ``timeframe``'s bar duration (the same constraint
            ``DatasetBuilder`` enforces).
        hyperparams: Keyword arguments passed to the model's trainer
            factory for this configuration.
    """

    model_name: str
    timeframe: TimeFrame
    horizon: timedelta
    hyperparams: Mapping[str, Any]

    def label(self) -> str:
        """Return a short, human-readable identifier for logging and reports."""
        params = ", ".join(f"{k}={v}" for k, v in sorted(self.hyperparams.items()))
        return f"{self.model_name}[{self.timeframe.value}->{self.horizon}]({params})"


@dataclass(frozen=True)
class ExperimentResult:
    """Aggregated dev-set walk-forward outcome for one ExperimentConfig.

    Never reflects the held-out test set: everything here comes from
    ``WalkForwardEvaluator`` folds run on the dev slice only, so this result
    is safe to use for model/hyperparameter selection without touching the
    final holdout.

    Attributes:
        config: The configuration this result was produced from.
        fold_results: Per-fold metrics, in chronological order.
        mean_directional_accuracy: Mean of ``directional_accuracy`` across
            folds -- the primary ranking criterion.
        std_directional_accuracy: Standard deviation across folds, used as
            a tie-breaker: a lower spread means the result is more likely to
            hold up across market regimes rather than being a lucky fold.
        mean_mae: Mean absolute error across folds.
        mean_rmse: Root mean squared error across folds.
    """

    config: ExperimentConfig
    fold_results: Tuple[FoldResult, ...]
    mean_directional_accuracy: float
    std_directional_accuracy: float
    mean_mae: float
    mean_rmse: float


__all__ = ["ExperimentConfig", "ExperimentResult"]
