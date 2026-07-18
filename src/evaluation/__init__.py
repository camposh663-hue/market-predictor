"""Systematic, leakage-safe comparison of models, hyperparameters, and horizons."""

from .backtest import BacktestResult, directional_backtest
from .experiment import ExperimentConfig, ExperimentResult
from .experiment_runner import Dataset, ExperimentRunner, HoldoutResult
from .model_registry import HYPERPARAMETER_GRIDS, TRAINER_FACTORIES, hyperparameter_combinations
from .threshold_search import ThresholdCandidateResult, select_confidence_threshold

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "Dataset",
    "ExperimentRunner",
    "HoldoutResult",
    "HYPERPARAMETER_GRIDS",
    "TRAINER_FACTORIES",
    "hyperparameter_combinations",
    "BacktestResult",
    "directional_backtest",
    "ThresholdCandidateResult",
    "select_confidence_threshold",
]
