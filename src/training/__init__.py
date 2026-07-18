"""Model training: fit and validate interchangeable models on (X, y) datasets."""

from .base_trainer import ModelTrainer
from .random_forest_trainer import RandomForestTrainer
from .time_series_split import PurgedWalkForwardSplit
from .walk_forward_evaluator import FoldResult, WalkForwardEvaluator
from .xgboost_trainer import XGBoostTrainer

__all__ = [
    "ModelTrainer",
    "RandomForestTrainer",
    "XGBoostTrainer",
    "PurgedWalkForwardSplit",
    "FoldResult",
    "WalkForwardEvaluator",
]
