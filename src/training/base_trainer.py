"""Abstract contract for interchangeable model-training implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ModelTrainer(ABC):
    """Define the contract every trainable model implementation must satisfy.

    Implementations own all algorithm-specific concerns (hyperparameters,
    the underlying library, how ``fit``/``predict`` are actually performed).
    The rest of the system only ever depends on this contract, so swapping
    Random Forest for XGBoost, LightGBM, or a neural network means writing a
    new class here, without touching the splitting, evaluation, or scripts
    that consume it.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the stable identifier for this trainer implementation.

        Returns:
            A stable identifier used for logging and persisted-model
            provenance, such as ``"random_forest"``.
        """
        ...

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model on a training slice.

        Args:
            X: Feature matrix, one row per timestamp.
            y: Target aligned with ``X`` by index.

        Raises:
            ValueError: If ``X`` and ``y`` do not share the same index, or
                either is empty.
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the target for each row of ``X``.

        Args:
            X: Feature matrix to predict on. Must have the same columns the
                model was fitted with.

        Returns:
            Predictions indexed the same way as ``X``.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        ...


__all__ = ["ModelTrainer"]
