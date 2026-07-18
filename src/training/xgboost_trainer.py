"""Gradient-boosted trees implementation of the ModelTrainer contract."""

from __future__ import annotations

import pandas as pd
from xgboost import XGBRegressor

from src.training.base_trainer import ModelTrainer


class XGBoostTrainer(ModelTrainer):
    """Gradient-boosted tree regressor, compared against RandomForestTrainer.

    Unlike Random Forest's independent, averaged trees, gradient boosting
    fits trees sequentially, each one correcting the errors of the ones
    before it. This can capture more signal when it exists, but also
    overfits noisy financial targets more readily, so it needs its
    regularizing hyperparameters (``learning_rate``, ``subsample``,
    ``max_depth``) tuned rather than left at defaults -- which is exactly
    what the evaluation grid in ``src/evaluation/`` searches over.

    Args:
        n_estimators: Number of sequential boosting rounds (trees).
        max_depth: Maximum depth of each tree. Kept shallow by default
            because deeper trees combined with boosting overfit quickly.
        learning_rate: Shrinkage applied to each tree's contribution; lower
            values need more estimators but generalize better.
        subsample: Fraction of rows sampled per tree, for regularization.
        colsample_bytree: Fraction of features sampled per tree, for
            regularization.
        random_state: Seed controlling sampling, for reproducible fits.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    @property
    def model_id(self) -> str:
        """Return the stable identifier for this trainer."""
        return "xgboost"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the gradient-boosted trees on a training slice.

        Args:
            X: Feature matrix, one row per timestamp.
            y: Target aligned with ``X`` by index.

        Raises:
            ValueError: If ``X`` and ``y`` do not share the same index, or
                either is empty.
        """
        if X.empty or y.empty:
            raise ValueError("X and y must not be empty")
        if not X.index.equals(y.index):
            raise ValueError("X and y must share the same index")

        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the log-return target for each row of ``X``.

        Args:
            X: Feature matrix to predict on. Must have the same columns the
                model was fitted with.

        Returns:
            Predictions indexed the same way as ``X``.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not self._fitted:
            raise RuntimeError("fit() must be called before predict()")

        predictions = self._model.predict(X)
        return pd.Series(predictions, index=X.index, name="prediction")

    @property
    def feature_importances(self) -> pd.Series:
        """Return per-feature importances from the fitted model.

        Returns:
            Importances indexed by feature name, descending.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not self._fitted:
            raise RuntimeError("fit() must be called before feature_importances")

        return pd.Series(
            self._model.feature_importances_,
            index=self._model.feature_names_in_,
            name="importance",
        ).sort_values(ascending=False)


__all__ = ["XGBoostTrainer"]
