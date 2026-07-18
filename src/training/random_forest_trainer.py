"""Random Forest baseline implementation of the ModelTrainer contract."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.training.base_trainer import ModelTrainer


class RandomForestTrainer(ModelTrainer):
    """Random Forest regressor used as the first reference model.

    Chosen as the baseline over gradient-boosted trees or sequence models
    (LSTM, Transformers) because it needs no feature scaling, is robust to
    the residual multicollinearity still present between indicators, trains
    in seconds on the full BTC/USDT history, and exposes feature
    importances that let us sanity-check whether the engineered indicators
    carry real signal before investing in more complex, harder-to-debug
    architectures.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree. ``None`` grows nodes until
            leaves are pure, which overfits readily on noisy financial
            targets, so callers are expected to pass a bounded value.
        min_samples_leaf: Minimum samples required at a leaf node; higher
            values regularize against noise in the log-return target.
        random_state: Seed controlling bootstrap sampling and feature
            subsampling, for reproducible fits.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ) -> None:
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    @property
    def model_id(self) -> str:
        """Return the stable identifier for this trainer."""
        return "random_forest"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the Random Forest on a training slice.

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
        """Return per-feature importances from the fitted forest.

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


__all__ = ["RandomForestTrainer"]
