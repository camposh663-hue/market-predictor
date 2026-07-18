"""Tests for RandomForestTrainer."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.training import RandomForestTrainer


def _dataset(n: int = 60):
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(seed=0)
    X = pd.DataFrame(
        {"feature_a": rng.normal(size=n), "feature_b": rng.normal(size=n)},
        index=index,
    )
    y = pd.Series(X["feature_a"] * 0.01, index=index, name="label")
    return X, y


class RandomForestTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = RandomForestTrainer(n_estimators=10, max_depth=3, min_samples_leaf=2)

    def test_model_id(self) -> None:
        self.assertEqual(self.trainer.model_id, "random_forest")

    def test_predict_before_fit_raises(self) -> None:
        X, _ = _dataset()
        with self.assertRaises(RuntimeError):
            self.trainer.predict(X)

    def test_feature_importances_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            _ = self.trainer.feature_importances

    def test_fit_raises_on_empty(self) -> None:
        X, y = _dataset()
        with self.assertRaises(ValueError):
            self.trainer.fit(X.iloc[:0], y.iloc[:0])

    def test_fit_raises_on_mismatched_index(self) -> None:
        X, y = _dataset()
        with self.assertRaises(ValueError):
            self.trainer.fit(X, y.iloc[:-1])

    def test_fit_predict_returns_series_indexed_like_input(self) -> None:
        X, y = _dataset()
        self.trainer.fit(X, y)

        predictions = self.trainer.predict(X)

        self.assertIsInstance(predictions, pd.Series)
        self.assertTrue(predictions.index.equals(X.index))

    def test_feature_importances_sum_to_one_and_are_sorted_descending(self) -> None:
        X, y = _dataset()
        self.trainer.fit(X, y)

        importances = self.trainer.feature_importances

        self.assertAlmostEqual(importances.sum(), 1.0, places=6)
        self.assertEqual(list(importances.index), sorted(
            importances.index, key=lambda name: -importances[name]
        ))


if __name__ == "__main__":
    unittest.main()
