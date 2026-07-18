"""Tests for WalkForwardEvaluator, using a fake trainer to isolate the splitter wiring."""

from __future__ import annotations

import unittest
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.training import PurgedWalkForwardSplit, RandomForestTrainer, WalkForwardEvaluator
from src.training.base_trainer import ModelTrainer


class _FakeTrainer(ModelTrainer):
    """Test double that predicts zero and records what it was fitted on."""

    def __init__(self) -> None:
        self.fit_calls: List[Tuple[pd.Index, pd.Index]] = []
        self._fitted = False

    @property
    def model_id(self) -> str:
        return "fake"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.fit_calls.append((X.index, y.index))
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("fit() must be called before predict()")
        return pd.Series(0.0, index=X.index)


def _dataset(n: int = 80):
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(seed=0)
    X = pd.DataFrame({"feature": rng.normal(size=n)}, index=index)
    y = pd.Series(rng.normal(size=n), index=index, name="label")
    return X, y


class WalkForwardEvaluatorFakeTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instances: List[_FakeTrainer] = []

        def factory() -> _FakeTrainer:
            trainer = _FakeTrainer()
            self.instances.append(trainer)
            return trainer

        self.splitter = PurgedWalkForwardSplit(n_splits=3, embargo=0)
        self.evaluator = WalkForwardEvaluator(splitter=self.splitter, trainer_factory=factory)

    def test_requests_one_fresh_trainer_per_fold(self) -> None:
        X, y = _dataset()
        results = self.evaluator.evaluate(X, y)

        self.assertEqual(len(results), 3)
        self.assertEqual(len(self.instances), 3)
        for trainer in self.instances:
            self.assertEqual(len(trainer.fit_calls), 1)

    def test_fold_sizes_match_the_splitter(self) -> None:
        X, y = _dataset()
        results = self.evaluator.evaluate(X, y)

        expected = list(self.splitter.split(X.index))
        for result, (train_idx, val_idx) in zip(results, expected):
            self.assertEqual(result.train_size, len(train_idx))
            self.assertEqual(result.val_size, len(val_idx))

    def test_metrics_match_zero_prediction_by_construction(self) -> None:
        X, y = _dataset()
        results = self.evaluator.evaluate(X, y)

        for result, (_, val_idx) in zip(results, self.splitter.split(X.index)):
            y_val = y.iloc[val_idx]
            self.assertAlmostEqual(result.mae, float(np.mean(np.abs(y_val))))
            self.assertAlmostEqual(result.rmse, float(np.sqrt(np.mean(y_val ** 2))))
            self.assertEqual(result.directional_accuracy, 0.0)

    def test_raises_on_empty(self) -> None:
        X, y = _dataset()
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(X.iloc[:0], y.iloc[:0])

    def test_raises_on_mismatched_index(self) -> None:
        X, y = _dataset()
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(X, y.iloc[:-1])


class WalkForwardEvaluatorRandomForestIntegrationTest(unittest.TestCase):
    def test_end_to_end_beats_chance_on_a_learnable_signal(self) -> None:
        n = 300
        index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        rng = np.random.default_rng(seed=1)
        signal = rng.normal(size=n)
        X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)}, index=index)
        y = pd.Series(signal * 0.02 + rng.normal(scale=0.001, size=n), index=index, name="label")

        splitter = PurgedWalkForwardSplit(n_splits=3, embargo=1)
        evaluator = WalkForwardEvaluator(
            splitter=splitter,
            trainer_factory=lambda: RandomForestTrainer(
                n_estimators=50, max_depth=4, min_samples_leaf=5
            ),
        )

        results = evaluator.evaluate(X, y)

        self.assertEqual(len(results), 3)
        mean_directional_accuracy = np.mean([r.directional_accuracy for r in results])
        self.assertGreater(mean_directional_accuracy, 0.5)


if __name__ == "__main__":
    unittest.main()
