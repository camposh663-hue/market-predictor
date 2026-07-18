"""Tests for select_confidence_threshold, using a fixed-prediction test double."""

from __future__ import annotations

import unittest

import pandas as pd

from src.evaluation import select_confidence_threshold
from src.training import ModelTrainer, PurgedWalkForwardSplit


class _FixedPredictionTrainer(ModelTrainer):
    """Test double that ignores X/y content and always returns preset predictions."""

    def __init__(self, predictions: pd.Series) -> None:
        self._predictions = predictions
        self._fitted = False

    @property
    def model_id(self) -> str:
        return "fixed"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("fit() must be called before predict()")
        return self._predictions.loc[X.index]


def _dataset(n: int = 60):
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    # Even rows go up, odd rows go down.
    y_true = pd.Series([0.01 if i % 2 == 0 else -0.01 for i in range(n)], index=index)
    # Rows where i % 4 in (0, 2) get a large, correctly-signed prediction;
    # the rest get a tiny, wrongly-signed one. Exactly half of every
    # 20-row block is "large and correct", the other half "small and wrong".
    predictions = pd.Series(
        [0.05 if i % 4 in (0, 2) else 0.001 for i in range(n)], index=index
    )
    X = pd.DataFrame({"dummy": range(n)}, index=index)
    return X, y_true, predictions


class SelectConfidenceThresholdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.y, self.predictions = _dataset(60)
        self.splitter = PurgedWalkForwardSplit(n_splits=2, embargo=0)

    def _factory(self):
        return lambda: _FixedPredictionTrainer(self.predictions)

    def test_unfiltered_baseline_matches_hand_calculation(self) -> None:
        _, candidates = select_confidence_threshold(
            self.X,
            self.y,
            self.splitter,
            self._factory(),
            keep_fractions=(1.0,),
            cost_per_trade=0.005,
            min_trades_per_fold=1,
        )

        baseline = candidates[0]
        self.assertEqual(baseline.keep_fraction, 1.0)
        self.assertEqual(baseline.mean_trades_per_fold, 20.0)
        # 10 rows net +0.005 (0.01 - 0.005), 10 rows net -0.015 (-0.01 - 0.005).
        self.assertAlmostEqual(baseline.mean_net_return, -0.005)
        self.assertAlmostEqual(baseline.std_net_return, 0.0)

    def test_filtering_to_top_half_isolates_the_correct_large_predictions(self) -> None:
        _, candidates = select_confidence_threshold(
            self.X,
            self.y,
            self.splitter,
            self._factory(),
            keep_fractions=(0.5,),
            cost_per_trade=0.005,
            min_trades_per_fold=1,
        )

        filtered = candidates[0]
        self.assertEqual(filtered.mean_trades_per_fold, 10.0)
        # Only the large, correct predictions survive: net = 0.01 - 0.005.
        self.assertAlmostEqual(filtered.mean_net_return, 0.005)

    def test_selects_the_fraction_with_the_best_net_return(self) -> None:
        chosen, _ = select_confidence_threshold(
            self.X,
            self.y,
            self.splitter,
            self._factory(),
            keep_fractions=(1.0, 0.5),
            cost_per_trade=0.005,
            min_trades_per_fold=1,
        )

        self.assertEqual(chosen, 0.5)

    def test_excludes_candidates_below_min_trades_per_fold(self) -> None:
        chosen, candidates = select_confidence_threshold(
            self.X,
            self.y,
            self.splitter,
            self._factory(),
            keep_fractions=(1.0, 0.5),
            cost_per_trade=0.005,
            min_trades_per_fold=15,
        )

        # 0.5 would win on return alone (10 trades/fold) but is excluded,
        # leaving 1.0 (20 trades/fold) as the only eligible candidate.
        self.assertEqual(chosen, 1.0)
        self.assertEqual(len(candidates), 2)

    def test_raises_when_no_candidate_meets_min_trades_per_fold(self) -> None:
        with self.assertRaises(ValueError):
            select_confidence_threshold(
                self.X,
                self.y,
                self.splitter,
                self._factory(),
                keep_fractions=(1.0, 0.5),
                cost_per_trade=0.005,
                min_trades_per_fold=1000,
            )


if __name__ == "__main__":
    unittest.main()
