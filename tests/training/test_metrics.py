"""Tests for directional_accuracy."""

from __future__ import annotations

import unittest

import pandas as pd

from src.training.metrics import directional_accuracy


def _series(values: list) -> pd.Series:
    return pd.Series(values, index=pd.RangeIndex(len(values)))


class DirectionalAccuracyTests(unittest.TestCase):
    def test_perfect_match_scores_one(self) -> None:
        y_true = _series([1.0, -2.0, 3.0, -4.0])
        y_pred = _series([0.5, -0.1, 2.0, -0.01])

        self.assertEqual(directional_accuracy(y_true, y_pred), 1.0)

    def test_all_opposite_scores_zero(self) -> None:
        y_true = _series([1.0, -2.0, 3.0, -4.0])
        y_pred = _series([-0.5, 0.1, -2.0, 0.01])

        self.assertEqual(directional_accuracy(y_true, y_pred), 0.0)

    def test_partial_match(self) -> None:
        y_true = _series([1.0, -2.0, 3.0, -4.0])
        y_pred = _series([0.5, 0.1, 2.0, 0.01])

        self.assertEqual(directional_accuracy(y_true, y_pred), 0.5)

    def test_zero_prediction_never_counts_as_a_match(self) -> None:
        y_true = _series([1.0, -2.0])
        y_pred = _series([0.0, 0.0])

        self.assertEqual(directional_accuracy(y_true, y_pred), 0.0)

    def test_raises_on_mismatched_index(self) -> None:
        y_true = pd.Series([1.0, 2.0], index=[0, 1])
        y_pred = pd.Series([1.0, 2.0], index=[0, 2])

        with self.assertRaises(ValueError):
            directional_accuracy(y_true, y_pred)

    def test_raises_on_empty(self) -> None:
        with self.assertRaises(ValueError):
            directional_accuracy(_series([]), _series([]))


if __name__ == "__main__":
    unittest.main()
