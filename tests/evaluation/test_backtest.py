"""Tests for directional_backtest."""

from __future__ import annotations

import unittest

import pandas as pd

from src.evaluation import directional_backtest


def _series(values: list) -> pd.Series:
    return pd.Series(values, index=pd.RangeIndex(len(values)))


class DirectionalBacktestTests(unittest.TestCase):
    def test_all_correct_direction_no_cost_matches_hand_calculation(self) -> None:
        y_true = _series([0.02, -0.01, 0.03])
        y_pred = _series([0.01, -0.02, 0.05])

        result = directional_backtest(y_true, y_pred, cost_per_trade=0.0)

        self.assertEqual(result.n_trades, 3)
        self.assertAlmostEqual(result.gross_total_return, 0.02 + 0.01 + 0.03)
        self.assertAlmostEqual(result.net_total_return, result.gross_total_return)
        self.assertEqual(result.win_rate, 1.0)

    def test_cost_is_subtracted_only_from_non_flat_trades(self) -> None:
        y_true = _series([0.02, -0.01, 0.03])
        y_pred = _series([0.01, -0.02, 0.0])

        result = directional_backtest(y_true, y_pred, cost_per_trade=0.001)

        self.assertEqual(result.n_trades, 2)
        expected_net_total = (0.02 - 0.001) + (0.01 - 0.001) + 0.0
        self.assertAlmostEqual(result.net_total_return, expected_net_total)
        self.assertAlmostEqual(result.gross_total_return, 0.02 + 0.01 + 0.0)

    def test_high_enough_cost_can_flip_a_winning_trade_to_a_loss(self) -> None:
        y_true = _series([0.001])
        y_pred = _series([0.5])

        result = directional_backtest(y_true, y_pred, cost_per_trade=0.002)

        self.assertLess(result.net_mean_return, 0.0)
        self.assertEqual(result.win_rate, 0.0)

    def test_zero_prediction_never_trades_and_never_pays_cost(self) -> None:
        y_true = _series([0.05, -0.05])
        y_pred = _series([0.0, 0.0])

        result = directional_backtest(y_true, y_pred, cost_per_trade=0.01)

        self.assertEqual(result.n_trades, 0)
        self.assertEqual(result.win_rate, 0.0)
        self.assertEqual(result.net_total_return, 0.0)

    def test_raises_on_empty(self) -> None:
        with self.assertRaises(ValueError):
            directional_backtest(_series([]), _series([]), cost_per_trade=0.001)

    def test_raises_on_mismatched_index(self) -> None:
        y_true = pd.Series([0.01, 0.02], index=[0, 1])
        y_pred = pd.Series([0.01, 0.02], index=[0, 2])

        with self.assertRaises(ValueError):
            directional_backtest(y_true, y_pred, cost_per_trade=0.001)

    def test_raises_on_negative_cost(self) -> None:
        y_true = _series([0.01])
        y_pred = _series([0.01])

        with self.assertRaises(ValueError):
            directional_backtest(y_true, y_pred, cost_per_trade=-0.001)


if __name__ == "__main__":
    unittest.main()
