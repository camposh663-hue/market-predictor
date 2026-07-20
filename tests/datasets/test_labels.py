"""Tests for realized_volatility_label."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from src.datasets import realized_volatility_label


class RealizedVolatilityLabelTests(unittest.TestCase):
    def test_matches_hand_calculated_forward_realized_volatility(self) -> None:
        closes = [100.0, 110.0, 121.0, 108.9, 119.79]
        df = pd.DataFrame({"close": closes})

        result = realized_volatility_label(df, bars_per_horizon=2)

        # Row 0's horizon covers rows 1-2: returns ln(110/100) and ln(121/110).
        r1 = math.log(110.0 / 100.0)
        r2 = math.log(121.0 / 110.0)
        expected_0 = math.sqrt(r1**2 + r2**2)
        self.assertAlmostEqual(result.iloc[0], expected_0)

        # Row 1's horizon covers rows 2-3: returns ln(121/110) and ln(108.9/121).
        r3 = math.log(108.9 / 121.0)
        expected_1 = math.sqrt(r2**2 + r3**2)
        self.assertAlmostEqual(result.iloc[1], expected_1)

    def test_row_zero_has_no_lookback_nan_because_the_window_looks_forward(self) -> None:
        # Row 0's horizon window is [1, bars_per_horizon] -- it never needs
        # row 0's own return (close.shift(1) is NaN there), so unlike the
        # default signed-return label, this label has no leading NaN at all;
        # only trailing rows (see below) lack a full forward window.
        df = pd.DataFrame({"close": [100.0, 110.0, 121.0, 108.9]})

        result = realized_volatility_label(df, bars_per_horizon=1)

        self.assertFalse(math.isnan(result.iloc[0]))

    def test_trailing_rows_are_nan_without_a_full_forward_window(self) -> None:
        df = pd.DataFrame({"close": [100.0, 110.0, 121.0, 108.9, 119.79]})

        result = realized_volatility_label(df, bars_per_horizon=2)

        self.assertTrue(math.isnan(result.iloc[-1]))
        self.assertTrue(math.isnan(result.iloc[-2]))

    def test_result_is_never_negative(self) -> None:
        df = pd.DataFrame({"close": [100.0, 95.0, 102.0, 98.0, 110.0, 90.0]})

        result = realized_volatility_label(df, bars_per_horizon=2)

        self.assertTrue((result.dropna() >= 0).all())


if __name__ == "__main__":
    unittest.main()
