"""Tests for MultiTimeframeFeatureCalculator."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import List, Sequence

import pandas as pd

from src.domain import MarketBar, TimeFrame
from src.features import FeatureCalculator, MultiTimeframeFeatureCalculator


def _bar(ts: datetime, close: float) -> MarketBar:
    return MarketBar(timestamp=ts, open=close, high=close, low=close, close=close, volume=1.0)


class _FakeCalculator(FeatureCalculator):
    """Returns one column per name, each equal to the bar's close.

    Lets tests control exact indicator values and column names without
    depending on pandas-ta-classic's lookback behavior.
    """

    def __init__(self, column_names: Sequence[str]) -> None:
        self._column_names = column_names

    def compute(self, bars: Sequence[MarketBar]) -> pd.DataFrame:
        ordered = sorted(bars, key=lambda bar: bar.timestamp)
        index = pd.DatetimeIndex([bar.timestamp for bar in ordered], tz="UTC")
        data = {name: [bar.close for bar in ordered] for name in self._column_names}
        return pd.DataFrame(data, index=index)


class MultiTimeframeFeatureCalculatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def _base_bars(self, count: int) -> List[MarketBar]:
        return [
            _bar(self.base_start + timedelta(minutes=15 * i), close=100.0 + i)
            for i in range(count)
        ]

    def test_compute_preserves_base_columns_and_adds_context_columns(self) -> None:
        base_bars = self._base_bars(20)
        context_bars = [_bar(self.base_start, close=42.0)]
        calculator = MultiTimeframeFeatureCalculator(
            base_calculator=_FakeCalculator(["base_feature"]),
            context_calculator=_FakeCalculator(["rsi_2", "sma_5_dist"]),
            context_bars={TimeFrame.FOUR_HOURS: context_bars},
        )

        df = calculator.compute(base_bars)

        self.assertIn("base_feature", df.columns)
        self.assertIn("rsi_2_4h", df.columns)
        self.assertIn("sma_5_dist_4h", df.columns)
        self.assertEqual(len(df), len(base_bars))

    def test_context_column_is_nan_before_first_context_bar_closes(self) -> None:
        base_bars = self._base_bars(20)  # spans 0..4h45m
        context_bars = [_bar(self.base_start, close=42.0)]  # closes at +4h
        calculator = MultiTimeframeFeatureCalculator(
            base_calculator=_FakeCalculator(["base_feature"]),
            context_calculator=_FakeCalculator(["rsi_2", "sma_5_dist"]),
            context_bars={TimeFrame.FOUR_HOURS: context_bars},
        )

        df = calculator.compute(base_bars)

        close_time = self.base_start + timedelta(hours=4)
        before = df.loc[df.index < close_time, "rsi_2_4h"]
        at_and_after = df.loc[df.index >= close_time, "rsi_2_4h"]
        self.assertTrue(before.isna().all())
        self.assertFalse(at_and_after.isna().any())
        self.assertTrue((at_and_after == 42.0).all())

    def test_context_value_updates_at_next_context_bar_close(self) -> None:
        base_bars = self._base_bars(40)  # spans 0..9h45m
        context_bars = [
            _bar(self.base_start, close=42.0),  # closes at +4h
            _bar(self.base_start + timedelta(hours=4), close=99.0),  # closes at +8h
        ]
        calculator = MultiTimeframeFeatureCalculator(
            base_calculator=_FakeCalculator(["base_feature"]),
            context_calculator=_FakeCalculator(["rsi_2", "sma_5_dist"]),
            context_bars={TimeFrame.FOUR_HOURS: context_bars},
        )

        df = calculator.compute(base_bars)

        first_close = self.base_start + timedelta(hours=4)
        second_close = self.base_start + timedelta(hours=8)
        between = df.loc[(df.index >= first_close) & (df.index < second_close), "rsi_2_4h"]
        after = df.loc[df.index >= second_close, "rsi_2_4h"]
        self.assertTrue((between == 42.0).all())
        self.assertTrue((after == 99.0).all())

    def test_trend_column_uses_longest_sma_dist_period(self) -> None:
        base_bars = self._base_bars(20)
        context_bars = [_bar(self.base_start, close=42.0)]
        calculator = MultiTimeframeFeatureCalculator(
            base_calculator=_FakeCalculator(["base_feature"]),
            context_calculator=_FakeCalculator(["rsi_2", "sma_5_dist", "sma_200_dist"]),
            context_bars={TimeFrame.FOUR_HOURS: context_bars},
        )

        df = calculator.compute(base_bars)

        self.assertIn("sma_200_dist_4h", df.columns)
        self.assertNotIn("sma_5_dist_4h", df.columns)

    def test_compute_merges_multiple_context_timeframes_independently(self) -> None:
        base_bars = self._base_bars(20)  # spans 0..4h45m, never reaches the 1d close
        context_4h = [_bar(self.base_start, close=42.0)]
        context_1d = [_bar(self.base_start, close=7.0)]
        calculator = MultiTimeframeFeatureCalculator(
            base_calculator=_FakeCalculator(["base_feature"]),
            context_calculator=_FakeCalculator(["rsi_2", "sma_5_dist"]),
            context_bars={TimeFrame.FOUR_HOURS: context_4h, TimeFrame.ONE_DAY: context_1d},
        )

        df = calculator.compute(base_bars)

        self.assertIn("rsi_2_4h", df.columns)
        self.assertIn("rsi_2_1d", df.columns)
        after_4h_close = df.loc[df.index >= self.base_start + timedelta(hours=4), "rsi_2_4h"]
        self.assertTrue((after_4h_close == 42.0).all())
        self.assertTrue(df["rsi_2_1d"].isna().all())


if __name__ == "__main__":
    unittest.main()
