"""Tests for FundingRateFeatureCalculator."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import List, Sequence

import pandas as pd

from src.domain import FundingRate, MarketBar
from src.features import FeatureCalculator, FundingRateFeatureCalculator


def _bar(ts: datetime, close: float) -> MarketBar:
    return MarketBar(timestamp=ts, open=close, high=close, low=close, close=close, volume=1.0)


def _funding_rate(ts: datetime, rate: float) -> FundingRate:
    return FundingRate(timestamp=ts, rate=rate)


class _FakeCalculator(FeatureCalculator):
    """Returns one deterministic column keyed by bar timestamp."""

    def compute(self, bars: Sequence[MarketBar]) -> pd.DataFrame:
        ordered = sorted(bars, key=lambda bar: bar.timestamp)
        index = pd.DatetimeIndex([bar.timestamp for bar in ordered], tz="UTC")
        return pd.DataFrame({"base_feature": [bar.close for bar in ordered]}, index=index)


class FundingRateFeatureCalculatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def _base_bars(self, count: int) -> List[MarketBar]:
        return [
            _bar(self.base_start + timedelta(minutes=15 * i), close=100.0 + i)
            for i in range(count)
        ]

    def test_compute_preserves_base_columns_and_adds_funding_columns(self) -> None:
        base_bars = self._base_bars(4)
        funding_rates = [_funding_rate(self.base_start, 0.0001)]
        calculator = FundingRateFeatureCalculator(
            base_calculator=_FakeCalculator(),
            funding_rates=funding_rates,
            lookback_periods=3,
        )

        df = calculator.compute(base_bars)

        self.assertIn("base_feature", df.columns)
        self.assertIn("funding_rate", df.columns)
        self.assertIn("funding_rate_cum_3", df.columns)
        self.assertEqual(len(df), len(base_bars))

    def test_no_time_shift_settlement_is_known_at_its_own_timestamp(self) -> None:
        # Base bars every 15 minutes; a settlement exactly at bar index 4
        # (60 minutes in) must already be visible on that same row, unlike
        # MultiTimeframeFeatureCalculator's shifted-to-close-time context.
        base_bars = self._base_bars(8)
        settlement_time = self.base_start + timedelta(minutes=60)
        funding_rates = [_funding_rate(settlement_time, 0.0002)]
        calculator = FundingRateFeatureCalculator(
            base_calculator=_FakeCalculator(),
            funding_rates=funding_rates,
            lookback_periods=1,
        )

        df = calculator.compute(base_bars)

        before = df.loc[df.index < settlement_time, "funding_rate"]
        at_and_after = df.loc[df.index >= settlement_time, "funding_rate"]
        self.assertTrue(before.isna().all())
        self.assertTrue((at_and_after == 0.0002).all())

    def test_cumulative_column_sums_last_n_settlements(self) -> None:
        base_bars = self._base_bars(4)
        funding_rates = [
            _funding_rate(self.base_start, 0.0001),
            _funding_rate(self.base_start + timedelta(minutes=15), 0.0002),
            _funding_rate(self.base_start + timedelta(minutes=30), 0.0003),
        ]
        calculator = FundingRateFeatureCalculator(
            base_calculator=_FakeCalculator(),
            funding_rates=funding_rates,
            lookback_periods=3,
        )

        df = calculator.compute(base_bars)

        last_row = df.iloc[-1]
        self.assertAlmostEqual(last_row["funding_rate"], 0.0003)
        self.assertAlmostEqual(last_row["funding_rate_cum_3"], 0.0001 + 0.0002 + 0.0003)

    def test_cumulative_column_is_nan_until_lookback_periods_are_available(self) -> None:
        base_bars = self._base_bars(4)
        funding_rates = [
            _funding_rate(self.base_start, 0.0001),
            _funding_rate(self.base_start + timedelta(minutes=15), 0.0002),
        ]
        calculator = FundingRateFeatureCalculator(
            base_calculator=_FakeCalculator(),
            funding_rates=funding_rates,
            lookback_periods=3,
        )

        df = calculator.compute(base_bars)

        # Only 2 settlements exist total, so a 3-period cumulative sum is
        # never available anywhere in this dataset.
        self.assertTrue(df["funding_rate_cum_3"].isna().all())

    def test_rows_before_first_settlement_are_nan(self) -> None:
        base_bars = self._base_bars(4)
        first_settlement = self.base_start + timedelta(minutes=30)
        funding_rates = [_funding_rate(first_settlement, 0.0001)]
        calculator = FundingRateFeatureCalculator(
            base_calculator=_FakeCalculator(),
            funding_rates=funding_rates,
            lookback_periods=1,
        )

        df = calculator.compute(base_bars)

        before = df.loc[df.index < first_settlement, "funding_rate"]
        self.assertTrue(before.isna().all())


if __name__ == "__main__":
    unittest.main()
