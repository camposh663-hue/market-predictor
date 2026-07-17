"""Tests for DatasetBuilder."""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone
from typing import List

from src.datasets import DatasetBuilder
from src.domain import MarketBar, TimeFrame
from src.features import IndicatorConfig, TechnicalIndicatorCalculator


def _bar(hour: int, close: float) -> MarketBar:
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hour)
    return MarketBar(
        timestamp=timestamp,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=10.0,
    )


def _bars(closes: List[float]) -> List[MarketBar]:
    return [_bar(i, close) for i, close in enumerate(closes)]


_TINY_CONFIG = IndicatorConfig(
    sma_periods=(2,),
    ema_periods=(2,),
    rsi_period=2,
    macd_fast=2,
    macd_slow=3,
    macd_signal=2,
    bbands_period=2,
    bbands_std=2.0,
    atr_period=2,
    stoch_k=2,
    stoch_d=2,
    stoch_smooth_k=1,
    adx_period=2,
    mfi_period=2,
    return_periods=(1,),
    rel_volume_period=2,
)


class DatasetBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = DatasetBuilder(
            feature_calculator=TechnicalIndicatorCalculator(config=_TINY_CONFIG)
        )

    def test_build_raises_on_non_positive_horizon(self) -> None:
        bars = _bars([100.0 + i for i in range(10)])

        with self.assertRaises(ValueError):
            self.builder.build(bars, TimeFrame.ONE_HOUR, timedelta(0))

    def test_build_raises_when_horizon_not_multiple_of_bar_duration(self) -> None:
        bars = _bars([100.0 + i for i in range(10)])

        with self.assertRaises(ValueError):
            self.builder.build(bars, TimeFrame.ONE_HOUR, timedelta(minutes=30))

    def test_build_raises_on_empty_bars(self) -> None:
        with self.assertRaises(ValueError):
            self.builder.build([], TimeFrame.ONE_HOUR, timedelta(hours=1))

    def test_build_x_excludes_raw_ohlcv_and_label_columns(self) -> None:
        bars = _bars([100.0 + i for i in range(30)])

        x, _ = self.builder.build(bars, TimeFrame.ONE_HOUR, timedelta(hours=2))

        for column in ("open", "high", "low", "close", "volume", "label"):
            self.assertNotIn(column, x.columns)
        self.assertIn("rsi_2", x.columns)
        self.assertIn("hour_sin", x.columns)

    def test_build_x_and_y_share_index_and_contain_no_nan(self) -> None:
        bars = _bars([100.0 + i for i in range(30)])

        x, y = self.builder.build(bars, TimeFrame.ONE_HOUR, timedelta(hours=2))

        self.assertTrue(x.index.equals(y.index))
        self.assertFalse(x.isna().any().any())
        self.assertFalse(y.isna().any())
        self.assertGreater(len(x), 0)

    def test_build_drops_tail_rows_without_a_future_label(self) -> None:
        bars = _bars([100.0 + i for i in range(30)])
        horizon = timedelta(hours=3)

        x, _ = self.builder.build(bars, TimeFrame.ONE_HOUR, horizon)

        for bar in bars[-3:]:
            self.assertNotIn(bar.timestamp, x.index)
        self.assertIn(bars[-4].timestamp, x.index)

    def test_label_matches_hand_calculated_log_return(self) -> None:
        closes = [100.0 + i for i in range(30)]
        bars = _bars(closes)
        horizon = timedelta(hours=2)

        _, y = self.builder.build(bars, TimeFrame.ONE_HOUR, horizon)

        probe_hour = 15
        probe_timestamp = bars[probe_hour].timestamp
        expected = math.log(closes[probe_hour + 2] / closes[probe_hour])
        self.assertAlmostEqual(y.loc[probe_timestamp], expected)


if __name__ == "__main__":
    unittest.main()
