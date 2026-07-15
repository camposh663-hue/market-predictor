"""Tests for TechnicalIndicatorCalculator."""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from src.domain import MarketBar
from src.features import IndicatorConfig, TechnicalIndicatorCalculator


def _bar(hour: int, close: float, volume: Optional[float] = 10.0) -> MarketBar:
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hour)
    return MarketBar(
        timestamp=timestamp,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=volume,
    )


def _bars(closes: List[float]) -> List[MarketBar]:
    return [_bar(i, close) for i, close in enumerate(closes)]


_SMALL_CONFIG = IndicatorConfig(
    sma_periods=(3, 5),
    ema_periods=(3,),
    rsi_period=3,
    macd_fast=3,
    macd_slow=5,
    macd_signal=2,
    bbands_period=3,
    bbands_std=2.0,
    atr_period=3,
)


class TechnicalIndicatorCalculatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calculator = TechnicalIndicatorCalculator(config=_SMALL_CONFIG)

    def test_compute_raises_on_empty_bars(self) -> None:
        with self.assertRaises(ValueError):
            self.calculator.compute([])

    def test_compute_preserves_ohlcv_and_row_count(self) -> None:
        bars = _bars([100 + i for i in range(15)])

        df = self.calculator.compute(bars)

        self.assertEqual(len(df), len(bars))
        for column in ("open", "high", "low", "close", "volume"):
            self.assertIn(column, df.columns)

    def test_compute_adds_expected_indicator_columns(self) -> None:
        bars = _bars([100 + i for i in range(15)])

        df = self.calculator.compute(bars)

        expected = {
            "sma_3",
            "sma_5",
            "ema_3",
            "rsi_3",
            "macd",
            "macd_hist",
            "macd_signal",
            "bb_lower",
            "bb_mid",
            "bb_upper",
            "atr_3",
        }
        self.assertTrue(expected.issubset(set(df.columns)))

    def test_compute_index_is_chronological_utc_regardless_of_input_order(self) -> None:
        bars = _bars([100 + i for i in range(15)])
        shuffled = [bars[3], bars[0], bars[2], bars[1]] + bars[4:]

        df = self.calculator.compute(shuffled)

        self.assertEqual(list(df.index), sorted(bar.timestamp for bar in bars))
        self.assertEqual(str(df.index.tz), "UTC")

    def test_sma_matches_hand_calculated_average(self) -> None:
        calculator = TechnicalIndicatorCalculator(config=IndicatorConfig(sma_periods=(2,)))
        bars = _bars([10.0, 20.0, 30.0, 40.0])

        df = calculator.compute(bars)

        self.assertTrue(math.isnan(df["sma_2"].iloc[0]))
        self.assertEqual(list(df["sma_2"].iloc[1:]), [15.0, 25.0, 35.0])

    def test_indicators_are_nan_when_fewer_bars_than_lookback_window(self) -> None:
        calculator = TechnicalIndicatorCalculator(config=IndicatorConfig(sma_periods=(2,)))
        bars = _bars([10.0, 20.0, 30.0, 40.0])

        df = calculator.compute(bars)

        self.assertTrue(df["macd"].isna().all())
        self.assertTrue(df["bb_lower"].isna().all())
        self.assertTrue(df["atr_14"].isna().all())


if __name__ == "__main__":
    unittest.main()
