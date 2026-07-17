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
    stoch_k=3,
    stoch_d=2,
    stoch_smooth_k=1,
    adx_period=3,
    mfi_period=3,
    return_periods=(1, 2),
    rel_volume_period=3,
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
            "sma_3_dist",
            "sma_5_dist",
            "ema_3_dist",
            "rsi_3",
            "macd",
            "macd_hist",
            "macd_signal",
            "bb_percent_b",
            "bb_bandwidth",
            "atr_3",
            "stoch_k",
            "stoch_d",
            "adx_3",
            "plus_di",
            "minus_di",
            "obv",
            "mfi_3",
            "return_1",
            "return_2",
            "rel_volume_3",
        }
        self.assertTrue(expected.issubset(set(df.columns)))

    def test_compute_index_is_chronological_utc_regardless_of_input_order(self) -> None:
        bars = _bars([100 + i for i in range(15)])
        shuffled = [bars[3], bars[0], bars[2], bars[1]] + bars[4:]

        df = self.calculator.compute(shuffled)

        self.assertEqual(list(df.index), sorted(bar.timestamp for bar in bars))
        self.assertEqual(str(df.index.tz), "UTC")

    def test_sma_dist_matches_hand_calculated_distance_from_price(self) -> None:
        calculator = TechnicalIndicatorCalculator(config=IndicatorConfig(sma_periods=(2,)))
        bars = _bars([10.0, 20.0, 30.0, 40.0])

        df = calculator.compute(bars)

        self.assertTrue(math.isnan(df["sma_2_dist"].iloc[0]))
        # sma_2 = [nan, 15, 25, 35]; dist = (close - sma) / sma
        expected = [(20.0 - 15.0) / 15.0, (30.0 - 25.0) / 25.0, (40.0 - 35.0) / 35.0]
        for actual, want in zip(df["sma_2_dist"].iloc[1:], expected):
            self.assertAlmostEqual(actual, want)

    def test_indicators_are_nan_when_fewer_bars_than_lookback_window(self) -> None:
        calculator = TechnicalIndicatorCalculator(config=IndicatorConfig(sma_periods=(2,)))
        bars = _bars([10.0, 20.0, 30.0, 40.0])

        df = calculator.compute(bars)

        self.assertTrue(df["macd"].isna().all())
        self.assertTrue(df["bb_percent_b"].isna().all())
        self.assertTrue(df["atr_14"].isna().all())
        self.assertTrue(df["adx_14"].isna().all())
        self.assertTrue(df["mfi_14"].isna().all())

    def test_return_matches_hand_calculated_log_return(self) -> None:
        calculator = TechnicalIndicatorCalculator(
            config=IndicatorConfig(return_periods=(1, 2))
        )
        bars = _bars([100.0, 110.0, 121.0])

        df = calculator.compute(bars)

        self.assertTrue(math.isnan(df["return_1"].iloc[0]))
        self.assertAlmostEqual(df["return_1"].iloc[1], math.log(110.0 / 100.0))
        self.assertAlmostEqual(df["return_1"].iloc[2], math.log(121.0 / 110.0))
        self.assertTrue(df["return_2"].iloc[:2].isna().all())
        self.assertAlmostEqual(df["return_2"].iloc[2], math.log(121.0 / 100.0))

    def test_compute_adds_expected_temporal_columns(self) -> None:
        bars = _bars([100 + i for i in range(15)])

        df = self.calculator.compute(bars)

        expected = {
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "session_asia",
            "session_europe",
            "session_us",
            "session_off",
        }
        self.assertTrue(expected.issubset(set(df.columns)))

    def test_hour_and_dow_cyclical_encodings_match_hand_calculated_values(self) -> None:
        # 2024-01-01T00:00:00Z is a Monday (dow=0); each bar advances one hour.
        bars = _bars([100.0, 101.0, 102.0])

        df = self.calculator.compute(bars)

        self.assertAlmostEqual(df["hour_sin"].iloc[0], math.sin(2 * math.pi * 0 / 24))
        self.assertAlmostEqual(df["hour_cos"].iloc[0], math.cos(2 * math.pi * 0 / 24))
        self.assertAlmostEqual(df["hour_sin"].iloc[2], math.sin(2 * math.pi * 2 / 24))
        self.assertAlmostEqual(df["dow_sin"].iloc[0], math.sin(2 * math.pi * 0 / 7))
        self.assertAlmostEqual(df["dow_cos"].iloc[0], math.cos(2 * math.pi * 0 / 7))

    def test_session_flags_are_one_hot_and_match_configured_hours(self) -> None:
        config = IndicatorConfig(
            asia_session_hours=(0, 8),
            europe_session_hours=(8, 13),
            us_session_hours=(13, 21),
        )
        calculator = TechnicalIndicatorCalculator(config=config)
        bars = _bars([100 + i for i in range(24)])

        df = calculator.compute(bars)

        session_columns = ["session_asia", "session_europe", "session_us", "session_off"]
        self.assertTrue((df[session_columns].sum(axis=1) == 1).all())
        self.assertEqual(df["session_asia"].iloc[0], 1)
        self.assertEqual(df["session_europe"].iloc[8], 1)
        self.assertEqual(df["session_us"].iloc[13], 1)
        self.assertEqual(df["session_off"].iloc[21], 1)

    def test_relative_volume_matches_hand_calculated_ratio(self) -> None:
        calculator = TechnicalIndicatorCalculator(
            config=IndicatorConfig(rel_volume_period=2)
        )
        bars = [
            _bar(0, close=100.0, volume=10.0),
            _bar(1, close=101.0, volume=20.0),
            _bar(2, close=102.0, volume=30.0),
        ]

        df = calculator.compute(bars)

        self.assertTrue(math.isnan(df["rel_volume_2"].iloc[0]))
        self.assertAlmostEqual(df["rel_volume_2"].iloc[1], 20.0 / 15.0)
        self.assertAlmostEqual(df["rel_volume_2"].iloc[2], 30.0 / 25.0)


if __name__ == "__main__":
    unittest.main()
