"""Technical-indicator feature calculator backed by pandas-ta-classic."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from src.domain import MarketBar

from .base_feature_calculator import FeatureCalculator
from .indicator_config import IndicatorConfig


class TechnicalIndicatorCalculator(FeatureCalculator):
    """Compute classic technical indicators from normalized OHLCV bars.

    pandas and pandas-ta-classic are implementation details confined to
    this class; callers only exchange ``MarketBar`` domain objects and the
    resulting DataFrame.

    Args:
        config: Indicator windows and parameters to use.
    """

    def __init__(self, config: IndicatorConfig = IndicatorConfig()) -> None:
        self._config = config

    def compute(self, bars: Sequence[MarketBar]) -> pd.DataFrame:
        """Compute the feature matrix for a chronological series of bars.

        Args:
            bars: Bars to compute features from, in any order.

        Returns:
            A DataFrame indexed by timezone-aware UTC ``timestamp``, sorted
            chronologically, containing OHLCV columns plus one column per
            configured indicator, plus cyclical hour-of-day/day-of-week
            encodings and one-hot trading-session flags derived from the
            UTC timestamp. Rows before an indicator's lookback window has
            been filled hold ``NaN`` for that indicator. Moving averages and
            Bollinger Bands are expressed relative to price (``sma_*_dist``,
            ``ema_*_dist``, ``bb_percent_b``, ``bb_bandwidth``) rather than
            in absolute price units, so they stay comparable across
            instruments with different price scales instead of duplicating
            the raw close price as a feature.

        Raises:
            ValueError: If ``bars`` is empty.
        """
        if not bars:
            raise ValueError("bars must not be empty")

        df = self._bars_to_frame(bars)
        config = self._config

        for period in config.sma_periods:
            sma = self._as_series(ta.sma(df["close"], length=period), df.index)
            df[f"sma_{period}_dist"] = (df["close"] - sma) / sma
        for period in config.ema_periods:
            ema = self._as_series(ta.ema(df["close"], length=period), df.index)
            df[f"ema_{period}_dist"] = (df["close"] - ema) / ema

        df[f"rsi_{config.rsi_period}"] = self._as_series(
            ta.rsi(df["close"], length=config.rsi_period), df.index
        )

        macd = ta.macd(
            df["close"],
            fast=config.macd_fast,
            slow=config.macd_slow,
            signal=config.macd_signal,
        )
        if macd is None:
            df["macd"] = float("nan")
            df["macd_hist"] = float("nan")
            df["macd_signal"] = float("nan")
        else:
            macd_columns = list(macd.columns)
            df["macd"] = macd[macd_columns[0]]
            df["macd_hist"] = macd[macd_columns[1]]
            df["macd_signal"] = macd[macd_columns[2]]

        bbands = ta.bbands(df["close"], length=config.bbands_period, std=config.bbands_std)
        if bbands is None:
            df["bb_percent_b"] = float("nan")
            df["bb_bandwidth"] = float("nan")
        else:
            bb_lower = self._column_starting_with(bbands, "BBL")
            bb_mid = self._column_starting_with(bbands, "BBM")
            bb_upper = self._column_starting_with(bbands, "BBU")
            df["bb_percent_b"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
            df["bb_bandwidth"] = (bb_upper - bb_lower) / bb_mid

        df[f"atr_{config.atr_period}"] = self._as_series(
            ta.atr(df["high"], df["low"], df["close"], length=config.atr_period),
            df.index,
        )

        stoch = ta.stoch(
            df["high"],
            df["low"],
            df["close"],
            k=config.stoch_k,
            d=config.stoch_d,
            smooth_k=config.stoch_smooth_k,
        )
        if stoch is None:
            df["stoch_k"] = float("nan")
            df["stoch_d"] = float("nan")
        else:
            stoch_columns = list(stoch.columns)
            df["stoch_k"] = stoch[stoch_columns[0]]
            df["stoch_d"] = stoch[stoch_columns[1]]

        adx = ta.adx(df["high"], df["low"], df["close"], length=config.adx_period)
        if adx is None:
            df[f"adx_{config.adx_period}"] = float("nan")
            df["plus_di"] = float("nan")
            df["minus_di"] = float("nan")
        else:
            adx_columns = list(adx.columns)
            df[f"adx_{config.adx_period}"] = adx[adx_columns[0]]
            df["plus_di"] = adx[adx_columns[1]]
            df["minus_di"] = adx[adx_columns[2]]

        df["obv"] = self._as_series(ta.obv(df["close"], df["volume"]), df.index)

        df[f"mfi_{config.mfi_period}"] = self._as_series(
            ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=config.mfi_period),
            df.index,
        )

        for period in config.return_periods:
            df[f"return_{period}"] = np.log(df["close"] / df["close"].shift(period))

        df[f"rel_volume_{config.rel_volume_period}"] = df["volume"] / self._as_series(
            ta.sma(df["volume"], length=config.rel_volume_period), df.index
        )

        hour = pd.Series(df.index.hour, index=df.index)
        day_of_week = pd.Series(df.index.dayofweek, index=df.index)

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

        df["session_asia"] = self._in_session_hours(hour, config.asia_session_hours)
        df["session_europe"] = self._in_session_hours(hour, config.europe_session_hours)
        df["session_us"] = self._in_session_hours(hour, config.us_session_hours)
        df["session_off"] = 1 - df[
            ["session_asia", "session_europe", "session_us"]
        ].max(axis=1)

        return df

    @staticmethod
    def _bars_to_frame(bars: Sequence[MarketBar]) -> pd.DataFrame:
        """Convert domain bars into an OHLCV DataFrame indexed by timestamp."""
        ordered: List[MarketBar] = sorted(bars, key=lambda bar: bar.timestamp)
        df = pd.DataFrame(
            {
                "timestamp": [bar.timestamp for bar in ordered],
                "open": [bar.open for bar in ordered],
                "high": [bar.high for bar in ordered],
                "low": [bar.low for bar in ordered],
                "close": [bar.close for bar in ordered],
                "volume": [bar.volume for bar in ordered],
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")

    @staticmethod
    def _in_session_hours(hour: pd.Series, session_hours: tuple) -> pd.Series:
        """Flag rows whose UTC hour falls in a ``(start, end)`` range, end exclusive."""
        start, end = session_hours
        return hour.between(start, end - 1).astype(int)

    @staticmethod
    def _column_starting_with(frame: pd.DataFrame, prefix: str) -> pd.Series:
        """Return the single column of ``frame`` whose name starts with ``prefix``."""
        matches = [column for column in frame.columns if column.startswith(prefix)]
        return frame[matches[0]]

    @staticmethod
    def _as_series(result: Optional[pd.Series], index: pd.Index) -> pd.Series:
        """Coerce an indicator result to a Series, filling ``NaN`` when unavailable.

        pandas-ta-classic returns ``None`` instead of a partially-``NaN``
        Series when there are fewer bars than an indicator's lookback
        window requires.
        """
        if result is None:
            return pd.Series(float("nan"), index=index)
        return result


__all__ = ["TechnicalIndicatorCalculator"]
