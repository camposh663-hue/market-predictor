"""Technical-indicator feature calculator backed by pandas-ta-classic."""

from __future__ import annotations

from typing import List, Optional, Sequence

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
            configured indicator. Rows before an indicator's lookback
            window has been filled hold ``NaN`` for that indicator.

        Raises:
            ValueError: If ``bars`` is empty.
        """
        if not bars:
            raise ValueError("bars must not be empty")

        df = self._bars_to_frame(bars)
        config = self._config

        for period in config.sma_periods:
            df[f"sma_{period}"] = self._as_series(
                ta.sma(df["close"], length=period), df.index
            )
        for period in config.ema_periods:
            df[f"ema_{period}"] = self._as_series(
                ta.ema(df["close"], length=period), df.index
            )

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
            df["bb_lower"] = float("nan")
            df["bb_mid"] = float("nan")
            df["bb_upper"] = float("nan")
        else:
            df["bb_lower"] = self._column_starting_with(bbands, "BBL")
            df["bb_mid"] = self._column_starting_with(bbands, "BBM")
            df["bb_upper"] = self._column_starting_with(bbands, "BBU")

        df[f"atr_{config.atr_period}"] = self._as_series(
            ta.atr(df["high"], df["low"], df["close"], length=config.atr_period),
            df.index,
        )

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
