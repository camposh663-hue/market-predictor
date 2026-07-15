"""Configuration for classic technical indicator calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class IndicatorConfig:
    """Parameters controlling which indicators are computed and their windows.

    Attributes:
        sma_periods: Window sizes for the Simple Moving Average.
        ema_periods: Window sizes for the Exponential Moving Average.
        rsi_period: Window size for the Relative Strength Index.
        macd_fast: Fast EMA window for MACD.
        macd_slow: Slow EMA window for MACD.
        macd_signal: Signal-line EMA window for MACD.
        bbands_period: Window size for Bollinger Bands.
        bbands_std: Standard-deviation multiplier for the Bollinger Bands.
        atr_period: Window size for the Average True Range.
    """

    sma_periods: Tuple[int, ...] = field(default=(20, 50, 200))
    ema_periods: Tuple[int, ...] = field(default=(12, 26))
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bbands_period: int = 20
    bbands_std: float = 2.0
    atr_period: int = 14


__all__ = ["IndicatorConfig"]
