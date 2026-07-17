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
        stoch_k: Lookback window for the Stochastic Oscillator's %K line.
        stoch_d: Smoothing window for the Stochastic Oscillator's %D line.
        stoch_smooth_k: Smoothing window applied to %K before %D is derived.
        adx_period: Window size for the Average Directional Index and DI+/DI-.
        mfi_period: Window size for the Money Flow Index.
        return_periods: Lookback windows (in bars) for past log returns.
        rel_volume_period: Window size for the relative-volume moving average.
        asia_session_hours: ``(start, end)`` UTC hour range, end exclusive,
            for the Asia trading session.
        europe_session_hours: ``(start, end)`` UTC hour range, end exclusive,
            for the Europe trading session.
        us_session_hours: ``(start, end)`` UTC hour range, end exclusive, for
            the US trading session. UTC hours not covered by any configured
            session are treated as off-session.
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
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth_k: int = 3
    adx_period: int = 14
    mfi_period: int = 14
    return_periods: Tuple[int, ...] = field(default=(1, 3, 5, 10))
    rel_volume_period: int = 20
    asia_session_hours: Tuple[int, int] = (0, 8)
    europe_session_hours: Tuple[int, int] = (8, 13)
    us_session_hours: Tuple[int, int] = (13, 21)


__all__ = ["IndicatorConfig"]
