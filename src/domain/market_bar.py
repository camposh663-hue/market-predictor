"""Normalized OHLCV market-bar entity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class MarketBar:
    """Represent one normalized OHLCV bar for a financial instrument.

    Attributes:
        timestamp: Opening time of the bar as a timezone-aware UTC datetime.
        open: Opening price for the bar.
        high: Highest traded or quoted price for the bar.
        low: Lowest traded or quoted price for the bar.
        close: Closing price for the bar.
        volume: Traded volume when the provider supplies a comparable value.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float]
