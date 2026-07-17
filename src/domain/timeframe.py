"""Canonical OHLCV aggregation intervals for the business domain."""

from __future__ import annotations

from datetime import timedelta
from enum import Enum


class TimeFrame(str, Enum):
    """Enumerate the bar intervals supported by the initial platform version.

    Providers receive this domain value and translate it to their own API
    interval representation. Code outside provider implementations must use
    enum members instead of raw timeframe strings.

    Attributes:
        ONE_MINUTE: One-minute bars.
        FIVE_MINUTES: Five-minute bars.
        FIFTEEN_MINUTES: Fifteen-minute bars.
        ONE_HOUR: One-hour bars.
        FOUR_HOURS: Four-hour bars.
        ONE_DAY: One-day bars.
    """

    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"

    @property
    def duration(self) -> timedelta:
        """Return the wall-clock duration spanned by one bar of this interval."""
        return _BAR_DURATIONS[self]


_BAR_DURATIONS = {
    TimeFrame.ONE_MINUTE: timedelta(minutes=1),
    TimeFrame.FIVE_MINUTES: timedelta(minutes=5),
    TimeFrame.FIFTEEN_MINUTES: timedelta(minutes=15),
    TimeFrame.ONE_HOUR: timedelta(hours=1),
    TimeFrame.FOUR_HOURS: timedelta(hours=4),
    TimeFrame.ONE_DAY: timedelta(days=1),
}
