"""Canonical OHLCV aggregation intervals for the business domain."""

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
