"""Normalized perpetual-futures funding-rate entity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FundingRate:
    """Represent one settled funding-rate payment for a perpetual future.

    Attributes:
        timestamp: Settlement time as a timezone-aware UTC datetime. Unlike
            a bar's opening time, this is the exact instant the rate becomes
            known -- no additional duration needs to be added to find when
            it is knowable.
        rate: Funding rate as a fraction (e.g. ``0.0001`` for 0.01%), paid by
            longs to shorts when positive.
    """

    timestamp: datetime
    rate: float
