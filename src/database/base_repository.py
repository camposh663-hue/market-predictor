"""Abstract contract for provider-agnostic market-bar storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Sequence

from src.domain import Instrument, MarketBar, TimeFrame


class BarRepository(ABC):
    """Define the contract for reading and writing normalized OHLCV bars.

    Implementations own all storage-specific concerns, including file
    layout, partitioning, and serialization format. They only exchange
    domain objects with the rest of the application.
    """

    @abstractmethod
    def write_bars(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
        bars: Sequence[MarketBar],
    ) -> None:
        """Persist bars for an instrument and timeframe.

        Args:
            instrument: Instrument the bars belong to.
            timeframe: Aggregation interval of the bars.
            bars: Bars to persist. Implementations must not assume the bars
                are sorted or free of duplicate timestamps.
        """
        ...

    @abstractmethod
    def read_bars(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> Sequence[MarketBar]:
        """Retrieve stored bars for an instrument within a time range.

        Args:
            instrument: Instrument whose bars are requested.
            timeframe: Aggregation interval of the bars.
            start: Inclusive, timezone-aware UTC start timestamp.
            end: Exclusive, timezone-aware UTC end timestamp.

        Returns:
            Stored bars ordered chronologically from oldest to newest.
        """
        ...

    @abstractmethod
    def latest_timestamp(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
    ) -> Optional[datetime]:
        """Return the timestamp of the most recent stored bar, if any.

        Args:
            instrument: Instrument to inspect.
            timeframe: Aggregation interval to inspect.

        Returns:
            The opening timestamp of the newest stored bar, or ``None`` when
            no bars are stored for the instrument and timeframe.
        """
        ...


__all__ = ["BarRepository"]
