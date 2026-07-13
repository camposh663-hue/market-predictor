"""Provider-agnostic orchestration for storing and retrieving market bars."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence

from src.domain import Instrument, MarketBar, TimeFrame

from .base_repository import BarRepository


class DataManager:
    """Coordinate storage and retrieval of normalized OHLCV bars.

    The DataManager never communicates with market-data providers and never
    depends on a concrete storage format. It only accepts ``MarketBar``
    objects, applies storage-independent business rules such as
    deduplication and chronological ordering, and delegates persistence to
    a ``BarRepository``.

    Args:
        repository: Storage backend used to persist and retrieve bars.
    """

    def __init__(self, repository: BarRepository) -> None:
        self._repository = repository

    def store_bars(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
        bars: Sequence[MarketBar],
    ) -> None:
        """Store new bars for an instrument and timeframe.

        Args:
            instrument: Instrument the bars belong to.
            timeframe: Aggregation interval of the bars.
            bars: Bars to store. Duplicate timestamps are resolved by
                keeping the last bar for each timestamp.

        Raises:
            ValueError: If ``bars`` is empty.
        """
        if not bars:
            raise ValueError("bars must not be empty")
        self._repository.write_bars(instrument, timeframe, self._deduplicate(bars))

    def get_bars(
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

        Raises:
            ValueError: If ``start`` is not strictly before ``end``.
        """
        if start >= end:
            raise ValueError("start must be strictly before end")
        return self._repository.read_bars(instrument, timeframe, start, end)

    def latest_timestamp(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
    ) -> Optional[datetime]:
        """Return the timestamp of the most recently stored bar, if any.

        Callers such as a provider-refresh use case can use this value to
        resume incremental downloads from the first missing bar instead of
        re-fetching the full history.

        Args:
            instrument: Instrument to inspect.
            timeframe: Aggregation interval to inspect.

        Returns:
            The opening timestamp of the newest stored bar, or ``None`` when
            no bars are stored for the instrument and timeframe.
        """
        return self._repository.latest_timestamp(instrument, timeframe)

    @staticmethod
    def _deduplicate(bars: Sequence[MarketBar]) -> List[MarketBar]:
        """Collapse bars sharing a timestamp and sort them chronologically.

        Args:
            bars: Bars to deduplicate, in any order.

        Returns:
            One bar per distinct timestamp, sorted oldest to newest. When two
            bars share a timestamp, the last one in ``bars`` wins.
        """
        by_timestamp: Dict[datetime, MarketBar] = {bar.timestamp: bar for bar in bars}
        return sorted(by_timestamp.values(), key=lambda bar: bar.timestamp)


__all__ = ["DataManager"]
