"""In-memory bar repository used for tests and local experimentation."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from src.domain import Instrument, MarketBar, TimeFrame

from .base_repository import BarRepository

_SeriesKey = Tuple[Instrument, TimeFrame]


class InMemoryRepository(BarRepository):
    """Store bars in a process-local dictionary.

    This repository never touches disk. It exists so that ``DataManager``
    and other collaborators can be tested without depending on a concrete
    storage backend such as Parquet.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory repository."""
        self._series: Dict[_SeriesKey, Dict[datetime, MarketBar]] = {}

    def write_bars(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
        bars: Sequence[MarketBar],
    ) -> None:
        """Persist bars, overwriting any existing bar with the same timestamp.

        Args:
            instrument: Instrument the bars belong to.
            timeframe: Aggregation interval of the bars.
            bars: Bars to persist.
        """
        series = self._series.setdefault((instrument, timeframe), {})
        for bar in bars:
            series[bar.timestamp] = bar

    def read_bars(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> Sequence[MarketBar]:
        """Return stored bars within ``[start, end)``, oldest first.

        Args:
            instrument: Instrument whose bars are requested.
            timeframe: Aggregation interval of the bars.
            start: Inclusive, timezone-aware UTC start timestamp.
            end: Exclusive, timezone-aware UTC end timestamp.

        Returns:
            Stored bars ordered chronologically from oldest to newest.
        """
        series = self._series.get((instrument, timeframe), {})
        matching: List[MarketBar] = [
            bar for timestamp, bar in series.items() if start <= timestamp < end
        ]
        matching.sort(key=lambda bar: bar.timestamp)
        return matching

    def latest_timestamp(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
    ) -> Optional[datetime]:
        """Return the newest stored timestamp, or ``None`` when empty.

        Args:
            instrument: Instrument to inspect.
            timeframe: Aggregation interval to inspect.

        Returns:
            The opening timestamp of the newest stored bar, or ``None``.
        """
        series = self._series.get((instrument, timeframe), {})
        if not series:
            return None
        return max(series)


__all__ = ["InMemoryRepository"]
