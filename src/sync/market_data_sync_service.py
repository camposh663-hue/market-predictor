"""Application service that keeps stored market bars in sync with a provider."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from src.database import DataManager
from src.domain import Instrument, MarketBar, TimeFrame
from src.providers import BaseProvider


class MarketDataSyncService:
    """Fetch missing market bars from a provider and persist them via a DataManager.

    This service is the only component allowed to depend on both a
    ``BaseProvider`` and a ``DataManager``. It contains no domain rules of
    its own: fetching and normalization stay inside the provider,
    deduplication and persistence stay inside the DataManager. Its only
    responsibility is deciding which time range still needs to be fetched.

    Args:
        provider: Source of historical OHLCV bars.
        data_manager: Destination used to store and query persisted bars.
    """

    def __init__(self, provider: BaseProvider, data_manager: DataManager) -> None:
        self._provider = provider
        self._data_manager = data_manager

    def sync_bars(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> Sequence[MarketBar]:
        """Fetch and persist bars missing between the latest stored bar and ``end``.

        If bars are already stored for the instrument and timeframe, the
        fetch resumes from the latest stored timestamp instead of ``start``,
        avoiding a full re-download on every call. Overlapping the last
        stored bar is intentional and harmless: the DataManager resolves it
        by keeping the freshest bar for that timestamp.

        Args:
            instrument: Instrument to synchronize.
            timeframe: Aggregation interval to synchronize.
            start: Inclusive, timezone-aware UTC start of the requested range.
            end: Exclusive, timezone-aware UTC end of the requested range.

        Returns:
            The bars fetched from the provider during this call. Empty when
            the stored data already covers the requested range.

        Raises:
            ValueError: If ``start`` is not strictly before ``end``.
        """
        if start >= end:
            raise ValueError("start must be strictly before end")

        latest_stored = self._data_manager.latest_timestamp(instrument, timeframe)
        fetch_start = start
        if latest_stored is not None and latest_stored > fetch_start:
            fetch_start = latest_stored

        if fetch_start >= end:
            return []

        bars = self._provider.get_historical_bars(
            instrument, timeframe, fetch_start, end
        )
        if bars:
            self._data_manager.store_bars(instrument, timeframe, bars)
        return bars


__all__ = ["MarketDataSyncService"]
