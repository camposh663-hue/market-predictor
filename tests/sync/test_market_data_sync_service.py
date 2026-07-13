"""Tests for MarketDataSyncService, using a fake provider and in-memory storage."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import List, Sequence, Tuple

from src.database import DataManager, InMemoryRepository
from src.domain import AssetClass, Instrument, MarketBar, TimeFrame
from src.providers import BaseProvider
from src.sync import MarketDataSyncService


def _bar(minute: int, close: float = 100.0) -> MarketBar:
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute)
    return MarketBar(
        timestamp=timestamp, open=close, high=close, low=close, close=close, volume=1.0
    )


class FakeProvider(BaseProvider):
    """Test double that serves preconfigured bars and records requested ranges."""

    def __init__(self, bars: Sequence[MarketBar]) -> None:
        self._bars = list(bars)
        self.requested_ranges: List[Tuple[datetime, datetime]] = []

    @property
    def provider_id(self) -> str:
        return "fake"

    def get_historical_bars(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> Sequence[MarketBar]:
        self.requested_ranges.append((start, end))
        return [bar for bar in self._bars if start <= bar.timestamp < end]


class MarketDataSyncServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instrument = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
        self.timeframe = TimeFrame.ONE_MINUTE
        self.data_manager = DataManager(InMemoryRepository())

    def test_sync_fetches_and_stores_bars_when_nothing_stored(self) -> None:
        bars = [_bar(0), _bar(1), _bar(2)]
        provider = FakeProvider(bars)
        service = MarketDataSyncService(provider, self.data_manager)

        result = service.sync_bars(
            self.instrument,
            self.timeframe,
            start=bars[0].timestamp,
            end=bars[-1].timestamp + timedelta(minutes=1),
        )

        self.assertEqual(list(result), bars)
        stored = self.data_manager.get_bars(
            self.instrument,
            self.timeframe,
            start=bars[0].timestamp,
            end=bars[-1].timestamp + timedelta(minutes=1),
        )
        self.assertEqual(list(stored), bars)
        self.assertEqual(
            provider.requested_ranges,
            [(bars[0].timestamp, bars[-1].timestamp + timedelta(minutes=1))],
        )

    def test_sync_resumes_from_latest_stored_timestamp(self) -> None:
        self.data_manager.store_bars(self.instrument, self.timeframe, [_bar(0), _bar(1)])
        provider = FakeProvider([_bar(1), _bar(2), _bar(3)])
        service = MarketDataSyncService(provider, self.data_manager)

        service.sync_bars(
            self.instrument,
            self.timeframe,
            start=_bar(0).timestamp,
            end=_bar(3).timestamp + timedelta(minutes=1),
        )

        self.assertEqual(
            provider.requested_ranges,
            [(_bar(1).timestamp, _bar(3).timestamp + timedelta(minutes=1))],
        )

    def test_sync_merges_fetched_bars_with_previously_stored_ones(self) -> None:
        self.data_manager.store_bars(self.instrument, self.timeframe, [_bar(0), _bar(1)])
        provider = FakeProvider([_bar(1), _bar(2), _bar(3)])
        service = MarketDataSyncService(provider, self.data_manager)

        service.sync_bars(
            self.instrument,
            self.timeframe,
            start=_bar(0).timestamp,
            end=_bar(3).timestamp + timedelta(minutes=1),
        )

        stored = self.data_manager.get_bars(
            self.instrument,
            self.timeframe,
            start=_bar(0).timestamp,
            end=_bar(3).timestamp + timedelta(minutes=1),
        )
        self.assertEqual(list(stored), [_bar(0), _bar(1), _bar(2), _bar(3)])

    def test_sync_skips_provider_call_when_already_up_to_date(self) -> None:
        self.data_manager.store_bars(self.instrument, self.timeframe, [_bar(0), _bar(1)])
        provider = FakeProvider([])
        service = MarketDataSyncService(provider, self.data_manager)

        result = service.sync_bars(
            self.instrument,
            self.timeframe,
            start=_bar(0).timestamp,
            end=_bar(1).timestamp,
        )

        self.assertEqual(result, [])
        self.assertEqual(provider.requested_ranges, [])

    def test_sync_does_not_store_when_provider_returns_no_bars(self) -> None:
        provider = FakeProvider([])
        service = MarketDataSyncService(provider, self.data_manager)

        result = service.sync_bars(
            self.instrument,
            self.timeframe,
            start=_bar(0).timestamp,
            end=_bar(1).timestamp,
        )

        self.assertEqual(result, [])
        self.assertIsNone(
            self.data_manager.latest_timestamp(self.instrument, self.timeframe)
        )

    def test_sync_rejects_non_increasing_range(self) -> None:
        provider = FakeProvider([])
        service = MarketDataSyncService(provider, self.data_manager)
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)

        with self.assertRaises(ValueError):
            service.sync_bars(self.instrument, self.timeframe, start=now, end=now)


if __name__ == "__main__":
    unittest.main()
