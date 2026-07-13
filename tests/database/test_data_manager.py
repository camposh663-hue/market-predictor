"""Tests for DataManager, backed by an in-memory repository."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.database import DataManager, InMemoryRepository
from src.domain import AssetClass, Instrument, MarketBar, TimeFrame


def _bar(minute: int, close: float = 100.0) -> MarketBar:
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute)
    return MarketBar(
        timestamp=timestamp,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


class DataManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instrument = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
        self.timeframe = TimeFrame.ONE_MINUTE
        self.manager = DataManager(InMemoryRepository())

    def test_store_and_get_bars_round_trip(self) -> None:
        bars = [_bar(0), _bar(1), _bar(2)]

        self.manager.store_bars(self.instrument, self.timeframe, bars)
        result = self.manager.get_bars(
            self.instrument,
            self.timeframe,
            start=bars[0].timestamp,
            end=bars[-1].timestamp + timedelta(minutes=1),
        )

        self.assertEqual(list(result), bars)

    def test_store_bars_deduplicates_by_timestamp_keeping_last(self) -> None:
        stale = _bar(0, close=100.0)
        fresh = _bar(0, close=200.0)

        self.manager.store_bars(self.instrument, self.timeframe, [stale, fresh])
        result = self.manager.get_bars(
            self.instrument,
            self.timeframe,
            start=stale.timestamp,
            end=stale.timestamp + timedelta(minutes=1),
        )

        self.assertEqual(list(result), [fresh])

    def test_store_bars_rejects_empty_sequence(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.store_bars(self.instrument, self.timeframe, [])

    def test_get_bars_rejects_non_increasing_range(self) -> None:
        now = datetime(2024, 1, 1, tzinfo=timezone.utc)

        with self.assertRaises(ValueError):
            self.manager.get_bars(self.instrument, self.timeframe, start=now, end=now)

    def test_latest_timestamp_is_none_when_empty(self) -> None:
        self.assertIsNone(
            self.manager.latest_timestamp(self.instrument, self.timeframe)
        )

    def test_latest_timestamp_returns_newest_stored_bar(self) -> None:
        bars = [_bar(0), _bar(5), _bar(2)]

        self.manager.store_bars(self.instrument, self.timeframe, bars)

        self.assertEqual(
            self.manager.latest_timestamp(self.instrument, self.timeframe),
            _bar(5).timestamp,
        )

    def test_series_are_isolated_by_timeframe(self) -> None:
        self.manager.store_bars(self.instrument, TimeFrame.ONE_MINUTE, [_bar(0)])

        self.assertIsNone(
            self.manager.latest_timestamp(self.instrument, TimeFrame.ONE_HOUR)
        )


if __name__ == "__main__":
    unittest.main()
