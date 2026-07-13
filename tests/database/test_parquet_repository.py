"""Tests for ParquetRepository, backed by a temporary directory."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.database import DataManager, ParquetRepository
from src.domain import AssetClass, Instrument, MarketBar, TimeFrame


def _bar(minute: int, close: float = 100.0, volume: float | None = 1.0) -> MarketBar:
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute)
    return MarketBar(
        timestamp=timestamp,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
    )


class ParquetRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base_path = Path(self._tmp.name)
        self.repository = ParquetRepository(base_path=self.base_path)
        self.instrument = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
        self.timeframe = TimeFrame.ONE_MINUTE

    def test_read_bars_returns_empty_when_no_file_exists(self) -> None:
        result = self.repository.read_bars(
            self.instrument,
            self.timeframe,
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

        self.assertEqual(result, [])

    def test_latest_timestamp_returns_none_when_no_file_exists(self) -> None:
        self.assertIsNone(
            self.repository.latest_timestamp(self.instrument, self.timeframe)
        )

    def test_write_then_read_round_trip(self) -> None:
        bars = [_bar(0), _bar(1, volume=None), _bar(2)]

        self.repository.write_bars(self.instrument, self.timeframe, bars)
        result = self.repository.read_bars(
            self.instrument,
            self.timeframe,
            start=bars[0].timestamp,
            end=bars[-1].timestamp + timedelta(minutes=1),
        )

        self.assertEqual(list(result), bars)

    def test_write_bars_creates_parent_folders(self) -> None:
        self.repository.write_bars(self.instrument, self.timeframe, [_bar(0)])

        expected_path = self.base_path / "crypto" / "BTCUSDT" / "1m.parquet"
        self.assertTrue(expected_path.exists())

    def test_symbol_with_slash_is_normalized_to_a_single_folder(self) -> None:
        self.repository.write_bars(self.instrument, self.timeframe, [_bar(0)])

        self.assertFalse((self.base_path / "crypto" / "BTC").exists())
        self.assertTrue((self.base_path / "crypto" / "BTCUSDT").is_dir())

    def test_second_write_merges_and_keeps_file_sorted(self) -> None:
        self.repository.write_bars(self.instrument, self.timeframe, [_bar(0), _bar(2)])
        self.repository.write_bars(self.instrument, self.timeframe, [_bar(1)])

        result = self.repository.read_bars(
            self.instrument,
            self.timeframe,
            start=_bar(0).timestamp,
            end=_bar(2).timestamp + timedelta(minutes=1),
        )

        self.assertEqual([bar.timestamp for bar in result], [
            _bar(0).timestamp,
            _bar(1).timestamp,
            _bar(2).timestamp,
        ])

    def test_second_write_overwrites_stale_bar_with_same_timestamp(self) -> None:
        stale = _bar(0, close=100.0)
        fresh = _bar(0, close=200.0)

        self.repository.write_bars(self.instrument, self.timeframe, [stale])
        self.repository.write_bars(self.instrument, self.timeframe, [fresh])

        result = self.repository.read_bars(
            self.instrument,
            self.timeframe,
            start=stale.timestamp,
            end=stale.timestamp + timedelta(minutes=1),
        )

        self.assertEqual(list(result), [fresh])

    def test_latest_timestamp_returns_newest_bar_after_merge(self) -> None:
        self.repository.write_bars(self.instrument, self.timeframe, [_bar(0), _bar(5)])
        self.repository.write_bars(self.instrument, self.timeframe, [_bar(2)])

        self.assertEqual(
            self.repository.latest_timestamp(self.instrument, self.timeframe),
            _bar(5).timestamp,
        )

    def test_series_are_isolated_by_timeframe_and_symbol(self) -> None:
        other_instrument = Instrument(symbol="ETH/USDT", asset_class=AssetClass.CRYPTO)

        self.repository.write_bars(self.instrument, TimeFrame.ONE_MINUTE, [_bar(0)])

        self.assertIsNone(
            self.repository.latest_timestamp(self.instrument, TimeFrame.ONE_HOUR)
        )
        self.assertIsNone(
            self.repository.latest_timestamp(other_instrument, TimeFrame.ONE_MINUTE)
        )

    def test_data_manager_works_unchanged_with_parquet_repository(self) -> None:
        manager = DataManager(self.repository)
        bars = [_bar(0), _bar(1)]

        manager.store_bars(self.instrument, self.timeframe, bars)
        result = manager.get_bars(
            self.instrument,
            self.timeframe,
            start=bars[0].timestamp,
            end=bars[-1].timestamp + timedelta(minutes=1),
        )

        self.assertEqual(list(result), bars)


if __name__ == "__main__":
    unittest.main()
