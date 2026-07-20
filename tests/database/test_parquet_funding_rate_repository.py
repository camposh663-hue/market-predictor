"""Tests for ParquetFundingRateRepository, backed by a temporary directory."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.database import ParquetFundingRateRepository
from src.domain import AssetClass, FundingRate, Instrument


def _rate(hour: int, rate: float = 0.0001) -> FundingRate:
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=8 * hour)
    return FundingRate(timestamp=timestamp, rate=rate)


class ParquetFundingRateRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base_path = Path(self._tmp.name)
        self.repository = ParquetFundingRateRepository(base_path=self.base_path)
        self.instrument = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)

    def test_read_funding_rates_returns_empty_when_no_file_exists(self) -> None:
        result = self.repository.read_funding_rates(
            self.instrument,
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

        self.assertEqual(result, [])

    def test_latest_timestamp_returns_none_when_no_file_exists(self) -> None:
        self.assertIsNone(self.repository.latest_timestamp(self.instrument))

    def test_write_then_read_round_trip(self) -> None:
        rates = [_rate(0), _rate(1), _rate(2)]

        self.repository.write_funding_rates(self.instrument, rates)
        result = self.repository.read_funding_rates(
            self.instrument,
            start=rates[0].timestamp,
            end=rates[-1].timestamp + timedelta(hours=1),
        )

        self.assertEqual(list(result), rates)

    def test_write_funding_rates_creates_parent_folders(self) -> None:
        self.repository.write_funding_rates(self.instrument, [_rate(0)])

        expected_path = self.base_path / "crypto" / "BTCUSDT" / "funding_rate.parquet"
        self.assertTrue(expected_path.exists())

    def test_symbol_with_slash_is_normalized_to_a_single_folder(self) -> None:
        self.repository.write_funding_rates(self.instrument, [_rate(0)])

        self.assertFalse((self.base_path / "crypto" / "BTC").exists())
        self.assertTrue((self.base_path / "crypto" / "BTCUSDT").is_dir())

    def test_second_write_merges_and_keeps_file_sorted(self) -> None:
        self.repository.write_funding_rates(self.instrument, [_rate(0), _rate(2)])
        self.repository.write_funding_rates(self.instrument, [_rate(1)])

        result = self.repository.read_funding_rates(
            self.instrument,
            start=_rate(0).timestamp,
            end=_rate(2).timestamp + timedelta(hours=1),
        )

        self.assertEqual(
            [rate.timestamp for rate in result],
            [_rate(0).timestamp, _rate(1).timestamp, _rate(2).timestamp],
        )

    def test_second_write_overwrites_stale_rate_with_same_timestamp(self) -> None:
        stale = _rate(0, rate=0.0001)
        fresh = _rate(0, rate=0.0005)

        self.repository.write_funding_rates(self.instrument, [stale])
        self.repository.write_funding_rates(self.instrument, [fresh])

        result = self.repository.read_funding_rates(
            self.instrument,
            start=stale.timestamp,
            end=stale.timestamp + timedelta(hours=1),
        )

        self.assertEqual(list(result), [fresh])

    def test_latest_timestamp_returns_newest_rate_after_merge(self) -> None:
        self.repository.write_funding_rates(self.instrument, [_rate(0), _rate(5)])
        self.repository.write_funding_rates(self.instrument, [_rate(2)])

        self.assertEqual(self.repository.latest_timestamp(self.instrument), _rate(5).timestamp)

    def test_instruments_are_isolated_by_symbol(self) -> None:
        other_instrument = Instrument(symbol="ETH/USDT", asset_class=AssetClass.CRYPTO)

        self.repository.write_funding_rates(self.instrument, [_rate(0)])

        self.assertIsNone(self.repository.latest_timestamp(other_instrument))


if __name__ == "__main__":
    unittest.main()
