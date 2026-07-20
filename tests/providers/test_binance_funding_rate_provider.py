"""Tests for BinanceFundingRateProvider, using a fake HTTP session."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from typing import Any, List

import requests

from src.domain import AssetClass, Instrument
from src.providers.binance_funding_rate_provider import (
    BinanceFundingRateAPIError,
    BinanceFundingRateProvider,
    BinanceFundingRateRequestError,
    BinanceFundingRateResponseError,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers: dict = {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    def __init__(self, pages: List[Any]) -> None:
        self._pages = list(pages)
        self.requests: List[dict] = []

    def get(self, url: str, params: dict, timeout: float) -> _FakeResponse:
        self.requests.append(params)
        payload = self._pages.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return _FakeResponse(200, payload)


class _RaisingSession:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def get(self, url: str, params: dict, timeout: float) -> _FakeResponse:
        raise self._exc


class _SmallPageProvider(BinanceFundingRateProvider):
    """Overrides the page size so pagination can be tested without 1000 fakes."""

    _MAX_RECORDS_PER_REQUEST = 2


def _record(funding_time_ms: int, rate: str = "0.0001") -> dict:
    return {"symbol": "BTCUSDT", "fundingRate": rate, "fundingTime": funding_time_ms, "markPrice": "50000"}


INSTRUMENT = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2024, 1, 2, tzinfo=timezone.utc)


class BinanceFundingRateProviderTests(unittest.TestCase):
    def test_provider_id(self) -> None:
        provider = BinanceFundingRateProvider(session=_FakeSession([[]]))
        self.assertEqual(provider.provider_id, "binance_futures")

    def test_parses_valid_records_into_domain_objects(self) -> None:
        session = _FakeSession([[_record(1_704_067_200_000, "0.00013")]])
        provider = BinanceFundingRateProvider(session=session)

        result = provider.get_historical_funding_rates(INSTRUMENT, START, END)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].rate, 0.00013)
        self.assertEqual(result[0].timestamp, datetime(2024, 1, 1, tzinfo=timezone.utc))

    def test_empty_page_returns_no_records(self) -> None:
        provider = BinanceFundingRateProvider(session=_FakeSession([[]]))

        result = provider.get_historical_funding_rates(INSTRUMENT, START, END)

        self.assertEqual(result, [])

    def test_pagination_advances_cursor_past_last_funding_time(self) -> None:
        page1 = [_record(1_704_067_200_000), _record(1_704_096_000_000)]  # full page (size 2)
        page2 = [_record(1_704_124_800_000)]  # partial page, stops pagination
        session = _FakeSession([page1, page2])
        provider = _SmallPageProvider(session=session)

        result = provider.get_historical_funding_rates(INSTRUMENT, START, END)

        self.assertEqual(len(result), 3)
        self.assertEqual(len(session.requests), 2)
        self.assertEqual(session.requests[1]["startTime"], 1_704_096_000_001)

    def test_raises_on_record_outside_requested_range(self) -> None:
        out_of_range_ms = int(END.timestamp() * 1000) + 1_000_000
        session = _FakeSession([[_record(out_of_range_ms)]])
        provider = BinanceFundingRateProvider(session=session)

        with self.assertRaises(BinanceFundingRateResponseError):
            provider.get_historical_funding_rates(INSTRUMENT, START, END)

    def test_raises_on_unordered_records(self) -> None:
        session = _FakeSession([[_record(1_704_096_000_000), _record(1_704_067_200_000)]])
        provider = BinanceFundingRateProvider(session=session)

        with self.assertRaises(BinanceFundingRateResponseError):
            provider.get_historical_funding_rates(INSTRUMENT, START, END)

    def test_raises_on_non_list_payload(self) -> None:
        session = _FakeSession([{"code": -1, "msg": "unexpected"}])
        provider = BinanceFundingRateProvider(session=session)

        with self.assertRaises(BinanceFundingRateResponseError):
            provider.get_historical_funding_rates(INSTRUMENT, START, END)

    def test_raises_api_error_on_http_failure(self) -> None:
        class _ErrorSession:
            def get(self, url: str, params: dict, timeout: float) -> _FakeResponse:
                return _FakeResponse(500, {"code": -1, "msg": "server error"})

        provider = BinanceFundingRateProvider(session=_ErrorSession())

        with self.assertRaises(BinanceFundingRateAPIError):
            provider.get_historical_funding_rates(INSTRUMENT, START, END)

    def test_raises_request_error_on_timeout(self) -> None:
        provider = BinanceFundingRateProvider(session=_RaisingSession(requests.Timeout()))

        with self.assertRaises(BinanceFundingRateRequestError):
            provider.get_historical_funding_rates(INSTRUMENT, START, END)

    def test_raises_on_naive_datetime(self) -> None:
        provider = BinanceFundingRateProvider(session=_FakeSession([[]]))

        with self.assertRaises(ValueError):
            provider.get_historical_funding_rates(INSTRUMENT, datetime(2024, 1, 1), END)

    def test_raises_when_start_not_before_end(self) -> None:
        provider = BinanceFundingRateProvider(session=_FakeSession([[]]))

        with self.assertRaises(ValueError):
            provider.get_historical_funding_rates(INSTRUMENT, END, START)


if __name__ == "__main__":
    unittest.main()
