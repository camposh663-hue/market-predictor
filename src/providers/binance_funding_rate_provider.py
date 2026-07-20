"""Binance USDS-margined perpetual-futures funding-rate provider."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Sequence, Tuple

import requests

from src.domain import FundingRate, Instrument

from .funding_rate_provider import FundingRateProvider


class BinanceFundingRateProviderError(RuntimeError):
    """Base exception raised by the Binance funding-rate provider."""


class BinanceFundingRateRequestError(BinanceFundingRateProviderError):
    """Raise when a request cannot be completed at the transport layer."""


class BinanceFundingRateAPIError(BinanceFundingRateProviderError):
    """Raise when Binance returns an unsuccessful HTTP response."""


class BinanceFundingRateResponseError(BinanceFundingRateProviderError):
    """Raise when Binance returns data outside the documented schema."""


class BinanceFundingRateProvider(FundingRateProvider):
    """Download normalized historical funding-rate settlements from Binance Futures.

    Mirrors ``BinanceProvider``'s HTTP, pagination, and error-handling shape
    (see ``src/providers/binance_provider.py``), against the funding-rate
    endpoint of Binance's USDS-margined futures API instead of spot klines.

    Args:
        base_url: Base URL of the Binance USDS-margined futures public API.
        timeout_seconds: Maximum number of seconds to wait for each HTTP
            request. The value must be greater than zero.
        session: Optional requests session to use for HTTP calls. Supplying a
            session supports connection reuse and deterministic tests.
    """

    _DEFAULT_BASE_URL = "https://fapi.binance.com"
    _FUNDING_RATE_PATH = "/fapi/v1/fundingRate"
    _MAX_RECORDS_PER_REQUEST = 1_000
    _DEFAULT_TIMEOUT_SECONDS = 10.0
    _EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Initialize a Binance Futures funding-rate provider.

        Args:
            base_url: Base URL of a Binance-compatible futures public API.
            timeout_seconds: Per-request timeout in seconds; must be positive.
            session: Optional requests session used for outbound HTTP requests.

        Raises:
            TypeError: If ``base_url`` is not a string.
            ValueError: If the base URL is empty or the timeout is invalid.
        """
        if not isinstance(base_url, str):
            raise TypeError("base_url must be a string.")

        normalized_base_url = base_url.strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("base_url must not be empty.")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError(
                "timeout_seconds must be a finite number greater than zero."
            )

        self._base_url = normalized_base_url
        self._timeout_seconds = timeout_seconds
        self._session = session if session is not None else requests.Session()

    @property
    def provider_id(self) -> str:
        """Return the stable identifier for this provider.

        Returns:
            The provider identifier ``"binance_futures"``.
        """
        return "binance_futures"

    def get_historical_funding_rates(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
    ) -> Sequence[FundingRate]:
        """Retrieve normalized funding-rate settlements from Binance Futures.

        ``start`` is inclusive and ``end`` is exclusive, matching the
        ``FundingRateProvider`` contract. All input datetimes must be
        timezone-aware; they are normalized to UTC for the Binance request.

        Args:
            instrument: Instrument to download funding-rate history for.
            start: Inclusive, timezone-aware range start.
            end: Exclusive, timezone-aware range end.

        Returns:
            Chronologically ordered normalized funding-rate settlements.

        Raises:
            ValueError: If the instrument or time range is invalid.
            BinanceFundingRateRequestError: If the HTTP request times out or
                cannot connect.
            BinanceFundingRateAPIError: If Binance returns an unsuccessful
                HTTP response.
            BinanceFundingRateResponseError: If Binance returns malformed or
                inconsistent funding-rate data.
        """
        self._validate_time_range(start, end)

        symbol = self._to_binance_symbol(instrument)
        start_milliseconds = self._to_start_milliseconds(start)
        end_milliseconds = self._to_end_milliseconds(end)

        if start_milliseconds > end_milliseconds:
            return []

        rates: List[FundingRate] = []
        current_start_milliseconds = start_milliseconds
        previous_funding_time: Optional[int] = None

        while current_start_milliseconds <= end_milliseconds:
            records = self._fetch_funding_rates(
                symbol=symbol,
                start_milliseconds=current_start_milliseconds,
                end_milliseconds=end_milliseconds,
            )
            if not records:
                break
            if len(records) > self._MAX_RECORDS_PER_REQUEST:
                raise BinanceFundingRateResponseError(
                    "Binance returned more funding-rate records than the "
                    f"documented limit of {self._MAX_RECORDS_PER_REQUEST}."
                )

            page_last_funding_time: Optional[int] = None
            for record in records:
                funding_time, funding_rate = self._to_funding_rate(record)

                if (
                    funding_time < current_start_milliseconds
                    or funding_time > end_milliseconds
                ):
                    raise BinanceFundingRateResponseError(
                        "Binance returned a funding-rate record outside the "
                        "requested time range."
                    )
                if (
                    previous_funding_time is not None
                    and funding_time <= previous_funding_time
                ):
                    raise BinanceFundingRateResponseError(
                        "Binance returned duplicate or unordered funding times."
                    )

                rates.append(funding_rate)
                previous_funding_time = funding_time
                page_last_funding_time = funding_time

            if page_last_funding_time is None:
                raise BinanceFundingRateResponseError(
                    "Binance returned an empty funding-rate page."
                )
            if (
                len(records) < self._MAX_RECORDS_PER_REQUEST
                or page_last_funding_time >= end_milliseconds
            ):
                break

            next_start_milliseconds = page_last_funding_time + 1
            if next_start_milliseconds <= current_start_milliseconds:
                raise BinanceFundingRateResponseError(
                    "Binance pagination did not advance the funding-rate time range."
                )
            current_start_milliseconds = next_start_milliseconds

        return rates

    def _fetch_funding_rates(
        self,
        *,
        symbol: str,
        start_milliseconds: int,
        end_milliseconds: int,
    ) -> List[Any]:
        """Request one page of raw Binance funding-rate records.

        Args:
            symbol: Binance-formatted futures symbol.
            start_milliseconds: Inclusive request start in Unix milliseconds.
            end_milliseconds: Inclusive request end in Unix milliseconds.

        Returns:
            One raw Binance funding-rate page. The data remains private to
            this class.

        Raises:
            BinanceFundingRateRequestError: If the HTTP request fails before
                a response.
            BinanceFundingRateAPIError: If Binance returns a non-success HTTP
                response.
            BinanceFundingRateResponseError: If the response body is not a
                JSON list.
        """
        params = {
            "symbol": symbol,
            "startTime": start_milliseconds,
            "endTime": end_milliseconds,
            "limit": self._MAX_RECORDS_PER_REQUEST,
        }

        try:
            response = self._session.get(
                f"{self._base_url}{self._FUNDING_RATE_PATH}",
                params=params,
                timeout=self._timeout_seconds,
            )
        except requests.Timeout as exc:
            raise BinanceFundingRateRequestError(
                "Timed out while requesting funding-rate history from Binance."
            ) from exc
        except requests.RequestException as exc:
            raise BinanceFundingRateRequestError(
                "Could not request funding-rate history from Binance."
            ) from exc

        self._raise_for_http_error(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise BinanceFundingRateResponseError(
                "Binance returned a response that is not valid JSON."
            ) from exc

        if not isinstance(payload, list):
            raise BinanceFundingRateResponseError(
                "Binance returned an unexpected funding-rate response payload."
            )

        return payload

    def _raise_for_http_error(self, response: requests.Response) -> None:
        """Translate unsuccessful Binance HTTP responses into provider errors.

        Args:
            response: HTTP response returned by the Binance endpoint.

        Raises:
            BinanceFundingRateAPIError: If the response status is not
                successful.
        """
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise BinanceFundingRateAPIError(
                f"Binance returned HTTP {response.status_code} for the "
                "funding-rate request."
            ) from exc

    @staticmethod
    def _to_binance_symbol(instrument: Instrument) -> str:
        """Translate a domain instrument into Binance's uppercase pair symbol.

        Args:
            instrument: Domain instrument to translate.

        Returns:
            Binance futures symbol, such as ``"BTCUSDT"``.

        Raises:
            ValueError: If the instrument symbol cannot be normalized safely.
        """
        normalized_symbol = instrument.symbol.strip().upper()
        for separator in ("/", "-", "_"):
            normalized_symbol = normalized_symbol.replace(separator, "")

        if not normalized_symbol or not normalized_symbol.isalnum():
            raise ValueError(
                "Instrument symbol must be a non-empty alphanumeric pair or use "
                "the '/', '-', or '_' separators."
            )
        return normalized_symbol

    @classmethod
    def _to_funding_rate(cls, record: Any) -> Tuple[int, FundingRate]:
        """Convert one raw Binance funding-rate record into a domain object.

        Args:
            record: Raw Binance funding-rate response entry.

        Returns:
            The raw funding timestamp in milliseconds and its normalized
            ``FundingRate``.

        Raises:
            BinanceFundingRateResponseError: If the record does not match
                Binance's schema or contains invalid values.
        """
        if not isinstance(record, dict):
            raise BinanceFundingRateResponseError(
                "Binance returned a funding-rate record that is not an object."
            )

        raw_funding_time = record.get("fundingTime")
        raw_funding_rate = record.get("fundingRate")
        if isinstance(raw_funding_time, bool) or not isinstance(raw_funding_time, int):
            raise BinanceFundingRateResponseError(
                "Binance returned a funding-rate record with a non-integer "
                "fundingTime."
            )

        try:
            funding_time = raw_funding_time
            timestamp = cls._EPOCH + timedelta(milliseconds=funding_time)
            rate = float(raw_funding_rate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BinanceFundingRateResponseError(
                "Binance returned a funding-rate record with an invalid rate "
                "or timestamp."
            ) from exc

        if not math.isfinite(rate):
            raise BinanceFundingRateResponseError(
                "Binance returned a non-finite funding rate."
            )

        return funding_time, FundingRate(timestamp=timestamp, rate=rate)

    @staticmethod
    def _validate_time_range(start: datetime, end: datetime) -> None:
        """Validate the timestamp requirements of the FundingRateProvider contract.

        Args:
            start: Inclusive range start.
            end: Exclusive range end.

        Raises:
            TypeError: If either boundary is not a datetime.
            ValueError: If a boundary is naive or the range is not increasing.
        """
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise TypeError("start and end must be datetime instances.")
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("start must be a timezone-aware datetime.")
        if end.tzinfo is None or end.utcoffset() is None:
            raise ValueError("end must be a timezone-aware datetime.")
        if start >= end:
            raise ValueError("start must be earlier than end.")

    @classmethod
    def _to_start_milliseconds(cls, value: datetime) -> int:
        """Convert an inclusive datetime to the first eligible UTC millisecond."""
        return cls._ceil_division(cls._to_epoch_microseconds(value), 1_000)

    @classmethod
    def _to_end_milliseconds(cls, value: datetime) -> int:
        """Convert an exclusive datetime to the last eligible UTC millisecond."""
        return cls._ceil_division(cls._to_epoch_microseconds(value), 1_000) - 1

    @classmethod
    def _to_epoch_microseconds(cls, value: datetime) -> int:
        """Convert a timezone-aware datetime into UTC Unix microseconds."""
        utc_value = value.astimezone(timezone.utc)
        delta = utc_value - cls._EPOCH
        return (
            (delta.days * 86_400 + delta.seconds) * 1_000_000
            + delta.microseconds
        )

    @staticmethod
    def _ceil_division(value: int, divisor: int) -> int:
        """Return the mathematical ceiling of an integer division."""
        return -(-value // divisor)


__all__ = [
    "BinanceFundingRateAPIError",
    "BinanceFundingRateProvider",
    "BinanceFundingRateProviderError",
    "BinanceFundingRateRequestError",
    "BinanceFundingRateResponseError",
]
