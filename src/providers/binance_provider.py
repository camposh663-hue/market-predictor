"""Binance Spot provider for normalized historical OHLCV market data."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import requests

from src.domain import AssetClass, Instrument, MarketBar, TimeFrame

from .base_provider import BaseProvider


class BinanceProviderError(RuntimeError):
    """Base exception raised by the Binance market-data provider."""


class BinanceRequestError(BinanceProviderError):
    """Raise when a request cannot be completed at the transport layer."""


class BinanceAPIError(BinanceProviderError):
    """Raise when Binance returns an unsuccessful HTTP response."""


class BinanceResponseError(BinanceProviderError):
    """Raise when Binance returns data outside the documented kline schema."""


class BinanceProvider(BaseProvider):
    """Download normalized historical OHLCV data from Binance Spot.

    The provider owns all Binance-specific concerns, including symbol and
    interval conversion, HTTP communication, pagination, and response parsing.
    It only exposes domain objects to the rest of the application.

    Args:
        base_url: Base URL of the Binance public market-data API. It can be
            overridden for a compatible test or deployment endpoint.
        timeout_seconds: Maximum number of seconds to wait for each HTTP
            request. The value must be greater than zero.
        session: Optional requests session to use for HTTP calls. Supplying a
            session supports connection reuse and deterministic tests.
    """

    _DEFAULT_BASE_URL = "https://data-api.binance.vision"
    _KLINES_PATH = "/api/v3/klines"
    _MAX_KLINES_PER_REQUEST = 1_000
    _DEFAULT_TIMEOUT_SECONDS = 10.0
    _EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
    _TIMEFRAME_TO_INTERVAL: Mapping[TimeFrame, str] = {
        TimeFrame.ONE_MINUTE: "1m",
        TimeFrame.FIVE_MINUTES: "5m",
        TimeFrame.FIFTEEN_MINUTES: "15m",
        TimeFrame.ONE_HOUR: "1h",
        TimeFrame.FOUR_HOURS: "4h",
        TimeFrame.ONE_DAY: "1d",
    }

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Initialize a Binance Spot historical-data provider.

        Args:
            base_url: Base URL of a Binance-compatible public market-data API.
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
        """Return the stable identifier for the Binance provider.

        Returns:
            The provider identifier ``"binance"``.
        """
        return "binance"

    def get_historical_bars(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> Sequence[MarketBar]:
        """Retrieve normalized OHLCV bars from Binance Spot.

        The requested interval follows the BaseProvider contract: ``start`` is
        inclusive and ``end`` is exclusive. All input datetimes must be
        timezone-aware; they are normalized to UTC for the Binance request.
        The range is applied to each bar's opening time, so callers requiring
        only closed bars must choose an appropriate completed interval boundary.

        Args:
            instrument: Cryptocurrency instrument to download from Binance.
            timeframe: Domain aggregation interval to translate for Binance.
            start: Inclusive, timezone-aware range start.
            end: Exclusive, timezone-aware range end.

        Returns:
            Chronologically ordered normalized OHLCV bars.

        Raises:
            ValueError: If the instrument or time range is invalid for Binance.
            BinanceRequestError: If the HTTP request times out or cannot connect.
            BinanceAPIError: If Binance returns an unsuccessful HTTP response.
            BinanceResponseError: If Binance returns malformed or inconsistent
                kline data.
        """
        self._validate_time_range(start, end)

        symbol = self._to_binance_symbol(instrument)
        interval = self._to_binance_interval(timeframe)
        start_milliseconds = self._to_start_milliseconds(start)
        end_milliseconds = self._to_end_milliseconds(end)

        if start_milliseconds > end_milliseconds:
            return []

        bars: List[MarketBar] = []
        current_start_milliseconds = start_milliseconds
        previous_open_time: Optional[int] = None

        while current_start_milliseconds <= end_milliseconds:
            klines = self._fetch_klines(
                symbol=symbol,
                interval=interval,
                start_milliseconds=current_start_milliseconds,
                end_milliseconds=end_milliseconds,
            )
            if not klines:
                break
            if len(klines) > self._MAX_KLINES_PER_REQUEST:
                raise BinanceResponseError(
                    "Binance returned more klines than the documented limit of "
                    f"{self._MAX_KLINES_PER_REQUEST}."
                )

            page_last_open_time: Optional[int] = None
            for kline in klines:
                open_time, market_bar = self._to_market_bar(kline)

                if (
                    open_time < current_start_milliseconds
                    or open_time > end_milliseconds
                ):
                    raise BinanceResponseError(
                        "Binance returned a kline outside the requested time range."
                    )
                if previous_open_time is not None and open_time <= previous_open_time:
                    raise BinanceResponseError(
                        "Binance returned duplicate or unordered kline open times."
                    )

                bars.append(market_bar)
                previous_open_time = open_time
                page_last_open_time = open_time

            if page_last_open_time is None:
                raise BinanceResponseError("Binance returned an empty kline page.")
            if (
                len(klines) < self._MAX_KLINES_PER_REQUEST
                or page_last_open_time >= end_milliseconds
            ):
                break

            next_start_milliseconds = page_last_open_time + 1
            if next_start_milliseconds <= current_start_milliseconds:
                raise BinanceResponseError(
                    "Binance pagination did not advance the kline time range."
                )
            current_start_milliseconds = next_start_milliseconds

        return bars

    def _fetch_klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_milliseconds: int,
        end_milliseconds: int,
    ) -> List[List[Any]]:
        """Request one page of raw Binance klines and validate its top-level shape.

        Args:
            symbol: Binance-formatted symbol.
            interval: Binance-formatted kline interval.
            start_milliseconds: Inclusive request start in Unix milliseconds.
            end_milliseconds: Inclusive request end in Unix milliseconds.

        Returns:
            One raw Binance kline page. The data remains private to this class.

        Raises:
            BinanceRequestError: If the HTTP request fails before a response.
            BinanceAPIError: If Binance returns a non-success HTTP response.
            BinanceResponseError: If the response body is not a JSON kline list.
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_milliseconds,
            "endTime": end_milliseconds,
            "limit": self._MAX_KLINES_PER_REQUEST,
        }

        try:
            response = self._session.get(
                f"{self._base_url}{self._KLINES_PATH}",
                params=params,
                timeout=self._timeout_seconds,
            )
        except requests.Timeout as exc:
            raise BinanceRequestError(
                "Timed out while requesting historical klines from Binance."
            ) from exc
        except requests.RequestException as exc:
            raise BinanceRequestError(
                "Could not request historical klines from Binance."
            ) from exc

        self._raise_for_http_error(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise BinanceResponseError(
                "Binance returned a response that is not valid JSON."
            ) from exc

        if not isinstance(payload, list):
            error_details = self._format_payload_error(payload)
            raise BinanceResponseError(
                "Binance returned an unexpected kline response payload"
                f"{error_details}."
            )

        return payload

    def _raise_for_http_error(self, response: requests.Response) -> None:
        """Translate unsuccessful Binance HTTP responses into provider errors.

        Args:
            response: HTTP response returned by the Binance endpoint.

        Raises:
            BinanceAPIError: If the response status is not successful.
        """
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise BinanceAPIError(
                self._format_http_error(response)
            ) from exc

    def _format_http_error(self, response: requests.Response) -> str:
        """Build a useful error message from a Binance HTTP error response.

        Args:
            response: Unsuccessful HTTP response from Binance.

        Returns:
            An error message containing the HTTP status and available Binance
            error details, including ``Retry-After`` when supplied.
        """
        try:
            payload = response.json()
        except ValueError:
            payload = None

        error_details = self._format_payload_error(payload)
        retry_after = response.headers.get("Retry-After")
        retry_details = (
            f" Retry after {retry_after} seconds." if retry_after else ""
        )
        return (
            f"Binance returned HTTP {response.status_code}{error_details}."
            f"{retry_details}"
        )

    @staticmethod
    def _format_payload_error(payload: Any) -> str:
        """Extract Binance error details from a JSON payload when available.

        Args:
            payload: Parsed JSON response payload.

        Returns:
            A formatted Binance code and message, or an empty string when the
            payload does not contain the documented error fields.
        """
        if not isinstance(payload, dict):
            return ""

        error_code = payload.get("code")
        error_message = payload.get("msg")
        if error_code is None and error_message is None:
            return ""
        if error_code is None:
            return f": {error_message}"
        if error_message is None:
            return f" (Binance code {error_code})"
        return f" (Binance code {error_code}: {error_message})"

    def _to_binance_symbol(self, instrument: Instrument) -> str:
        """Translate a domain instrument into Binance's uppercase pair symbol.

        Args:
            instrument: Domain cryptocurrency instrument to translate.

        Returns:
            Binance Spot symbol, such as ``"BTCUSDT"``.

        Raises:
            ValueError: If the instrument cannot represent a Binance Spot pair.
        """
        if instrument.asset_class is not AssetClass.CRYPTO:
            raise ValueError("Binance Spot only supports crypto instruments.")

        if instrument.venue is not None and instrument.venue.strip().lower() not in {
            "binance",
            "binance_spot",
        }:
            raise ValueError(
                "A BinanceProvider can only download instruments for Binance Spot."
            )

        has_base_currency = instrument.base_currency is not None
        has_quote_currency = instrument.quote_currency is not None
        if has_base_currency != has_quote_currency:
            raise ValueError(
                "Instrument base_currency and quote_currency must be provided "
                "together for Binance symbol conversion."
            )

        if has_base_currency and has_quote_currency:
            base_currency = self._normalize_symbol_component(
                instrument.base_currency,
                "base_currency",
            )
            quote_currency = self._normalize_symbol_component(
                instrument.quote_currency,
                "quote_currency",
            )
            return f"{base_currency}{quote_currency}"

        return self._normalize_symbol(instrument.symbol)

    @staticmethod
    def _normalize_symbol_component(value: Optional[str], field_name: str) -> str:
        """Normalize one base or quote currency code for a Binance symbol.

        Args:
            value: Currency code supplied by the domain instrument.
            field_name: Name used to explain an invalid value.

        Returns:
            Uppercase alphanumeric currency code.

        Raises:
            ValueError: If the value is missing or not alphanumeric.
        """
        if value is None:
            raise ValueError(f"{field_name} must not be None.")

        normalized_value = value.strip().upper()
        if not normalized_value or not normalized_value.isalnum():
            raise ValueError(
                f"{field_name} must be a non-empty alphanumeric currency code."
            )
        return normalized_value

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize a canonical crypto pair symbol into Binance format.

        Args:
            symbol: Domain symbol, such as ``"BTC/USDT"`` or ``"BTCUSDT"``.

        Returns:
            Uppercase alphanumeric Binance symbol.

        Raises:
            ValueError: If the symbol cannot be normalized safely.
        """
        normalized_symbol = symbol.strip().upper()
        for separator in ("/", "-", "_"):
            normalized_symbol = normalized_symbol.replace(separator, "")

        if not normalized_symbol or not normalized_symbol.isalnum():
            raise ValueError(
                "Instrument symbol must be a non-empty alphanumeric pair or use "
                "the '/', '-', or '_' separators."
            )
        return normalized_symbol

    @classmethod
    def _to_binance_interval(cls, timeframe: TimeFrame) -> str:
        """Translate a domain timeframe into Binance's interval representation.

        Args:
            timeframe: Domain timeframe requested by the caller.

        Returns:
            Binance interval string for the supplied timeframe.

        Raises:
            ValueError: If the timeframe is not supported by this provider.
        """
        if not isinstance(timeframe, TimeFrame):
            raise ValueError("timeframe must be a TimeFrame enum value.")

        try:
            return cls._TIMEFRAME_TO_INTERVAL[timeframe]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported timeframe for Binance Spot: {timeframe!r}."
            ) from exc

    @classmethod
    def _to_market_bar(cls, kline: Sequence[Any]) -> Tuple[int, MarketBar]:
        """Convert one raw Binance kline into a normalized market bar.

        Args:
            kline: Raw twelve-field Binance kline response entry.

        Returns:
            The raw opening timestamp in milliseconds and its normalized bar.

        Raises:
            BinanceResponseError: If the kline does not match Binance's schema
                or contains invalid numeric values.
        """
        if not isinstance(kline, list) or len(kline) != 12:
            raise BinanceResponseError(
                "Binance returned a kline that does not contain exactly 12 fields."
            )

        raw_open_time = kline[0]
        if isinstance(raw_open_time, bool) or not isinstance(raw_open_time, int):
            raise BinanceResponseError(
                "Binance returned a kline with a non-integer open timestamp."
            )

        try:
            open_time = raw_open_time
            timestamp = cls._EPOCH + timedelta(milliseconds=open_time)
            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            volume = float(kline[5])
        except (TypeError, ValueError, OverflowError) as exc:
            raise BinanceResponseError(
                "Binance returned a kline with invalid timestamp or OHLCV values."
            ) from exc

        values = (open_price, high_price, low_price, close_price, volume)
        if not all(math.isfinite(value) for value in values):
            raise BinanceResponseError(
                "Binance returned a kline with non-finite OHLCV values."
            )
        if (
            open_price <= 0
            or high_price <= 0
            or low_price <= 0
            or close_price <= 0
            or volume < 0
            or high_price < max(open_price, close_price)
            or low_price > min(open_price, close_price)
            or low_price > high_price
        ):
            raise BinanceResponseError(
                "Binance returned a kline with inconsistent OHLCV values."
            )

        return open_time, MarketBar(
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )

    @staticmethod
    def _validate_time_range(start: datetime, end: datetime) -> None:
        """Validate the timestamp requirements of the BaseProvider contract.

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
        """Convert an inclusive datetime to the first eligible UTC millisecond.

        Args:
            value: Timezone-aware datetime to convert.

        Returns:
            Ceiling Unix timestamp in milliseconds.
        """
        return cls._ceil_division(cls._to_epoch_microseconds(value), 1_000)

    @classmethod
    def _to_end_milliseconds(cls, value: datetime) -> int:
        """Convert an exclusive datetime to the last eligible UTC millisecond.

        Args:
            value: Timezone-aware exclusive datetime boundary.

        Returns:
            Last Unix timestamp in milliseconds strictly before ``value``.
        """
        return cls._ceil_division(cls._to_epoch_microseconds(value), 1_000) - 1

    @classmethod
    def _to_epoch_microseconds(cls, value: datetime) -> int:
        """Convert a timezone-aware datetime into UTC Unix microseconds.

        Args:
            value: Timezone-aware datetime to convert.

        Returns:
            Unix timestamp in microseconds without floating-point rounding.
        """
        utc_value = value.astimezone(timezone.utc)
        delta = utc_value - cls._EPOCH
        return (
            (delta.days * 86_400 + delta.seconds) * 1_000_000
            + delta.microseconds
        )

    @staticmethod
    def _ceil_division(value: int, divisor: int) -> int:
        """Return the mathematical ceiling of an integer division.

        Args:
            value: Dividend to divide.
            divisor: Positive divisor.

        Returns:
            The smallest integer greater than or equal to ``value / divisor``.
        """
        return -(-value // divisor)


__all__ = [
    "BinanceAPIError",
    "BinanceProvider",
    "BinanceProviderError",
    "BinanceRequestError",
    "BinanceResponseError",
]
