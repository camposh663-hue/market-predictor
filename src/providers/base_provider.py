"""Provider-agnostic contracts for normalized financial market data.

Concrete providers are responsible for translating their external API contracts
into the types defined in this module. Code outside the providers package must
not depend on provider-specific payloads, symbols, or client libraries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Sequence


class AssetClass(str, Enum):
    """Identify the financial category of an instrument.

    Attributes:
        CRYPTO: Cryptocurrency instruments and trading pairs.
        EQUITY: Listed stocks and other equity instruments.
        FOREX: Foreign-exchange currency pairs.
        INDEX: Market indexes.
        OTHER: A supported asset class not yet represented by a dedicated value.
    """

    CRYPTO = "crypto"
    EQUITY = "equity"
    FOREX = "forex"
    INDEX = "index"
    OTHER = "other"


@dataclass(frozen=True)
class Instrument:
    """Describe a provider-independent financial instrument.

    Attributes:
        symbol: Stable, application-level symbol for the instrument.
        asset_class: Financial category to which the instrument belongs.
        venue: Optional exchange, market, or data venue identifier.
        base_currency: Optional base currency, when the instrument has one.
        quote_currency: Optional quote or settlement currency, when available.
    """

    symbol: str
    asset_class: AssetClass
    venue: Optional[str] = None
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None


@dataclass(frozen=True)
class MarketBar:
    """Represent one normalized OHLCV bar for a financial instrument.

    Attributes:
        timestamp: Start time of the bar as a timezone-aware UTC datetime.
        open: Opening price for the bar.
        high: Highest traded or quoted price for the bar.
        low: Lowest traded or quoted price for the bar.
        close: Closing price for the bar.
        volume: Traded volume when the provider supplies a comparable value.
    """

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Optional[Decimal]


@dataclass(frozen=True)
class MarketQuote:
    """Represent the most recent normalized quote for a financial instrument.

    Attributes:
        timestamp: Quote timestamp as a timezone-aware UTC datetime.
        bid: Best available bid price, when provided by the data source.
        ask: Best available ask price, when provided by the data source.
        last: Most recent transaction or reference price, when available.
    """

    timestamp: datetime
    bid: Optional[Decimal]
    ask: Optional[Decimal]
    last: Optional[Decimal]


class BaseProvider(ABC):
    """Define the normalized contract for financial market-data providers.

    Implementations isolate provider-specific authentication, symbol formats,
    pagination, transport, and response schemas. They must return normalized
    domain objects so that downstream modules remain provider independent.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Return the stable identifier of this provider.

        Returns:
            A stable, human-readable identifier, such as ``"binance"`` or
            ``"yahoo_finance"``.
        """
        ...

    @abstractmethod
    def get_instrument(
        self,
        symbol: str,
        *,
        asset_class: Optional[AssetClass] = None,
        venue: Optional[str] = None,
    ) -> Instrument:
        """Resolve an application symbol into normalized instrument metadata.

        Args:
            symbol: Application-level symbol, such as ``"BTC/USDT"``,
                ``"AAPL"``, or ``"EUR/USD"``. Implementations translate it
                to the provider's symbol format internally.
            asset_class: Optional asset-class constraint for ambiguous symbols.
            venue: Optional exchange, market, or data-venue constraint.

        Returns:
            The normalized instrument metadata for the requested symbol.
        """
        ...

    @abstractmethod
    def list_instruments(
        self,
        *,
        asset_class: Optional[AssetClass] = None,
        venue: Optional[str] = None,
    ) -> Sequence[Instrument]:
        """List the instruments available through this provider.

        Args:
            asset_class: Optional filter for a financial asset category.
            venue: Optional filter for an exchange, market, or data venue.

        Returns:
            Provider-supported instruments represented with normalized metadata.
        """
        ...

    @abstractmethod
    def get_supported_timeframes(
        self,
        instrument: Instrument,
    ) -> Sequence[str]:
        """Return normalized bar timeframes available for an instrument.

        Args:
            instrument: Instrument for which timeframe availability is queried.

        Returns:
            Supported timeframe labels, such as ``"1m"``, ``"1h"``, or
            ``"1d"``.
        """
        ...

    @abstractmethod
    def get_historical_bars(
        self,
        instrument: Instrument,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Sequence[MarketBar]:
        """Retrieve normalized historical OHLCV bars for a fixed time range.

        Args:
            instrument: Instrument whose historical data is requested.
            timeframe: Normalized aggregation interval, such as ``"1h"`` or
                ``"1d"``.
            start: Inclusive range start as a timezone-aware UTC datetime.
            end: Exclusive range end as a timezone-aware UTC datetime.

        Returns:
            Historical bars ordered chronologically from oldest to newest.
        """
        ...

    @abstractmethod
    def get_latest_quote(self, instrument: Instrument) -> MarketQuote:
        """Retrieve the latest available quote for an instrument.

        Args:
            instrument: Instrument for which the most recent quote is needed.

        Returns:
            A normalized quote containing available bid, ask, and last prices.
        """
        ...


__all__ = [
    "AssetClass",
    "BaseProvider",
    "Instrument",
    "MarketBar",
    "MarketQuote",
]
