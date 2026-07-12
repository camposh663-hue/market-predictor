"""Abstract contracts for provider-agnostic historical market data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence

from src.domain import Instrument, MarketBar, TimeFrame


class BaseProvider(ABC):
    """Define the contract for normalized historical OHLCV data providers.

    Implementations translate domain objects into provider-specific request
    formats and normalize external responses before returning them. Provider
    authentication, transport, pagination, and API schemas must remain inside
    the concrete provider implementation.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Return the stable identifier for this provider.

        Returns:
            A stable identifier used for configuration, logging, and data
            provenance, such as ``"binance"`` or ``"yahoo_finance"``.
        """
        ...

    @abstractmethod
    def get_historical_bars(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> Sequence[MarketBar]:
        """Retrieve normalized OHLCV bars for an instrument and time range.

        Implementations translate the domain instrument and timeframe into
        provider-specific identifiers and handle transport and pagination.

        Args:
            instrument: Instrument whose OHLCV history is requested.
            timeframe: Requested aggregation interval.
            start: Inclusive, timezone-aware UTC start timestamp.
            end: Exclusive, timezone-aware UTC end timestamp.

        Returns:
            Normalized bars ordered chronologically from oldest to newest.
            Each timestamp represents the opening time of its interval.
        """
        ...


__all__ = ["BaseProvider"]
