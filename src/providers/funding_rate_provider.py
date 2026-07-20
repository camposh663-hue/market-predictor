"""Abstract contract for provider-agnostic historical funding-rate data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence

from src.domain import FundingRate, Instrument


class FundingRateProvider(ABC):
    """Define the contract for normalized historical funding-rate data.

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
            provenance, such as ``"binance_futures"``.
        """
        ...

    @abstractmethod
    def get_historical_funding_rates(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
    ) -> Sequence[FundingRate]:
        """Retrieve normalized funding-rate settlements for a time range.

        Args:
            instrument: Instrument whose funding-rate history is requested.
            start: Inclusive, timezone-aware UTC start timestamp.
            end: Exclusive, timezone-aware UTC end timestamp.

        Returns:
            Normalized funding rates ordered chronologically from oldest to
            newest.
        """
        ...


__all__ = ["FundingRateProvider"]
