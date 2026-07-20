"""Abstract contract for provider-agnostic funding-rate storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Sequence

from src.domain import FundingRate, Instrument


class FundingRateRepository(ABC):
    """Define the contract for reading and writing normalized funding rates.

    Implementations own all storage-specific concerns, including file
    layout, partitioning, and serialization format. They only exchange
    domain objects with the rest of the application.
    """

    @abstractmethod
    def write_funding_rates(
        self,
        instrument: Instrument,
        funding_rates: Sequence[FundingRate],
    ) -> None:
        """Persist funding rates for an instrument.

        Args:
            instrument: Instrument the funding rates belong to.
            funding_rates: Rates to persist. Implementations must not assume
                the rates are sorted or free of duplicate timestamps.
        """
        ...

    @abstractmethod
    def read_funding_rates(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
    ) -> Sequence[FundingRate]:
        """Retrieve stored funding rates for an instrument within a time range.

        Args:
            instrument: Instrument whose funding rates are requested.
            start: Inclusive, timezone-aware UTC start timestamp.
            end: Exclusive, timezone-aware UTC end timestamp.

        Returns:
            Stored funding rates ordered chronologically from oldest to
            newest.
        """
        ...

    @abstractmethod
    def latest_timestamp(self, instrument: Instrument) -> Optional[datetime]:
        """Return the timestamp of the most recent stored funding rate, if any.

        Args:
            instrument: Instrument to inspect.

        Returns:
            The settlement timestamp of the newest stored funding rate, or
            ``None`` when none are stored for the instrument.
        """
        ...


__all__ = ["FundingRateRepository"]
