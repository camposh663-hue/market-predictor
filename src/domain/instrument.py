"""Provider-independent financial instrument entity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .asset_class import AssetClass


@dataclass(frozen=True)
class Instrument:
    """Describe a financial instrument independently of a data provider.

    Attributes:
        symbol: Stable application-level symbol for the instrument.
        asset_class: Financial category to which the instrument belongs.
        venue: Optional exchange, market, or data-venue identifier.
        base_currency: Optional base currency, when the instrument has one.
        quote_currency: Optional quote or settlement currency, when available.
    """

    symbol: str
    asset_class: AssetClass
    venue: Optional[str] = None
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
