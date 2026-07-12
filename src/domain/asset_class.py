"""Financial asset-class definitions for the business domain."""

from enum import Enum


class AssetClass(str, Enum):
    """Identify the financial category of an instrument.

    Attributes:
        CRYPTO: Cryptocurrency instruments and trading pairs.
        EQUITY: Listed stocks and other equity instruments.
        FOREX: Foreign-exchange currency pairs.
        INDEX: Market indexes.
        OTHER: An asset class without a dedicated domain value yet.
    """

    CRYPTO = "crypto"
    EQUITY = "equity"
    FOREX = "forex"
    INDEX = "index"
    OTHER = "other"
