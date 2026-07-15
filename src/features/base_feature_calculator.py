"""Abstract contract for provider-agnostic feature calculation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd

from src.domain import MarketBar


class FeatureCalculator(ABC):
    """Define the contract for turning market bars into a feature matrix.

    Implementations own all indicator-library-specific concerns. They
    accept normalized ``MarketBar`` domain objects and return a single
    DataFrame that is the deliverable of this layer: OHLCV columns plus one
    column per computed feature, ready to feed a model.
    """

    @abstractmethod
    def compute(self, bars: Sequence[MarketBar]) -> pd.DataFrame:
        """Compute the feature matrix for a chronological series of bars.

        Args:
            bars: Bars to compute features from, in any order.

        Returns:
            A DataFrame indexed by timezone-aware UTC ``timestamp``, sorted
            chronologically from oldest to newest, containing the source
            OHLCV columns plus one column per computed feature. Rows before
            a feature's lookback window has been filled hold ``NaN`` for
            that feature; this is expected, not an error.

        Raises:
            ValueError: If ``bars`` is empty.
        """
        ...


__all__ = ["FeatureCalculator"]
