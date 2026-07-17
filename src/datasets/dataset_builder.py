"""Combine engineered features with a future log-return label for training."""

from __future__ import annotations

from datetime import timedelta
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from src.domain import MarketBar, TimeFrame
from src.features import FeatureCalculator

_RAW_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


class DatasetBuilder:
    """Build a model-ready ``(X, y)`` table from raw bars.

    Combines the feature matrix produced by a ``FeatureCalculator`` with a
    future log-return label over a configurable horizon. Raw OHLCV columns
    are excluded from ``X``: they are not comparable across instruments with
    different price scales, so only engineered, scale-relative columns
    (indicators, past returns, temporal encodings) are used as features.

    Args:
        feature_calculator: Produces the feature matrix from raw bars.
    """

    def __init__(self, feature_calculator: FeatureCalculator) -> None:
        self._feature_calculator = feature_calculator

    def build(
        self,
        bars: Sequence[MarketBar],
        timeframe: TimeFrame,
        horizon: timedelta,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Build the training table for a horizon expressed as a duration.

        Args:
            bars: Bars to build the dataset from, in any order.
            timeframe: Aggregation interval ``bars`` were sampled at. Used to
                convert ``horizon`` into a number of bars.
            horizon: Prediction horizon, e.g. ``timedelta(hours=4)``. Must be
                a positive, exact multiple of ``timeframe``'s bar duration.

        Returns:
            A tuple ``(X, y)`` sharing the same UTC-timestamp index: ``X``
            holds only engineered feature columns (technical indicators,
            past returns, temporal encodings), and ``y`` is the aligned
            future log-return label, ``ln(close[t + horizon] / close[t])``.
            Rows without a full indicator lookback or without a future bar
            to compute the label from are dropped from both.

        Raises:
            ValueError: If ``bars`` is empty, or ``horizon`` is not a
                positive, exact multiple of the timeframe's bar duration.
        """
        bars_per_horizon = self._bars_per_horizon(timeframe, horizon)

        df = self._feature_calculator.compute(bars)
        df["label"] = np.log(df["close"].shift(-bars_per_horizon) / df["close"])
        df = df.dropna()

        y = df["label"]
        x = df.drop(columns=["label", *_RAW_PRICE_COLUMNS])
        return x, y

    @staticmethod
    def _bars_per_horizon(timeframe: TimeFrame, horizon: timedelta) -> int:
        """Convert a prediction horizon into a whole number of bars.

        Args:
            timeframe: Aggregation interval bars are sampled at.
            horizon: Prediction horizon to convert.

        Returns:
            The number of bars ``horizon`` spans at ``timeframe``'s interval.

        Raises:
            ValueError: If ``horizon`` is not positive or not an exact
                multiple of the timeframe's bar duration.
        """
        if horizon <= timedelta(0):
            raise ValueError("horizon must be positive")
        bar_duration = timeframe.duration
        if horizon % bar_duration != timedelta(0):
            raise ValueError(
                f"horizon ({horizon}) must be an exact multiple of the "
                f"{timeframe.value} bar duration ({bar_duration})"
            )
        return horizon // bar_duration


__all__ = ["DatasetBuilder"]
