"""Feature calculator that adds macro-regime context from higher timeframes."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from src.domain import MarketBar, TimeFrame

from .base_feature_calculator import FeatureCalculator


class MultiTimeframeFeatureCalculator(FeatureCalculator):
    """Augment a base timeframe's features with context from higher timeframes.

    A single-timeframe calculator only ever sees recent price action at its
    own resolution -- a 15-minute RSI cannot tell a model whether the
    market has been in a multi-day uptrend or downtrend. This wraps a base
    ``FeatureCalculator`` (used at the timeframe being predicted) and, for
    each configured higher timeframe, computes a small, deliberately
    curated pair of "regime" columns -- RSI and the longest-window SMA
    distance, not the full indicator set -- from that timeframe's own bars,
    then merges them onto the base index. Duplicating every indicator per
    context timeframe would reintroduce the same redundancy that justified
    switching SMA/EMA/Bollinger to relative distances in the first place
    (see docs/ESTADO_DEL_PROYECTO.md section 4.5); RSI and long-SMA trend
    are enough to capture "overbought/oversold" and "which way is the
    bigger picture trending" without that blow-up.

    Each context row is only attached to base rows at or after that
    context bar's *close* time, never its open time: a bar's indicators
    depend on its own OHLCV, so they are only knowable once the bar has
    finished, not the instant it opens. Base rows earlier than the first
    available context bar hold ``NaN`` for the context columns, same as
    any other indicator's unfilled lookback window -- ``DatasetBuilder``
    already drops such rows, so no special handling is needed downstream.

    Args:
        base_calculator: Computes the standard feature set for the
            timeframe being predicted.
        context_calculator: Computes the indicator set each higher
            timeframe's context columns are drawn from.
        context_bars: Higher-timeframe bars to derive context from, keyed
            by their ``TimeFrame``. Each timeframe contributes an
            ``rsi_{period}_{timeframe}`` and an
            ``sma_{longest_period}_dist_{timeframe}`` column.
    """

    def __init__(
        self,
        base_calculator: FeatureCalculator,
        context_calculator: FeatureCalculator,
        context_bars: Mapping[TimeFrame, Sequence[MarketBar]],
    ) -> None:
        self._base_calculator = base_calculator
        self._context_calculator = context_calculator
        self._context_bars = context_bars

    def compute(self, bars: Sequence[MarketBar]) -> pd.DataFrame:
        """Compute the base feature set plus merged higher-timeframe context.

        Args:
            bars: Bars of the timeframe being predicted, in any order.

        Returns:
            The base calculator's DataFrame with one additional RSI column
            and one additional long-SMA-distance column per configured
            context timeframe, each suffixed with that timeframe's value
            (e.g. ``rsi_14_4h``, ``sma_200_dist_4h``). Rows before a
            context timeframe's first closed bar, or before that
            timeframe's own indicator lookback is filled, hold ``NaN`` for
            its columns.

        Raises:
            ValueError: If ``bars`` is empty.
        """
        merged = self._base_calculator.compute(bars)

        for timeframe, context_bars in self._context_bars.items():
            context_df = self._context_calculator.compute(context_bars)
            rsi_column = next(c for c in context_df.columns if c.startswith("rsi_"))
            sma_dist_columns = sorted(
                (c for c in context_df.columns if c.startswith("sma_") and c.endswith("_dist")),
                key=lambda name: int(name.split("_")[1]),
            )
            trend_column = sma_dist_columns[-1]

            context_slice = context_df[[rsi_column, trend_column]].copy()
            # Shift to close time: this context only becomes knowable once
            # the bar it was computed from has finished.
            context_slice.index = context_slice.index + timeframe.duration
            context_slice.columns = [
                f"{rsi_column}_{timeframe.value}",
                f"{trend_column}_{timeframe.value}",
            ]

            merged = pd.merge_asof(
                merged,
                context_slice,
                left_index=True,
                right_index=True,
                direction="backward",
            )

        return merged


__all__ = ["MultiTimeframeFeatureCalculator"]
