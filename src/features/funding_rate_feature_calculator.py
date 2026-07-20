"""Feature calculator that adds perpetual-futures funding-rate context."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.domain import FundingRate, MarketBar

from .base_feature_calculator import FeatureCalculator


class FundingRateFeatureCalculator(FeatureCalculator):
    """Augment a base timeframe's features with perpetual funding-rate context.

    Funding rate is what longs pay shorts (or vice versa, when negative) on
    a perpetual future, settled on a fixed schedule (every 8h for BTCUSDT).
    A sustained positive rate means longs are paying a premium to stay
    leveraged long -- a documented real signal of crowded positioning, not
    derived from price at all, unlike every other feature in this project.

    Unlike ``MultiTimeframeFeatureCalculator``, no time shift is applied
    when merging: a funding settlement's timestamp *is* the exact instant
    the rate becomes known, not an interval's opening time that only closes
    later.

    Args:
        base_calculator: Computes the standard feature set for the
            timeframe being predicted.
        funding_rates: Historical funding-rate settlements to merge as
            context, in any order.
        lookback_periods: Number of most recent settlements summed into the
            cumulative-funding column (default 3, i.e. 24h of BTCUSDT's 8h
            settlement schedule).
    """

    def __init__(
        self,
        base_calculator: FeatureCalculator,
        funding_rates: Sequence[FundingRate],
        lookback_periods: int = 3,
    ) -> None:
        self._base_calculator = base_calculator
        self._funding_rates = funding_rates
        self._lookback_periods = lookback_periods

    def compute(self, bars: Sequence[MarketBar]) -> pd.DataFrame:
        """Compute the base feature set plus merged funding-rate context.

        Args:
            bars: Bars of the timeframe being predicted, in any order.

        Returns:
            The base calculator's DataFrame with two additional columns:
            ``funding_rate`` (the last known settled rate) and
            ``funding_rate_cum_{lookback_periods}`` (the sum of the last
            ``lookback_periods`` known settlements, a proxy for sustained
            positioning pressure). Rows before the first available
            settlement hold ``NaN`` for both, same as any other indicator's
            unfilled lookback window.
        """
        merged = self._base_calculator.compute(bars)

        ordered = sorted(self._funding_rates, key=lambda rate: rate.timestamp)
        funding_df = pd.DataFrame(
            {"funding_rate": [rate.rate for rate in ordered]},
            index=pd.DatetimeIndex([rate.timestamp for rate in ordered], tz="UTC"),
        )
        cum_column = f"funding_rate_cum_{self._lookback_periods}"
        funding_df[cum_column] = (
            funding_df["funding_rate"]
            .rolling(window=self._lookback_periods, min_periods=self._lookback_periods)
            .sum()
        )

        return pd.merge_asof(
            merged,
            funding_df,
            left_index=True,
            right_index=True,
            direction="backward",
        )


__all__ = ["FundingRateFeatureCalculator"]
