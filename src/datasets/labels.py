"""Alternative label functions for DatasetBuilder, beyond the default log-return."""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility_label(df: pd.DataFrame, bars_per_horizon: int) -> pd.Series:
    """Compute forward realized volatility over the next ``bars_per_horizon`` bars.

    Realized volatility is the standard measure used to check whether future
    price movement *magnitude* -- as opposed to direction -- is predictable:
    ``sqrt(sum of squared per-bar log-returns over the horizon)``. Unlike the
    signed log-return label, this target is always non-negative and has no
    notion of direction, so directional-accuracy metrics do not apply to it.

    Args:
        df: Feature-calculator output still holding the raw ``close`` column
            (as passed to a ``DatasetBuilder`` label function).
        bars_per_horizon: Number of bars the horizon spans.

    Returns:
        A series aligned with ``df``'s index. Only the trailing rows without
        a full forward window of bars hold ``NaN`` -- the horizon window
        always starts at ``t + 1``, so it never needs row ``t``'s own
        return (the one ``NaN`` from ``close.shift(1)``), unlike the
        default label. ``DatasetBuilder.build()``'s ``dropna()`` still
        drops any row an *engineered feature* leaves unfilled during its own
        warm-up, regardless of this label.
    """
    log_returns = np.log(df["close"] / df["close"].shift(1))
    squared_returns = log_returns**2
    forward_realized_variance = (
        squared_returns.rolling(window=bars_per_horizon).sum().shift(-bars_per_horizon)
    )
    return np.sqrt(forward_realized_variance)


__all__ = ["realized_volatility_label"]
