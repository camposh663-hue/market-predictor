"""Simple, cost-aware backtest of a directional trading rule.

Treats every row as an independent round-trip trade sized by the predicted
direction, held for exactly one prediction horizon. This is a first-pass
vectorized backtest, not a portfolio simulator: it does not model
overlapping positions, capital constraints, or funding costs. It exists to
answer one narrow question cheaply -- does a model's directional edge
survive realistic exchange fees at all -- before investing in a real
execution/serving layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Aggregate outcome of a cost-aware directional backtest.

    Attributes:
        n_trades: Number of rows with a nonzero predicted direction (a
            "flat" row, where the prediction is exactly zero, takes no
            position and pays no cost).
        gross_mean_return: Mean per-trade log-return before costs.
        net_mean_return: Mean per-trade log-return after subtracting
            ``cost_per_trade`` from every trade.
        gross_total_return: Sum of gross per-trade returns.
        net_total_return: Sum of net per-trade returns.
        win_rate: Fraction of non-flat trades with a positive net return.
    """

    n_trades: int
    gross_mean_return: float
    net_mean_return: float
    gross_total_return: float
    net_total_return: float
    win_rate: float


def directional_backtest(
    y_true: pd.Series, y_pred: pd.Series, cost_per_trade: float
) -> BacktestResult:
    """Simulate a directional trading rule sized by predicted sign.

    Each row is treated as an independent round-trip trade: go long when
    the prediction is positive, short when negative, flat when exactly
    zero. A trade's gross return is ``sign(y_pred) * y_true`` -- betting on
    the predicted direction and collecting the true forward return.
    ``cost_per_trade`` (e.g. round-trip exchange fees, as a fraction) is
    subtracted from every non-flat trade.

    Args:
        y_true: True forward log-returns.
        y_pred: Predicted log-returns, aligned with ``y_true`` by index.
        cost_per_trade: Round-trip trading cost as a fraction (e.g. 0.002
            for 0.2%), charged once per non-flat row.

    Returns:
        Aggregate gross and net performance of the simulated trades.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` do not share the same
            index, either is empty, or ``cost_per_trade`` is negative.
    """
    if y_true.empty or y_pred.empty:
        raise ValueError("y_true and y_pred must not be empty")
    if not y_true.index.equals(y_pred.index):
        raise ValueError("y_true and y_pred must share the same index")
    if cost_per_trade < 0:
        raise ValueError("cost_per_trade must not be negative")

    direction = np.sign(y_pred.to_numpy())
    gross = direction * y_true.to_numpy()
    is_trade = direction != 0
    net = gross - np.where(is_trade, cost_per_trade, 0.0)

    n_trades = int(is_trade.sum())
    return BacktestResult(
        n_trades=n_trades,
        gross_mean_return=float(np.mean(gross[is_trade])) if n_trades else 0.0,
        net_mean_return=float(np.mean(net[is_trade])) if n_trades else 0.0,
        gross_total_return=float(np.sum(gross)),
        net_total_return=float(np.sum(net)),
        win_rate=float(np.mean(net[is_trade] > 0)) if n_trades else 0.0,
    )


__all__ = ["BacktestResult", "directional_backtest"]
