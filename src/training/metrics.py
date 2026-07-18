"""Evaluation metrics for log-return predictions.

MAE and RMSE are the standard regression error metrics and are already
provided by scikit-learn, so they are used directly rather than
reimplemented here. Directional accuracy has no scikit-learn equivalent and
matters more than raw error for a trading signal: getting the sign of the
future move right is what a prediction is used for downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute the fraction of predictions that match the true sign.

    Args:
        y_true: True log-return labels.
        y_pred: Predicted log-return values, aligned with ``y_true`` by
            index.

    Returns:
        The fraction, in ``[0, 1]``, of rows where ``y_true`` and ``y_pred``
        have the same sign. A prediction of exactly zero never counts as a
        match, since it asserts no direction.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` do not share the same
            index, or either is empty.
    """
    if y_true.empty or y_pred.empty:
        raise ValueError("y_true and y_pred must not be empty")
    if not y_true.index.equals(y_pred.index):
        raise ValueError("y_true and y_pred must share the same index")

    matches = np.sign(y_true.to_numpy()) == np.sign(y_pred.to_numpy())
    return float(np.mean(matches))


__all__ = ["directional_accuracy"]
