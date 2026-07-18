"""Strict, leakage-safe chronological splitting for time-series training data."""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd

_ArrayPair = Tuple[np.ndarray, np.ndarray]


class PurgedWalkForwardSplit:
    """Chronological walk-forward splitter with an embargo against label leakage.

    Financial time-series labels are forward-looking (a row's label is
    computed from a close price ``horizon`` bars ahead of it), so a plain
    chronological split can still leak: the last training rows' labels may
    depend on prices that fall inside the evaluation window. This splitter
    never shuffles, and drops an ``embargo`` number of rows immediately
    after every train segment before evaluation starts, so no label crosses
    the boundary in either direction.

    Two independent guarantees are provided:

    - ``test_split``: carves off the most recent ``test_size`` fraction of
      the data as a final holdout, purged with ``embargo``. This slice must
      only be evaluated once, after all model/hyperparameter selection is
      finished on the remaining ("dev") data.
    - ``split``: expanding-window walk-forward cross-validation over
      whatever index is passed in (typically the dev slice from
      ``test_split``), for robustness checks across market regimes before
      touching the test set.

    Args:
        n_splits: Number of walk-forward folds to yield from ``split``.
        embargo: Number of rows purged between a train segment and the
            evaluation segment that follows it. Should be at least the
            number of bars spanned by the prediction horizon, since that is
            how far labels look into the future.
        test_size: Fraction of the full series reserved as the final,
            untouched holdout test set.

    Raises:
        ValueError: If ``n_splits`` is not at least 2, ``embargo`` is
            negative, or ``test_size`` is not in ``(0, 1)``.
    """

    def __init__(
        self,
        n_splits: int,
        embargo: int,
        test_size: float = 0.15,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if embargo < 0:
            raise ValueError("embargo must not be negative")
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1, exclusive")

        self._n_splits = n_splits
        self._embargo = embargo
        self._test_size = test_size

    def test_split(self, index: pd.Index) -> _ArrayPair:
        """Carve off a final, embargoed holdout test set.

        Args:
            index: Chronologically sorted index to split (typically
                ``X.index`` from ``DatasetBuilder.build``).

        Returns:
            A tuple ``(dev_positions, test_positions)`` of positional
            (``iloc``-style) indices. ``dev_positions`` is every row
            available for cross-validation and model selection;
            ``test_positions`` is the final holdout, to be scored only once.

        Raises:
            ValueError: If ``index`` is too short for the configured
                ``test_size`` and ``embargo`` to leave any dev rows.
        """
        n = len(index)
        test_n = max(1, int(n * self._test_size))
        test_start = n - test_n
        dev_end = test_start - self._embargo

        if dev_end <= 0:
            raise ValueError(
                f"index has {n} rows, too few for test_size={self._test_size} "
                f"and embargo={self._embargo} to leave any dev rows"
            )

        return np.arange(0, dev_end), np.arange(test_start, n)

    def split(self, index: pd.Index) -> Iterator[_ArrayPair]:
        """Yield expanding-window, embargoed walk-forward folds.

        Args:
            index: Chronologically sorted index to split, typically the
                dev slice returned by ``test_split``.

        Yields:
            Tuples ``(train_positions, val_positions)`` of positional
            indices. ``train_positions`` always starts at row 0 and grows
            with each fold; ``val_positions`` immediately follows the
            ``embargo`` gap after it.

        Raises:
            ValueError: If ``index`` has too few rows to carve out
                ``n_splits`` non-empty folds under the configured embargo.
        """
        n = len(index)
        fold_size = n // (self._n_splits + 1)

        if fold_size <= self._embargo:
            raise ValueError(
                f"index has {n} rows, too few for n_splits={self._n_splits} "
                f"with embargo={self._embargo}"
            )

        for i in range(1, self._n_splits + 1):
            train_end = fold_size * i
            val_start = train_end + self._embargo
            val_end = min(val_start + fold_size, n)

            yield np.arange(0, train_end), np.arange(val_start, val_end)


__all__ = ["PurgedWalkForwardSplit"]
