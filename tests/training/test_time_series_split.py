"""Tests for PurgedWalkForwardSplit."""

from __future__ import annotations

import unittest

import pandas as pd

from src.training import PurgedWalkForwardSplit


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


class PurgedWalkForwardSplitConstructionTests(unittest.TestCase):
    def test_raises_on_n_splits_below_two(self) -> None:
        with self.assertRaises(ValueError):
            PurgedWalkForwardSplit(n_splits=1, embargo=1)

    def test_raises_on_negative_embargo(self) -> None:
        with self.assertRaises(ValueError):
            PurgedWalkForwardSplit(n_splits=2, embargo=-1)

    def test_raises_on_test_size_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            PurgedWalkForwardSplit(n_splits=2, embargo=1, test_size=1.0)
        with self.assertRaises(ValueError):
            PurgedWalkForwardSplit(n_splits=2, embargo=1, test_size=0.0)


class TestSplitTests(unittest.TestCase):
    def test_reserves_final_fraction_as_test(self) -> None:
        splitter = PurgedWalkForwardSplit(n_splits=2, embargo=0, test_size=0.2)
        dev, test = splitter.test_split(_index(100))

        self.assertEqual(len(test), 20)
        self.assertEqual(list(test), list(range(80, 100)))

    def test_embargo_is_purged_between_dev_and_test(self) -> None:
        splitter = PurgedWalkForwardSplit(n_splits=2, embargo=5, test_size=0.2)
        dev, test = splitter.test_split(_index(100))

        self.assertEqual(dev[-1], 74)
        self.assertEqual(test[0], 80)

    def test_raises_when_too_few_rows_for_test_size_and_embargo(self) -> None:
        splitter = PurgedWalkForwardSplit(n_splits=2, embargo=50, test_size=0.5)
        with self.assertRaises(ValueError):
            splitter.test_split(_index(20))


class WalkForwardSplitTests(unittest.TestCase):
    def test_yields_n_splits_folds(self) -> None:
        splitter = PurgedWalkForwardSplit(n_splits=3, embargo=0)
        folds = list(splitter.split(_index(80)))

        self.assertEqual(len(folds), 3)

    def test_train_window_expands_each_fold(self) -> None:
        splitter = PurgedWalkForwardSplit(n_splits=3, embargo=0)
        folds = list(splitter.split(_index(80)))

        train_sizes = [len(train) for train, _ in folds]
        self.assertEqual(train_sizes, sorted(train_sizes))
        self.assertLess(train_sizes[0], train_sizes[-1])

    def test_never_shuffles_train_stays_before_validation(self) -> None:
        splitter = PurgedWalkForwardSplit(n_splits=3, embargo=0)
        for train, val in splitter.split(_index(80)):
            self.assertLess(train.max(), val.min())

    def test_embargo_gap_separates_train_and_validation(self) -> None:
        splitter = PurgedWalkForwardSplit(n_splits=3, embargo=4)
        for train, val in splitter.split(_index(80)):
            self.assertGreaterEqual(val.min() - train.max(), 5)

    def test_raises_when_not_enough_rows_for_folds(self) -> None:
        splitter = PurgedWalkForwardSplit(n_splits=5, embargo=10)
        with self.assertRaises(ValueError):
            list(splitter.split(_index(20)))


if __name__ == "__main__":
    unittest.main()
