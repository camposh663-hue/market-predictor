"""Tests for TimeFrame.duration."""

from __future__ import annotations

import unittest
from datetime import timedelta

from src.domain import TimeFrame


class TimeFrameDurationTests(unittest.TestCase):
    def test_duration_matches_each_declared_interval(self) -> None:
        self.assertEqual(TimeFrame.ONE_MINUTE.duration, timedelta(minutes=1))
        self.assertEqual(TimeFrame.FIVE_MINUTES.duration, timedelta(minutes=5))
        self.assertEqual(TimeFrame.FIFTEEN_MINUTES.duration, timedelta(minutes=15))
        self.assertEqual(TimeFrame.ONE_HOUR.duration, timedelta(hours=1))
        self.assertEqual(TimeFrame.FOUR_HOURS.duration, timedelta(hours=4))
        self.assertEqual(TimeFrame.ONE_DAY.duration, timedelta(days=1))


if __name__ == "__main__":
    unittest.main()
