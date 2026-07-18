"""Tests for ExperimentRunner, using small synthetic data and configs for speed."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import List

from src.domain import MarketBar, TimeFrame
from src.evaluation import ExperimentConfig, ExperimentRunner
from src.features import IndicatorConfig, TechnicalIndicatorCalculator

_TINY_CONFIG = IndicatorConfig(
    sma_periods=(2,),
    ema_periods=(2,),
    rsi_period=2,
    macd_fast=2,
    macd_slow=3,
    macd_signal=2,
    bbands_period=2,
    bbands_std=2.0,
    atr_period=2,
    stoch_k=2,
    stoch_d=2,
    stoch_smooth_k=1,
    adx_period=2,
    mfi_period=2,
    return_periods=(1,),
    rel_volume_period=2,
)


def _bars(n: int) -> List[MarketBar]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = []
    for i in range(n):
        close = 100.0 + (i % 7) - (i % 3)
        bars.append(
            MarketBar(
                timestamp=start + timedelta(hours=i),
                open=close,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=10.0 + i % 5,
            )
        )
    return bars


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = ExperimentRunner(
            feature_calculator=TechnicalIndicatorCalculator(config=_TINY_CONFIG),
            n_splits=2,
            test_size=0.2,
        )
        self.bars_by_timeframe = {TimeFrame.ONE_HOUR: _bars(250)}
        self.configs = [
            ExperimentConfig(
                model_name="random_forest",
                timeframe=TimeFrame.ONE_HOUR,
                horizon=timedelta(hours=2),
                hyperparams={"n_estimators": 10, "max_depth": 2},
            ),
            ExperimentConfig(
                model_name="random_forest",
                timeframe=TimeFrame.ONE_HOUR,
                horizon=timedelta(hours=2),
                hyperparams={"n_estimators": 10, "max_depth": 4},
            ),
            ExperimentConfig(
                model_name="xgboost",
                timeframe=TimeFrame.ONE_HOUR,
                horizon=timedelta(hours=2),
                hyperparams={"n_estimators": 10, "max_depth": 2},
            ),
        ]

    def test_run_returns_one_result_per_config(self) -> None:
        results = self.runner.run(self.configs, self.bars_by_timeframe)

        self.assertEqual(len(results), len(self.configs))
        result_model_names = {r.config.model_name for r in results}
        self.assertEqual(result_model_names, {"random_forest", "xgboost"})

    def test_run_sorts_best_first_by_mean_directional_accuracy(self) -> None:
        results = self.runner.run(self.configs, self.bars_by_timeframe)

        scores = [(-r.mean_directional_accuracy, r.std_directional_accuracy) for r in results]
        self.assertEqual(scores, sorted(scores))

    def test_run_raises_on_missing_timeframe_bars(self) -> None:
        config = ExperimentConfig(
            model_name="random_forest",
            timeframe=TimeFrame.FOUR_HOURS,
            horizon=timedelta(hours=8),
            hyperparams={"n_estimators": 10, "max_depth": 2},
        )
        with self.assertRaises(KeyError):
            self.runner.run([config], self.bars_by_timeframe)

    def test_evaluate_holdout_requires_a_prior_run(self) -> None:
        with self.assertRaises(KeyError):
            self.runner.evaluate_holdout(self.configs[0])

    def test_evaluate_holdout_returns_fitted_trainer_and_metrics(self) -> None:
        self.runner.run(self.configs, self.bars_by_timeframe)

        trainer, result = self.runner.evaluate_holdout(self.configs[0])

        self.assertEqual(trainer.model_id, "random_forest")
        self.assertEqual(result.config, self.configs[0])
        self.assertGreaterEqual(result.directional_accuracy, 0.0)
        self.assertLessEqual(result.directional_accuracy, 1.0)
        self.assertGreaterEqual(result.mae, 0.0)
        self.assertGreaterEqual(result.rmse, 0.0)


if __name__ == "__main__":
    unittest.main()
