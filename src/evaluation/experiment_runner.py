"""Run a grid of experiments, strictly separating dev-set search from the holdout."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.datasets import DatasetBuilder
from src.domain import MarketBar, TimeFrame
from src.features import FeatureCalculator
from src.training import ModelTrainer, PurgedWalkForwardSplit, WalkForwardEvaluator
from src.training.metrics import directional_accuracy

from .experiment import ExperimentConfig, ExperimentResult
from .model_registry import TRAINER_FACTORIES


@dataclass(frozen=True)
class Dataset:
    """A (timeframe, horizon) dataset, already split into dev and holdout.

    Attributes:
        X_dev: Feature matrix available for model/hyperparameter selection.
        y_dev: Target aligned with ``X_dev``.
        X_test: Feature matrix for the final, untouched holdout.
        y_test: Target aligned with ``X_test``.
        splitter: The splitter used to produce this dev/test boundary,
            reused to generate walk-forward folds within ``X_dev``.
    """

    X_dev: pd.DataFrame
    y_dev: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    splitter: PurgedWalkForwardSplit


@dataclass(frozen=True)
class HoldoutResult:
    """Metrics from scoring one final configuration on its untouched test set.

    Attributes:
        config: The configuration that was evaluated.
        mae: Mean absolute error on the holdout.
        rmse: Root mean squared error on the holdout.
        directional_accuracy: Fraction of holdout rows with the correct
            predicted sign.
    """

    config: ExperimentConfig
    mae: float
    rmse: float
    directional_accuracy: float


class ExperimentRunner:
    """Search a grid of configurations using only dev-set walk-forward folds.

    The held-out test set carved out for each (timeframe, horizon) dataset
    is never touched by ``run``: it only exists to be scored once, by
    ``evaluate_holdout``, after a winning configuration has already been
    chosen from dev-set results. Running many configurations and picking
    whichever scores best *on the test set* would quietly reintroduce the
    same data-snooping bias the purged split is meant to prevent -- this
    split of responsibilities is what keeps a broad search honest.

    Datasets are built once per distinct (timeframe, horizon) pair and
    cached, since many hyperparameter combinations typically share one.

    Args:
        feature_calculator: Produces the feature matrix for every dataset
            built by this runner.
        n_splits: Number of walk-forward folds per configuration.
        test_size: Fraction of each dataset reserved as its final holdout.
    """

    def __init__(
        self,
        feature_calculator: FeatureCalculator,
        n_splits: int = 5,
        test_size: float = 0.15,
    ) -> None:
        self._builder = DatasetBuilder(feature_calculator=feature_calculator)
        self._n_splits = n_splits
        self._test_size = test_size
        self._datasets: Dict[Tuple[TimeFrame, timedelta], Dataset] = {}

    def run(
        self,
        configs: Sequence[ExperimentConfig],
        bars_by_timeframe: Mapping[TimeFrame, Sequence[MarketBar]],
    ) -> List[ExperimentResult]:
        """Evaluate every configuration on its dev-set walk-forward folds.

        Args:
            configs: Configurations to evaluate.
            bars_by_timeframe: Bars available for each timeframe referenced
                by ``configs``.

        Returns:
            One ``ExperimentResult`` per configuration, sorted best-first by
            mean directional accuracy, ties broken by lower variance across
            folds.

        Raises:
            KeyError: If a config's timeframe is missing from
                ``bars_by_timeframe``, or its model_name is not registered.
        """
        results = []
        for config in configs:
            dataset = self._dataset_for(config, bars_by_timeframe)
            evaluator = WalkForwardEvaluator(
                splitter=dataset.splitter,
                trainer_factory=self._factory_for(config),
            )
            fold_results = evaluator.evaluate(dataset.X_dev, dataset.y_dev)
            das = [fold.directional_accuracy for fold in fold_results]
            results.append(
                ExperimentResult(
                    config=config,
                    fold_results=tuple(fold_results),
                    mean_directional_accuracy=float(np.mean(das)),
                    std_directional_accuracy=float(np.std(das)),
                    mean_mae=float(np.mean([fold.mae for fold in fold_results])),
                    mean_rmse=float(np.mean([fold.rmse for fold in fold_results])),
                )
            )

        return sorted(
            results,
            key=lambda result: (-result.mean_directional_accuracy, result.std_directional_accuracy),
        )

    def evaluate_holdout(self, config: ExperimentConfig) -> Tuple[ModelTrainer, HoldoutResult]:
        """Fit ``config`` on all dev data and score it once on its holdout.

        Args:
            config: The winning configuration, already run through ``run``
                (its dataset must already be cached).

        Returns:
            The fitted trainer and its holdout metrics.

        Raises:
            KeyError: If ``config``'s (timeframe, horizon) dataset was never
                built via ``run``.
        """
        dataset = self._datasets[(config.timeframe, config.horizon)]
        trainer = self._factory_for(config)()
        trainer.fit(dataset.X_dev, dataset.y_dev)
        predictions = trainer.predict(dataset.X_test)

        result = HoldoutResult(
            config=config,
            mae=mean_absolute_error(dataset.y_test, predictions),
            rmse=mean_squared_error(dataset.y_test, predictions) ** 0.5,
            directional_accuracy=directional_accuracy(dataset.y_test, predictions),
        )
        return trainer, result

    def _dataset_for(
        self,
        config: ExperimentConfig,
        bars_by_timeframe: Mapping[TimeFrame, Sequence[MarketBar]],
    ) -> Dataset:
        key = (config.timeframe, config.horizon)
        if key not in self._datasets:
            bars = bars_by_timeframe[config.timeframe]
            X, y = self._builder.build(bars, config.timeframe, config.horizon)
            embargo = config.horizon // config.timeframe.duration
            splitter = PurgedWalkForwardSplit(
                n_splits=self._n_splits, embargo=embargo, test_size=self._test_size
            )
            dev_pos, test_pos = splitter.test_split(X.index)
            self._datasets[key] = Dataset(
                X_dev=X.iloc[dev_pos],
                y_dev=y.iloc[dev_pos],
                X_test=X.iloc[test_pos],
                y_test=y.iloc[test_pos],
                splitter=splitter,
            )
        return self._datasets[key]

    @staticmethod
    def _factory_for(config: ExperimentConfig):
        model_name = config.model_name
        hyperparams = config.hyperparams
        return lambda: TRAINER_FACTORIES[model_name](hyperparams)


__all__ = ["Dataset", "HoldoutResult", "ExperimentRunner"]
