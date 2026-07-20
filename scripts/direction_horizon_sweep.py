"""Does predicting *direction* (not volatility) hold up at longer horizons,
if built on a properly large sample?

The original search (scripts/run_experiments.py) tested 1d -> 3 days on
daily bars: it looked competitive in dev (51.2%) but collapsed to 47.6% in
holdout -- worse than chance, attributed to noise from a small dataset
(only 3,257 daily rows). This retests longer directional horizons the same
way the volatility sweep did (scripts/realized_volatility_experiment.py):
built from 1h bars (~78k rows) instead of 1d bars, so even a multi-day
horizon keeps a large, dense sample instead of a thin one.

Uses the existing, already-tested WalkForwardEvaluator/directional_accuracy
machinery unchanged -- this is the same label DatasetBuilder always used by
default, just swept across more horizons on a finer base timeframe.

Run with: python -m scripts.direction_horizon_sweep
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.database import DataManager, ParquetRepository
from src.datasets import DatasetBuilder
from src.domain import AssetClass, Instrument, TimeFrame
from src.features import TechnicalIndicatorCalculator
from src.training import PurgedWalkForwardSplit, WalkForwardEvaluator, XGBoostTrainer
from src.training.metrics import directional_accuracy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INSTRUMENT = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
TEST_SIZE = 0.15
N_SPLITS = 5


@dataclass(frozen=True)
class _Config:
    label: str
    timeframe: TimeFrame
    horizon: timedelta


CONFIGS = (
    _Config("1h -> 4h", TimeFrame.ONE_HOUR, timedelta(hours=4)),
    _Config("1h -> 1d", TimeFrame.ONE_HOUR, timedelta(hours=24)),
    _Config("1h -> 3d", TimeFrame.ONE_HOUR, timedelta(days=3)),
    _Config("1h -> 7d", TimeFrame.ONE_HOUR, timedelta(days=7)),
    _Config("1h -> 14d", TimeFrame.ONE_HOUR, timedelta(days=14)),
)


def main() -> None:
    repository = ParquetRepository(base_path=DATA_DIR)
    data_manager = DataManager(repository)

    for config in CONFIGS:
        print(f"\n{'=' * 70}\n{config.label}\n{'=' * 70}")

        bars = data_manager.get_bars(
            INSTRUMENT,
            config.timeframe,
            start=datetime(2017, 1, 1, tzinfo=timezone.utc),
            end=datetime.now(timezone.utc),
        )
        print(f"Loaded {len(bars)} {config.timeframe.value} bars.")

        builder = DatasetBuilder(feature_calculator=TechnicalIndicatorCalculator())
        X, y = builder.build(bars, config.timeframe, config.horizon)
        print(f"Dataset built: {len(X)} rows, {X.shape[1]} features.")

        embargo = config.horizon // config.timeframe.duration
        splitter = PurgedWalkForwardSplit(n_splits=N_SPLITS, embargo=embargo, test_size=TEST_SIZE)
        dev_pos, test_pos = splitter.test_split(X.index)
        X_dev, y_dev = X.iloc[dev_pos], y.iloc[dev_pos]
        X_test, y_test = X.iloc[test_pos], y.iloc[test_pos]

        evaluator = WalkForwardEvaluator(splitter=splitter, trainer_factory=XGBoostTrainer)
        fold_results = evaluator.evaluate(X_dev, y_dev)
        das = [fold.directional_accuracy for fold in fold_results]
        for fold in fold_results:
            print(f"  fold {fold.fold}: da={fold.directional_accuracy:.4f}")
        print(f"  mean dev directional accuracy: {np.mean(das):.4f} (std {np.std(das):.4f})")

        final_model = XGBoostTrainer()
        final_model.fit(X_dev, y_dev)
        predictions = final_model.predict(X_test)
        holdout_da = directional_accuracy(y_test, predictions)
        print(f"  HOLDOUT directional accuracy: {holdout_da:.4f}")


if __name__ == "__main__":
    main()
