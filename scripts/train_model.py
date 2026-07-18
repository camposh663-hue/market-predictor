"""Train and rigorously validate the first baseline model on BTC/USDT.

Validation strategy (see docs/ESTADO_DEL_PROYECTO.md for the full rationale):

1. The most recent TEST_SIZE fraction of the dataset is carved off as a
   final holdout test set, purged with an embargo, and is only ever scored
   once, after every other decision below is already final.
2. On the remaining ("dev") data, N_SPLITS purged, expanding-window
   walk-forward folds report per-fold metrics, so performance across
   different market regimes (bull/bear/crab) is visible instead of hidden
   behind one lucky split.
3. Only after reviewing those folds is a final model fit on all dev data
   and scored once on the untouched test set.

Run with: python -m scripts.train_model
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.database import DataManager, ParquetRepository
from src.datasets import DatasetBuilder
from src.domain import AssetClass, Instrument, TimeFrame
from src.features import TechnicalIndicatorCalculator
from src.training import PurgedWalkForwardSplit, RandomForestTrainer, WalkForwardEvaluator
from src.training.metrics import directional_accuracy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
INSTRUMENT = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
TIMEFRAME = TimeFrame.ONE_HOUR
HORIZON = timedelta(hours=4)
N_SPLITS = 5
TEST_SIZE = 0.15


def main() -> None:
    repository = ParquetRepository(base_path=DATA_DIR)
    data_manager = DataManager(repository)
    bars = data_manager.get_bars(
        INSTRUMENT,
        TIMEFRAME,
        start=datetime(2017, 1, 1, tzinfo=timezone.utc),
        end=datetime.now(timezone.utc),
    )

    builder = DatasetBuilder(feature_calculator=TechnicalIndicatorCalculator())
    X, y = builder.build(bars, TIMEFRAME, HORIZON)
    print(f"Dataset: {len(X)} rows, {X.shape[1]} features.")

    embargo_bars = HORIZON // TIMEFRAME.duration
    splitter = PurgedWalkForwardSplit(
        n_splits=N_SPLITS, embargo=embargo_bars, test_size=TEST_SIZE
    )
    dev_pos, test_pos = splitter.test_split(X.index)
    X_dev, y_dev = X.iloc[dev_pos], y.iloc[dev_pos]
    X_test, y_test = X.iloc[test_pos], y.iloc[test_pos]
    print(
        f"Dev set: {len(X_dev)} rows "
        f"({X_dev.index[0]} -> {X_dev.index[-1]}). "
        f"Held-out test set: {len(X_test)} rows "
        f"({X_test.index[0]} -> {X_test.index[-1]})."
    )

    evaluator = WalkForwardEvaluator(
        splitter=splitter,
        trainer_factory=lambda: RandomForestTrainer(),
    )
    fold_results = evaluator.evaluate(X_dev, y_dev)

    print("\nWalk-forward cross-validation (dev set only, model selection use):")
    for result in fold_results:
        print(
            f"  fold {result.fold}: train={result.train_size:>6} "
            f"val={result.val_size:>5}  mae={result.mae:.5f}  "
            f"rmse={result.rmse:.5f}  "
            f"directional_accuracy={result.directional_accuracy:.3f}"
        )
    mean_da = np.mean([r.directional_accuracy for r in fold_results])
    std_da = np.std([r.directional_accuracy for r in fold_results])
    print(f"  mean directional_accuracy = {mean_da:.3f} (std {std_da:.3f})")

    print("\nFitting final model on all dev data, scoring the held-out test set once...")
    final_model = RandomForestTrainer()
    final_model.fit(X_dev, y_dev)
    test_predictions = final_model.predict(X_test)

    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5
    test_da = directional_accuracy(y_test, test_predictions)
    print(
        f"  TEST  mae={test_mae:.5f}  rmse={test_rmse:.5f}  "
        f"directional_accuracy={test_da:.3f}"
    )

    print("\nTop 10 most important features:")
    print(final_model.feature_importances.head(10).to_string())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{final_model.model_id}_btcusdt_1h_4h.joblib"
    joblib.dump(final_model, model_path)
    print(f"\nSaved final model to {model_path}")


if __name__ == "__main__":
    main()
