"""Test whether Binance Futures funding-rate context improves the 15m->30min
XGBoost winner over the single-timeframe baseline (51.6% dev / 51.8% holdout
directional accuracy -- see docs/ESTADO_DEL_PROYECTO.md section 4.10), and
if so, re-run confidence threshold selection and the cost-aware backtest
with it.

Follows the same dev/holdout discipline as
scripts/multi_timeframe_experiment.py: the holdout is only scored once, and
only if the dev-set walk-forward result with the expanded feature set
actually clears the single-timeframe baseline.

Run with: python -m scripts.funding_rate_experiment
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np

from src.database import DataManager, ParquetFundingRateRepository, ParquetRepository
from src.datasets import DatasetBuilder
from src.domain import AssetClass, Instrument, TimeFrame
from src.evaluation import directional_backtest, select_confidence_threshold
from src.features import FundingRateFeatureCalculator, TechnicalIndicatorCalculator
from src.training import PurgedWalkForwardSplit, WalkForwardEvaluator, XGBoostTrainer
from src.training.metrics import directional_accuracy

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
INSTRUMENT = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
TIMEFRAME = TimeFrame.FIFTEEN_MINUTES
HORIZON = timedelta(minutes=30)
N_SPLITS = 5
TEST_SIZE = 0.15
ROUND_TRIP_COST = 0.002
KEEP_FRACTIONS = (1.0, 0.5, 0.25, 0.1, 0.05)
MIN_TRADES_PER_FOLD = 100
WINNING_HYPERPARAMS = {"max_depth": 3, "n_estimators": 200}

# Single-timeframe winner's dev-set mean directional accuracy, from the
# 32-configuration search already run (scripts/run_experiments.py). The bar
# funding-rate context must clear before the holdout is touched.
BASELINE_DEV_ACCURACY = 0.516


def main() -> None:
    repository = ParquetRepository(base_path=DATA_DIR)
    data_manager = DataManager(repository)
    funding_repository = ParquetFundingRateRepository(base_path=DATA_DIR)

    bars_15m = data_manager.get_bars(
        INSTRUMENT,
        TIMEFRAME,
        start=datetime(2017, 1, 1, tzinfo=timezone.utc),
        end=datetime.now(timezone.utc),
    )
    funding_rates = funding_repository.read_funding_rates(
        INSTRUMENT,
        start=datetime(2017, 1, 1, tzinfo=timezone.utc),
        end=datetime.now(timezone.utc),
    )
    print(f"Loaded {len(bars_15m)} {TIMEFRAME.value} bars.")
    print(f"Loaded {len(funding_rates)} funding-rate settlements.")

    feature_calculator = FundingRateFeatureCalculator(
        base_calculator=TechnicalIndicatorCalculator(),
        funding_rates=funding_rates,
    )
    builder = DatasetBuilder(feature_calculator=feature_calculator)
    X, y = builder.build(bars_15m, TIMEFRAME, HORIZON)
    print(f"\nDataset built: {len(X)} rows, {X.shape[1]} features.")

    embargo = HORIZON // TIMEFRAME.duration
    splitter = PurgedWalkForwardSplit(n_splits=N_SPLITS, embargo=embargo, test_size=TEST_SIZE)
    dev_pos, test_pos = splitter.test_split(X.index)
    X_dev, y_dev = X.iloc[dev_pos], y.iloc[dev_pos]
    X_test, y_test = X.iloc[test_pos], y.iloc[test_pos]

    print("\nPhase 1 -- walk-forward dev accuracy with funding-rate context:")
    evaluator = WalkForwardEvaluator(
        splitter=splitter,
        trainer_factory=lambda: XGBoostTrainer(**WINNING_HYPERPARAMS),
    )
    fold_results = evaluator.evaluate(X_dev, y_dev)
    das = [fold.directional_accuracy for fold in fold_results]
    mean_da = float(np.mean(das))
    std_da = float(np.std(das))
    for fold in fold_results:
        print(
            f"  fold {fold.fold}: da={fold.directional_accuracy:.4f}  "
            f"mae={fold.mae:.5f}  rmse={fold.rmse:.5f}"
        )
    print(f"  mean dev directional accuracy: {mean_da:.4f} (std {std_da:.4f})")
    print(f"  single-timeframe baseline:     {BASELINE_DEV_ACCURACY:.4f}")

    if mean_da <= BASELINE_DEV_ACCURACY:
        print(
            "\nFunding-rate context did not beat the single-timeframe dev "
            "baseline. Stopping here without touching the holdout, per the "
            "project's no-data-snooping discipline (see "
            "docs/ESTADO_DEL_PROYECTO.md section 4.10)."
        )
        return

    print("\nPhase 2 -- confidence threshold search on dev folds only (holdout untouched):")
    chosen_fraction, candidates = select_confidence_threshold(
        X_dev,
        y_dev,
        splitter,
        trainer_factory=lambda: XGBoostTrainer(**WINNING_HYPERPARAMS),
        keep_fractions=KEEP_FRACTIONS,
        cost_per_trade=ROUND_TRIP_COST,
        min_trades_per_fold=MIN_TRADES_PER_FOLD,
    )
    for c in candidates:
        eligible = "eligible" if c.mean_trades_per_fold >= MIN_TRADES_PER_FOLD else "excluded (too few trades)"
        print(
            f"  keep_fraction={c.keep_fraction:<5} trades/fold={c.mean_trades_per_fold:>8.1f}  "
            f"net_mean_return={c.mean_net_return:>10.6f}  std={c.std_net_return:.6f}  ({eligible})"
        )
    print(f"\nChosen keep_fraction: {chosen_fraction}")

    print("\nPhase 3 -- fitting on all dev data, confirming once on the untouched holdout:")
    final_model = XGBoostTrainer(**WINNING_HYPERPARAMS)
    final_model.fit(X_dev, y_dev)

    dev_predictions = final_model.predict(X_dev)
    magnitude_cutoff = dev_predictions.abs().quantile(1.0 - chosen_fraction)
    print(f"  magnitude cutoff learned from dev data: {magnitude_cutoff:.6f}")

    test_predictions = final_model.predict(X_test)
    mask = test_predictions.abs() >= magnitude_cutoff
    filtered_test_predictions = test_predictions.where(mask, 0.0)

    unfiltered = directional_backtest(y_test, test_predictions, ROUND_TRIP_COST)
    filtered = directional_backtest(y_test, filtered_test_predictions, ROUND_TRIP_COST)
    holdout_da = directional_accuracy(y_test, test_predictions)

    print(f"\n  HOLDOUT directional_accuracy={holdout_da:.4f}")
    print(
        f"  HOLDOUT unfiltered:  n_trades={unfiltered.n_trades:>6}  "
        f"net_mean_bp={unfiltered.net_mean_return * 10_000:>8.3f}  "
        f"net_total_%={unfiltered.net_total_return * 100:>10.3f}  "
        f"win_rate={unfiltered.win_rate:.3f}"
    )
    print(
        f"  HOLDOUT filtered  :  n_trades={filtered.n_trades:>6}  "
        f"net_mean_bp={filtered.net_mean_return * 10_000:>8.3f}  "
        f"net_total_%={filtered.net_total_return * 100:>10.3f}  "
        f"win_rate={filtered.win_rate:.3f}"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"best_{final_model.model_id}_15m_03000_funding_rate.joblib"
    joblib.dump(final_model, model_path)
    print(f"\n  saved to {model_path}")


if __name__ == "__main__":
    main()
