"""Search a confidence threshold for the 15m->30min winner, strictly on dev data.

The full-frequency backtest showed the per-trade edge is far smaller than
round-trip costs. This checks whether trading only the model's
highest-magnitude predictions -- fewer trades, presumably a larger average
edge -- can turn that into a net-positive result.

Two-phase process, kept strictly separate:
1. Search every candidate keep_fraction using only the dev-set walk-forward
   folds (src/evaluation/threshold_search.py). The holdout is not touched.
2. Convert the single chosen fraction into a fixed magnitude cutoff learned
   from all of the dev data, then apply that fixed cutoff to the holdout
   exactly once. The cutoff is a fixed number decided in advance, not a
   percentile recomputed on the holdout itself -- recomputing "top X% of
   this month's predictions" only works in hindsight, not for a live rule.

Run with: python -m scripts.threshold_search
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.database import DataManager, ParquetRepository
from src.datasets import DatasetBuilder
from src.domain import AssetClass, Instrument, TimeFrame
from src.evaluation import directional_backtest, select_confidence_threshold
from src.features import TechnicalIndicatorCalculator
from src.training import PurgedWalkForwardSplit, XGBoostTrainer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INSTRUMENT = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
TIMEFRAME = TimeFrame.FIFTEEN_MINUTES
HORIZON = timedelta(minutes=30)
N_SPLITS = 5
TEST_SIZE = 0.15
ROUND_TRIP_COST = 0.002
KEEP_FRACTIONS = (1.0, 0.5, 0.25, 0.1, 0.05)
MIN_TRADES_PER_FOLD = 100
WINNING_HYPERPARAMS = {"max_depth": 3, "n_estimators": 200}


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

    embargo = HORIZON // TIMEFRAME.duration
    splitter = PurgedWalkForwardSplit(n_splits=N_SPLITS, embargo=embargo, test_size=TEST_SIZE)
    dev_pos, test_pos = splitter.test_split(X.index)
    X_dev, y_dev = X.iloc[dev_pos], y.iloc[dev_pos]
    X_test, y_test = X.iloc[test_pos], y.iloc[test_pos]

    print("Phase 1 -- searching keep_fraction on dev folds only (holdout untouched):\n")
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

    print("\nPhase 2 -- fitting on all dev data, confirming once on the untouched holdout:")
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

    print(
        f"\n  HOLDOUT unfiltered:  n_trades={unfiltered.n_trades:>6}  "
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


if __name__ == "__main__":
    main()
