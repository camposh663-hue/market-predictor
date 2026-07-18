"""Search models x hyperparameters x timeframe/horizon on BTC/USDT, strictly.

Every configuration is scored only on dev-set walk-forward folds (see
src/evaluation/experiment_runner.py). Only the single best configuration is
ever evaluated against its held-out test set, and only once, after the
search is finished -- searching broadly and scoring every candidate against
the same holdout would quietly reintroduce the data-snooping bias the
purged split exists to prevent.

Run with: python -m scripts.run_experiments
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.database import DataManager, ParquetRepository
from src.domain import AssetClass, Instrument, TimeFrame
from src.evaluation import ExperimentConfig, ExperimentRunner, hyperparameter_combinations
from src.features import TechnicalIndicatorCalculator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
INSTRUMENT = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
MODEL_NAMES = ("random_forest", "xgboost")
N_SPLITS = 5
TEST_SIZE = 0.15

# (timeframe, horizon): horizon must be an exact multiple of the timeframe's
# bar duration. Covers every native timeframe already backfilled in data/.
TIMEFRAME_HORIZONS = (
    (TimeFrame.FIFTEEN_MINUTES, timedelta(minutes=30)),
    (TimeFrame.ONE_HOUR, timedelta(hours=4)),
    (TimeFrame.FOUR_HOURS, timedelta(hours=24)),
    (TimeFrame.ONE_DAY, timedelta(days=3)),
)


def _load_bars(data_manager: DataManager) -> dict:
    timeframes = {tf for tf, _ in TIMEFRAME_HORIZONS}
    bars_by_timeframe = {}
    for timeframe in timeframes:
        bars = data_manager.get_bars(
            INSTRUMENT,
            timeframe,
            start=datetime(2017, 1, 1, tzinfo=timezone.utc),
            end=datetime.now(timezone.utc),
        )
        bars_by_timeframe[timeframe] = bars
        print(f"Loaded {len(bars)} bars for {timeframe.value}.")
    return bars_by_timeframe


def _build_configs() -> list:
    configs = []
    for timeframe, horizon in TIMEFRAME_HORIZONS:
        for model_name in MODEL_NAMES:
            for hyperparams in hyperparameter_combinations(model_name):
                configs.append(
                    ExperimentConfig(
                        model_name=model_name,
                        timeframe=timeframe,
                        horizon=horizon,
                        hyperparams=hyperparams,
                    )
                )
    return configs


def main() -> None:
    repository = ParquetRepository(base_path=DATA_DIR)
    data_manager = DataManager(repository)
    bars_by_timeframe = _load_bars(data_manager)

    runner = ExperimentRunner(
        feature_calculator=TechnicalIndicatorCalculator(),
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
    )

    all_configs = _build_configs()
    all_results = []
    print(f"\nTotal configs to run: {len(all_configs)}", flush=True)
    for i, config in enumerate(all_configs, start=1):
        start = time.time()
        print(f"[{i}/{len(all_configs)}] {config.label()} ...", flush=True)
        result = runner.run([config], bars_by_timeframe)[0]
        all_results.append(result)
        elapsed = time.time() - start
        print(
            f"  -> mean_da={result.mean_directional_accuracy:.3f} "
            f"(std {result.std_directional_accuracy:.3f})  mae={result.mean_mae:.5f}  "
            f"rmse={result.mean_rmse:.5f}  ({elapsed:.1f}s)",
            flush=True,
        )

    all_results.sort(key=lambda r: (-r.mean_directional_accuracy, r.std_directional_accuracy))

    leaderboard = pd.DataFrame(
        [
            {
                "rank": rank,
                "model": r.config.model_name,
                "timeframe": r.config.timeframe.value,
                "horizon": str(r.config.horizon),
                "hyperparams": dict(r.config.hyperparams),
                "mean_directional_accuracy": r.mean_directional_accuracy,
                "std_directional_accuracy": r.std_directional_accuracy,
                "mean_mae": r.mean_mae,
                "mean_rmse": r.mean_rmse,
            }
            for rank, r in enumerate(all_results, start=1)
        ]
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "experiment_results.csv"
    leaderboard.to_csv(report_path, index=False)
    print(f"\nFull leaderboard ({len(leaderboard)} configs) saved to {report_path}")

    print("\nTop 10 configs by dev-set mean directional accuracy:")
    print(leaderboard.head(10).to_string(index=False))

    print(
        "\nBest config per timeframe/horizon, fit on all dev data and scored "
        "once on its own held-out test set:"
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for timeframe, horizon in TIMEFRAME_HORIZONS:
        # all_results is already sorted best-first overall; filtering to this
        # group preserves that order, so the first match is the group winner.
        best_in_group = next(
            r for r in all_results if r.config.timeframe is timeframe and r.config.horizon == horizon
        )
        trainer, holdout = runner.evaluate_holdout(best_in_group.config)
        print(f"  {best_in_group.config.label()}")
        print(
            f"    HOLDOUT  mae={holdout.mae:.5f}  rmse={holdout.rmse:.5f}  "
            f"directional_accuracy={holdout.directional_accuracy:.3f}"
        )

        horizon_label = str(horizon).replace(" ", "").replace(":", "")
        model_path = MODELS_DIR / f"best_{trainer.model_id}_{timeframe.value}_{horizon_label}.joblib"
        joblib.dump(trainer, model_path)
        print(f"    saved to {model_path}")


if __name__ == "__main__":
    main()
