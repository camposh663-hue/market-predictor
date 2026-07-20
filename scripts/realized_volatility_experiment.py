"""Phase 1: is BTC/USDT realized volatility predictable, at all?

Before building any options/Deribit infrastructure (Phase 2), this checks
the cheap premise first: does an XGBoost model trained on the existing
technical indicators forecast forward realized volatility any better than
the naive "recent volatility persists" baseline? This is a regression
problem over an always-positive magnitude, not a directional classification
one, so it's scored with R^2 and Pearson correlation, not directional
accuracy -- WalkForwardEvaluator (coupled to directional_accuracy) does not
apply here, so this script runs its own walk-forward loop directly over
PurgedWalkForwardSplit.

Horizons checked, from short to long:
  - 15m -> 30min: huge dataset (~312k rows), a cheap sanity check of the
    general premise (does volatility cluster in this data at all).
  - 1h -> {4h, 1d, 3d, 7d, 14d}: built from hourly bars (~78k rows), so even
    a multi-day horizon keeps a large, densely-overlapping sample -- unlike
    building the same horizons from daily bars, where the step between
    observations is a full day instead of an hour. 7d in particular matters
    because weekly is Deribit's most liquid listed option maturity, more so
    than 30-day (DVOL's constant-maturity tenor, not an actual tradable
    expiry) or ultra-short-dated options.
  - 1d -> 30 days: smallest dataset (~3.3k rows), the horizon that would
    align directly with Deribit's DVOL index if it holds up.

There is no candidate search here (one model per horizon, fixed in
advance), so the holdout is evaluated once per horizon directly -- the
"gate before touching holdout" discipline used in the direction-prediction
experiments existed to prevent picking a winner among many candidates
before peeking; that doesn't apply to a single pre-registered comparison.

Run with: python -m scripts.realized_volatility_experiment
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.database import DataManager, ParquetRepository
from src.datasets import DatasetBuilder, realized_volatility_label
from src.domain import AssetClass, Instrument, MarketBar, TimeFrame
from src.features import TechnicalIndicatorCalculator
from src.training import PurgedWalkForwardSplit, XGBoostTrainer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INSTRUMENT = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
TEST_SIZE = 0.15


@dataclass(frozen=True)
class _Config:
    label: str
    timeframe: TimeFrame
    horizon: timedelta
    n_splits: int


CONFIGS = (
    _Config("15m -> 30min", TimeFrame.FIFTEEN_MINUTES, timedelta(minutes=30), n_splits=5),
    _Config("1h -> 4h", TimeFrame.ONE_HOUR, timedelta(hours=4), n_splits=5),
    _Config("1h -> 1d", TimeFrame.ONE_HOUR, timedelta(hours=24), n_splits=5),
    _Config("1h -> 3d", TimeFrame.ONE_HOUR, timedelta(days=3), n_splits=5),
    _Config("1h -> 7d", TimeFrame.ONE_HOUR, timedelta(days=7), n_splits=5),
    _Config("1h -> 14d", TimeFrame.ONE_HOUR, timedelta(days=14), n_splits=5),
    _Config("1d -> 30 days", TimeFrame.ONE_DAY, timedelta(days=30), n_splits=3),
)


def _past_realized_volatility(bars: Sequence[MarketBar], bars_per_horizon: int) -> pd.Series:
    """Backward-looking realized volatility: the naive persistence baseline.

    Same rolling sum of squared log-returns as ``realized_volatility_label``,
    but evaluated at each row's own position instead of shifted forward --
    i.e. "volatility over the last N bars", used as a no-model prediction of
    "volatility over the next N bars".
    """
    ordered = sorted(bars, key=lambda bar: bar.timestamp)
    index = pd.DatetimeIndex([bar.timestamp for bar in ordered], tz="UTC")
    closes = pd.Series([bar.close for bar in ordered], index=index)
    log_returns = np.log(closes / closes.shift(1))
    return (log_returns**2).rolling(window=bars_per_horizon).sum() ** 0.5


def _correlation(a: pd.Series, b: pd.Series) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


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

        builder = DatasetBuilder(
            feature_calculator=TechnicalIndicatorCalculator(),
            label_fn=realized_volatility_label,
        )
        X, y = builder.build(bars, config.timeframe, config.horizon)
        print(f"Dataset built: {len(X)} rows, {X.shape[1]} features.")

        bars_per_horizon = config.horizon // config.timeframe.duration
        baseline_full = _past_realized_volatility(bars, bars_per_horizon)

        embargo = bars_per_horizon
        splitter = PurgedWalkForwardSplit(
            n_splits=config.n_splits, embargo=embargo, test_size=TEST_SIZE
        )
        dev_pos, test_pos = splitter.test_split(X.index)
        X_dev, y_dev = X.iloc[dev_pos], y.iloc[dev_pos]
        X_test, y_test = X.iloc[test_pos], y.iloc[test_pos]

        print("\nWalk-forward dev folds (model vs. naive persistence baseline):")
        model_r2s: List[float] = []
        baseline_r2s: List[float] = []
        for fold, (train_pos, val_pos) in enumerate(splitter.split(X_dev.index), start=1):
            X_train, y_train = X_dev.iloc[train_pos], y_dev.iloc[train_pos]
            X_val, y_val = X_dev.iloc[val_pos], y_dev.iloc[val_pos]

            trainer = XGBoostTrainer()
            trainer.fit(X_train, y_train)
            model_pred = trainer.predict(X_val)
            baseline_pred = baseline_full.reindex(X_val.index)

            model_r2 = r2_score(y_val, model_pred)
            baseline_r2 = r2_score(y_val, baseline_pred)
            model_r2s.append(model_r2)
            baseline_r2s.append(baseline_r2)

            print(
                f"  fold {fold}: model R^2={model_r2:>7.4f} (corr={_correlation(y_val, model_pred):.4f})"
                f"   baseline R^2={baseline_r2:>7.4f} (corr={_correlation(y_val, baseline_pred):.4f})"
            )

        print(
            f"  mean: model R^2={np.mean(model_r2s):.4f}   "
            f"baseline R^2={np.mean(baseline_r2s):.4f}"
        )

        print("\nHoldout (fit on all dev data, scored once):")
        final_model = XGBoostTrainer()
        final_model.fit(X_dev, y_dev)
        model_pred = final_model.predict(X_test)
        baseline_pred = baseline_full.reindex(X_test.index)

        print(
            f"  model:    R^2={r2_score(y_test, model_pred):.4f}  "
            f"corr={_correlation(y_test, model_pred):.4f}"
        )
        print(
            f"  baseline: R^2={r2_score(y_test, baseline_pred):.4f}  "
            f"corr={_correlation(y_test, baseline_pred):.4f}"
        )


if __name__ == "__main__":
    main()
