"""How low would Binance fees need to go for the existing models to break even?

Net return per trade is gross return minus a fixed round-trip cost. So the
gross mean return at a given confidence filter *is*, by definition, the
maximum round-trip cost the strategy could tolerate and still break even.
This script reports that number directly, instead of guessing a fee and
checking if it clears -- it answers "what fee would we need", which is the
question that matters for judging whether a lower-cost venue/tier could
realistically close the gap (see docs/ESTADO_DEL_PROYECTO.md section 9,
lever 1: "atacar el coste, no solo el edge").

No retraining happens here: both models were already fit once on all dev
data (scripts/run_experiments.py for the single-timeframe winner,
scripts/multi_timeframe_experiment.py for the multi-timeframe one) and are
loaded from disk. Re-deriving the confidence cutoff from dev predictions and
scoring gross (cost-free) returns on the holdout is a read-only diagnostic
on an already-frozen result, not a new search -- no new configuration is
being chosen by looking at the holdout here, so it doesn't reintroduce the
data-snooping bias the rest of this project's methodology avoids.

Run with: python -m scripts.breakeven_cost_analysis
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.database import DataManager, ParquetRepository
from src.datasets import DatasetBuilder
from src.domain import AssetClass, Instrument, TimeFrame
from src.evaluation import directional_backtest
from src.features import MultiTimeframeFeatureCalculator, TechnicalIndicatorCalculator
from src.training import PurgedWalkForwardSplit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
INSTRUMENT = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
TIMEFRAME = TimeFrame.FIFTEEN_MINUTES
HORIZON = timedelta(minutes=30)
N_SPLITS = 5
TEST_SIZE = 0.15
KEEP_FRACTIONS = (1.0, 0.5, 0.25, 0.1, 0.05)

MODEL_VARIANTS = (
    ("single-timeframe (15m only)", "best_xgboost_15m_03000.joblib", False),
    ("multi-timeframe (15m + 4h/1d)", "best_xgboost_15m_03000_multi_timeframe.joblib", True),
)


def _load_bars(data_manager: DataManager, timeframe: TimeFrame):
    return data_manager.get_bars(
        INSTRUMENT,
        timeframe,
        start=datetime(2017, 1, 1, tzinfo=timezone.utc),
        end=datetime.now(timezone.utc),
    )


def main() -> None:
    repository = ParquetRepository(base_path=DATA_DIR)
    data_manager = DataManager(repository)

    bars_15m = _load_bars(data_manager, TIMEFRAME)
    context_bars = {
        TimeFrame.FOUR_HOURS: _load_bars(data_manager, TimeFrame.FOUR_HOURS),
        TimeFrame.ONE_DAY: _load_bars(data_manager, TimeFrame.ONE_DAY),
    }

    embargo = HORIZON // TIMEFRAME.duration
    splitter = PurgedWalkForwardSplit(n_splits=N_SPLITS, embargo=embargo, test_size=TEST_SIZE)

    rows = []
    for label, filename, use_multi_timeframe in MODEL_VARIANTS:
        feature_calculator = (
            MultiTimeframeFeatureCalculator(
                base_calculator=TechnicalIndicatorCalculator(),
                context_calculator=TechnicalIndicatorCalculator(),
                context_bars=context_bars,
            )
            if use_multi_timeframe
            else TechnicalIndicatorCalculator()
        )
        builder = DatasetBuilder(feature_calculator=feature_calculator)
        X, y = builder.build(bars_15m, TIMEFRAME, HORIZON)

        dev_pos, test_pos = splitter.test_split(X.index)
        X_dev, X_test, y_test = X.iloc[dev_pos], X.iloc[test_pos], y.iloc[test_pos]

        model = joblib.load(MODELS_DIR / filename)
        dev_predictions = model.predict(X_dev)
        test_predictions = model.predict(X_test)

        for fraction in KEEP_FRACTIONS:
            cutoff = dev_predictions.abs().quantile(1.0 - fraction)
            mask = test_predictions.abs() >= cutoff
            filtered = test_predictions.where(mask, 0.0)
            gross = directional_backtest(y_test, filtered, cost_per_trade=0.0)

            rows.append(
                {
                    "model": label,
                    "keep_fraction": fraction,
                    "n_trades": gross.n_trades,
                    "gross_win_rate": gross.win_rate,
                    "breakeven_cost_bp": gross.gross_mean_return * 10_000,
                }
            )

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(
        "Breakeven round-trip cost = the fee level at which net return per "
        "trade turns exactly zero (gross mean return, cost=0). Current "
        "Binance spot taker baseline used elsewhere in this project: 20 bp "
        "round-trip (0.1% x 2, no BNB/VIP discount).\n"
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
