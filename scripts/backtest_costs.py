"""Cost-aware backtest of every saved best-per-group model, on its holdout only.

Uses one predetermined trading rule (long/short sized by the predicted
sign, one round-trip trade per row, held for exactly one horizon) and one
realistic cost assumption, both fixed before looking at these results.
Trying several cost or threshold variants and keeping whichever looks best
would reintroduce the same data-snooping bias the purged holdout exists to
prevent -- this script computes exactly one scenario per model, once.

Run with: python -m scripts.backtest_costs
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
from src.features import TechnicalIndicatorCalculator
from src.training import PurgedWalkForwardSplit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
INSTRUMENT = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
TEST_SIZE = 0.15
N_SPLITS = 5

# (timeframe, horizon, saved model filename) for each group winner already
# persisted by scripts/run_experiments.py.
GROUP_MODELS = (
    (TimeFrame.FIFTEEN_MINUTES, timedelta(minutes=30), "best_xgboost_15m_03000.joblib"),
    (TimeFrame.ONE_HOUR, timedelta(hours=4), "best_xgboost_1h_40000.joblib"),
    (TimeFrame.FOUR_HOURS, timedelta(hours=24), "best_xgboost_4h_1day,00000.joblib"),
    (TimeFrame.ONE_DAY, timedelta(days=3), "best_random_forest_1d_3days,00000.joblib"),
)

# Binance spot standard taker fee is 0.1% per fill; a round-trip trade
# (enter + exit) pays it twice. This is the non-discounted baseline (no BNB
# fee rebate, no VIP volume tier) -- a conservative cost, not best-case.
ROUND_TRIP_COST = 0.002


def main() -> None:
    repository = ParquetRepository(base_path=DATA_DIR)
    data_manager = DataManager(repository)
    builder = DatasetBuilder(feature_calculator=TechnicalIndicatorCalculator())

    rows = []
    for timeframe, horizon, filename in GROUP_MODELS:
        bars = data_manager.get_bars(
            INSTRUMENT,
            timeframe,
            start=datetime(2017, 1, 1, tzinfo=timezone.utc),
            end=datetime.now(timezone.utc),
        )
        X, y = builder.build(bars, timeframe, horizon)

        embargo = horizon // timeframe.duration
        splitter = PurgedWalkForwardSplit(n_splits=N_SPLITS, embargo=embargo, test_size=TEST_SIZE)
        _, test_pos = splitter.test_split(X.index)
        X_test, y_test = X.iloc[test_pos], y.iloc[test_pos]

        model = joblib.load(MODELS_DIR / filename)
        predictions = model.predict(X_test)

        gross = directional_backtest(y_test, predictions, cost_per_trade=0.0)
        net = directional_backtest(y_test, predictions, cost_per_trade=ROUND_TRIP_COST)

        rows.append(
            {
                "timeframe": timeframe.value,
                "horizon": str(horizon),
                "model": model.model_id,
                "n_trades": gross.n_trades,
                "period": f"{X_test.index[0].date()} -> {X_test.index[-1].date()}",
                "gross_total_%": gross.gross_total_return * 100,
                "net_total_%": net.net_total_return * 100,
                "gross_mean_bp": gross.gross_mean_return * 10_000,
                "net_mean_bp": net.net_mean_return * 10_000,
                "net_win_rate": net.win_rate,
            }
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(
        f"Round-trip cost assumption: {ROUND_TRIP_COST * 100:.2f}% "
        f"(Binance spot taker fee, non-discounted, one entry + one exit per trade)\n"
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
