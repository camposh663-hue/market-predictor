"""Backfill older historical OHLCV bars that a forward-only sync won't fetch.

MarketDataSyncService only ever resumes forward from the latest stored bar
(see src/sync/market_data_sync_service.py) -- extending history further into
the past needs a direct provider -> DataManager write instead. Run with:
    python -m scripts.backfill_data
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.database import DataManager, ParquetRepository
from src.domain import AssetClass, Instrument, TimeFrame
from src.providers.binance_provider import BinanceProvider

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INSTRUMENT = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
TIMEFRAMES = (
    TimeFrame.ONE_HOUR,
    TimeFrame.FIFTEEN_MINUTES,
    TimeFrame.FOUR_HOURS,
    TimeFrame.ONE_DAY,
)
BACKFILL_START = datetime(2017, 8, 17, tzinfo=timezone.utc)  # BTC/USDT listing on Binance


def main() -> None:
    provider = BinanceProvider()
    repository = ParquetRepository(base_path=DATA_DIR)
    data_manager = DataManager(repository)
    now = datetime.now(timezone.utc)

    for timeframe in TIMEFRAMES:
        existing = data_manager.get_bars(INSTRUMENT, timeframe, BACKFILL_START, now)
        backfill_end = min(bar.timestamp for bar in existing) if existing else now

        if backfill_end <= BACKFILL_START:
            print(f"{timeframe.value}: nothing to backfill; already reaches the listing date.")
            continue

        bars = provider.get_historical_bars(INSTRUMENT, timeframe, BACKFILL_START, backfill_end)
        if not bars:
            print(f"{timeframe.value}: Binance returned no bars for the backfill range.")
            continue

        data_manager.store_bars(INSTRUMENT, timeframe, bars)
        print(f"Backfilled {len(bars)} bars for {INSTRUMENT.symbol} ({timeframe.value}).")
        print(f"Range: {bars[0].timestamp} -> {bars[-1].timestamp}")


if __name__ == "__main__":
    main()
