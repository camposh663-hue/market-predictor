"""Download recent historical OHLCV bars and persist them to Parquet.

Run with: python -m scripts.sync_data
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.database import DataManager, ParquetRepository
from src.domain import AssetClass, Instrument, TimeFrame
from src.providers.binance_provider import BinanceProvider
from src.sync import MarketDataSyncService

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY = timedelta(days=180)
INSTRUMENT = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
TIMEFRAMES = (
    TimeFrame.ONE_HOUR,
    TimeFrame.FIFTEEN_MINUTES,
    TimeFrame.FOUR_HOURS,
    TimeFrame.ONE_DAY,
)


def main() -> None:
    provider = BinanceProvider()
    repository = ParquetRepository(base_path=DATA_DIR)
    data_manager = DataManager(repository)
    sync_service = MarketDataSyncService(provider, data_manager)

    end = datetime.now(timezone.utc)
    start = end - HISTORY

    for timeframe in TIMEFRAMES:
        fetched = sync_service.sync_bars(INSTRUMENT, timeframe, start, end)
        latest = data_manager.latest_timestamp(INSTRUMENT, timeframe)
        print(f"Fetched {len(fetched)} new bars for {INSTRUMENT.symbol} ({timeframe.value}).")
        print(f"Latest stored bar: {latest}")


if __name__ == "__main__":
    main()
