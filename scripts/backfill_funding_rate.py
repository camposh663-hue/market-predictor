"""Backfill the full historical funding-rate settlements for BTC/USDT perpetual.

Mirrors scripts/backfill_data.py's direct provider -> repository write (no
sync service involved, since this is a one-off historical load rather than
an incremental forward-only sync). Run with:
    python -m scripts.backfill_funding_rate
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.database import ParquetFundingRateRepository
from src.domain import AssetClass, Instrument
from src.providers.binance_funding_rate_provider import BinanceFundingRateProvider

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INSTRUMENT = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO)
BACKFILL_START = datetime(2019, 9, 8, tzinfo=timezone.utc)  # BTCUSDT perpetual launch


def main() -> None:
    provider = BinanceFundingRateProvider()
    repository = ParquetFundingRateRepository(base_path=DATA_DIR)
    now = datetime.now(timezone.utc)

    existing = repository.read_funding_rates(INSTRUMENT, BACKFILL_START, now)
    backfill_end = min(rate.timestamp for rate in existing) if existing else now

    if backfill_end <= BACKFILL_START:
        print("Nothing to backfill; already reaches the perpetual's launch date.")
        return

    rates = provider.get_historical_funding_rates(INSTRUMENT, BACKFILL_START, backfill_end)
    if not rates:
        print("Binance returned no funding-rate records for the backfill range.")
        return

    repository.write_funding_rates(INSTRUMENT, rates)
    print(f"Backfilled {len(rates)} funding-rate settlements for {INSTRUMENT.symbol}.")
    print(f"Range: {rates[0].timestamp} -> {rates[-1].timestamp}")


if __name__ == "__main__":
    main()
