"""Parquet-backed repository for provider-agnostic funding-rate storage."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from src.domain import FundingRate, Instrument

from .funding_rate_repository import FundingRateRepository

_COLUMNS = ["timestamp", "rate"]
_NUMERIC_DTYPES = {"rate": "float64"}


class ParquetFundingRateRepository(FundingRateRepository):
    """Persist normalized funding rates as Parquet files on the local filesystem.

    Rates are stored one file per instrument, partitioned by asset class and
    symbol, alongside that instrument's OHLCV bar files::

        {base_path}/{asset_class}/{symbol}/funding_rate.parquet

    Each write merges the incoming rates with any rates already on disk,
    resolves duplicate timestamps in favor of the incoming rate, and keeps
    the file sorted chronologically. Pandas and pyarrow are implementation
    details confined to this class; every public method exchanges only
    domain objects with its callers.

    Args:
        base_path: Root directory under which Parquet files are stored.
    """

    def __init__(self, base_path: Path = Path("data")) -> None:
        self._base_path = Path(base_path)

    def write_funding_rates(
        self,
        instrument: Instrument,
        funding_rates: Sequence[FundingRate],
    ) -> None:
        """Merge funding rates into the Parquet file for an instrument.

        Args:
            instrument: Instrument the funding rates belong to.
            funding_rates: Rates to persist. A rate in ``funding_rates``
                replaces any rate already on disk that shares its timestamp.
        """
        path = self._path_for(instrument)
        existing = self._read_dataframe(path)
        incoming = self._rates_to_dataframe(funding_rates)
        combined = (
            pd.concat([existing, incoming], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        self._write_dataframe(combined, path)

    def read_funding_rates(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
    ) -> Sequence[FundingRate]:
        """Retrieve stored funding rates for an instrument within ``[start, end)``.

        Args:
            instrument: Instrument whose funding rates are requested.
            start: Inclusive, timezone-aware UTC start timestamp.
            end: Exclusive, timezone-aware UTC end timestamp.

        Returns:
            Stored funding rates ordered chronologically from oldest to
            newest.
        """
        df = self._read_dataframe(self._path_for(instrument))
        if df.empty:
            return []
        mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
        return self._dataframe_to_rates(df.loc[mask])

    def latest_timestamp(self, instrument: Instrument) -> Optional[datetime]:
        """Return the newest stored settlement timestamp, or ``None`` when empty.

        Args:
            instrument: Instrument to inspect.

        Returns:
            The settlement timestamp of the newest stored funding rate, or
            ``None``.
        """
        df = self._read_dataframe(self._path_for(instrument))
        if df.empty:
            return None
        return df["timestamp"].max().to_pydatetime()

    def _path_for(self, instrument: Instrument) -> Path:
        """Build the Parquet file path for an instrument's funding-rate history."""
        symbol_folder = self._normalize_symbol(instrument.symbol)
        return (
            self._base_path
            / instrument.asset_class.value
            / symbol_folder
            / "funding_rate.parquet"
        )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Strip path-separator characters so the symbol is a safe folder name."""
        return symbol.replace("/", "").replace("\\", "")

    def _read_dataframe(self, path: Path) -> pd.DataFrame:
        """Read a funding-rate file into a DataFrame, or an empty one if absent."""
        if not path.exists():
            return self._empty_dataframe()
        df = pd.read_parquet(path, engine="pyarrow")
        if df.empty:
            return self._empty_dataframe()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def _write_dataframe(self, df: pd.DataFrame, path: Path) -> None:
        """Write a DataFrame to Parquet, creating parent folders as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", index=False)

    @staticmethod
    def _empty_dataframe() -> pd.DataFrame:
        """Build an empty, correctly typed funding-rate DataFrame."""
        df = pd.DataFrame(columns=_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.astype(_NUMERIC_DTYPES)

    @staticmethod
    def _rates_to_dataframe(funding_rates: Sequence[FundingRate]) -> pd.DataFrame:
        """Convert domain funding rates into a DataFrame ready to persist."""
        df = pd.DataFrame(
            {
                "timestamp": [rate.timestamp for rate in funding_rates],
                "rate": [rate.rate for rate in funding_rates],
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.astype(_NUMERIC_DTYPES)

    @staticmethod
    def _dataframe_to_rates(df: pd.DataFrame) -> List[FundingRate]:
        """Convert a DataFrame of funding rates into domain objects, oldest first."""
        return [
            FundingRate(timestamp=row.timestamp.to_pydatetime(), rate=float(row.rate))
            for row in df.sort_values("timestamp").itertuples(index=False)
        ]


__all__ = ["ParquetFundingRateRepository"]
