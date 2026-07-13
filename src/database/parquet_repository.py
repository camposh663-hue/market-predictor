"""Parquet-backed repository for provider-agnostic market-bar storage."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from src.domain import Instrument, MarketBar, TimeFrame

from .base_repository import BarRepository

_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_NUMERIC_DTYPES = {
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}


class ParquetRepository(BarRepository):
    """Persist normalized OHLCV bars as Parquet files on the local filesystem.

    Bars are stored one file per instrument and timeframe, partitioned by
    asset class and symbol::

        {base_path}/{asset_class}/{symbol}/{timeframe}.parquet

    Each write merges the incoming bars with any bars already on disk,
    resolves duplicate timestamps in favor of the incoming bar, and keeps
    the file sorted chronologically. Pandas and pyarrow are implementation
    details confined to this class; every public method exchanges only
    domain objects with its callers.

    Args:
        base_path: Root directory under which Parquet files are stored.
    """

    def __init__(self, base_path: Path = Path("data")) -> None:
        self._base_path = Path(base_path)

    def write_bars(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
        bars: Sequence[MarketBar],
    ) -> None:
        """Merge bars into the Parquet file for an instrument and timeframe.

        Args:
            instrument: Instrument the bars belong to.
            timeframe: Aggregation interval of the bars.
            bars: Bars to persist. A bar in ``bars`` replaces any bar already
                on disk that shares its timestamp.
        """
        path = self._path_for(instrument, timeframe)
        existing = self._read_dataframe(path)
        incoming = self._bars_to_dataframe(bars)
        combined = (
            pd.concat([existing, incoming], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        self._write_dataframe(combined, path)

    def read_bars(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> Sequence[MarketBar]:
        """Retrieve stored bars for an instrument within ``[start, end)``.

        Args:
            instrument: Instrument whose bars are requested.
            timeframe: Aggregation interval of the bars.
            start: Inclusive, timezone-aware UTC start timestamp.
            end: Exclusive, timezone-aware UTC end timestamp.

        Returns:
            Stored bars ordered chronologically from oldest to newest.
        """
        df = self._read_dataframe(self._path_for(instrument, timeframe))
        if df.empty:
            return []
        mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
        return self._dataframe_to_bars(df.loc[mask])

    def latest_timestamp(
        self,
        instrument: Instrument,
        timeframe: TimeFrame,
    ) -> Optional[datetime]:
        """Return the newest stored timestamp, or ``None`` when empty.

        Args:
            instrument: Instrument to inspect.
            timeframe: Aggregation interval to inspect.

        Returns:
            The opening timestamp of the newest stored bar, or ``None``.
        """
        df = self._read_dataframe(self._path_for(instrument, timeframe))
        if df.empty:
            return None
        return df["timestamp"].max().to_pydatetime()

    def _path_for(self, instrument: Instrument, timeframe: TimeFrame) -> Path:
        """Build the Parquet file path for an instrument and timeframe."""
        symbol_folder = self._normalize_symbol(instrument.symbol)
        return (
            self._base_path
            / instrument.asset_class.value
            / symbol_folder
            / f"{timeframe.value}.parquet"
        )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Strip path-separator characters so the symbol is a safe folder name.

        Args:
            symbol: Instrument symbol, e.g. ``"BTC/USDT"`` or ``"AAPL"``.

        Returns:
            A symbol safe to use as a single filesystem path segment.
        """
        return symbol.replace("/", "").replace("\\", "")

    def _read_dataframe(self, path: Path) -> pd.DataFrame:
        """Read a bars file into a DataFrame, or an empty one if absent."""
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
        """Build an empty, correctly typed bars DataFrame."""
        df = pd.DataFrame(columns=_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.astype(_NUMERIC_DTYPES)

    @staticmethod
    def _bars_to_dataframe(bars: Sequence[MarketBar]) -> pd.DataFrame:
        """Convert domain bars into a DataFrame ready to persist."""
        df = pd.DataFrame(
            {
                "timestamp": [bar.timestamp for bar in bars],
                "open": [bar.open for bar in bars],
                "high": [bar.high for bar in bars],
                "low": [bar.low for bar in bars],
                "close": [bar.close for bar in bars],
                "volume": [bar.volume for bar in bars],
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.astype(_NUMERIC_DTYPES)

    @staticmethod
    def _dataframe_to_bars(df: pd.DataFrame) -> List[MarketBar]:
        """Convert a DataFrame of bars into domain objects, oldest first."""
        bars: List[MarketBar] = []
        for row in df.sort_values("timestamp").itertuples(index=False):
            volume = None if pd.isna(row.volume) else float(row.volume)
            bars.append(
                MarketBar(
                    timestamp=row.timestamp.to_pydatetime(),
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=volume,
                )
            )
        return bars


__all__ = ["ParquetRepository"]
