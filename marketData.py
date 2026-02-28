
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketData:
    """Immutable snapshot of the raw CSV data."""

    prices: pd.DataFrame       # index=date, cols=INSTRUMENT_1..10
    signals: pd.DataFrame      # index=date, cols=INSTRUMENT_i_trendN
    volumes: pd.DataFrame      # index=date, cols=INSTRUMENT_i_vol
    cash_rate: pd.DataFrame    # index=date, cols=1mo..30yr

    @property
    def assets(self) -> list[str]:
        """List of instrument column names from the prices frame."""
        return list(self.prices.columns)

    @property
    def n_assets(self) -> int:
        return len(self.assets)


def load_market_data(data_dir: str | Path = "data/2024-12-31") -> MarketData:
    """Load all four CSVs from *data_dir* into a MarketData container."""
    data_dir = Path(data_dir)

    def _read(name: str) -> pd.DataFrame:
        df = pd.read_csv(data_dir / name, parse_dates=["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        return df

    return MarketData(
        prices=_read("prices.csv"),
        signals=_read("signals.csv"),
        volumes=_read("volumes.csv"),
        cash_rate=_read("cash_rate.csv"),
    )
