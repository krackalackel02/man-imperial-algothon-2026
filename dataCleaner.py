"""
Data cleaning pipeline for the Algothon portfolio.

Subclass BaseDataCleaner and override any of:
  - clean_prices()
  - clean_signals()
  - clean_volumes()
  - clean_cash_rate()
  - clean_data()        (orchestrator – calls the four above by default)

Usage
-----
>>> data = load_market_data()
>>> cleaner = DefaultDataCleaner(data)
>>> clean = cleaner.clean_data()   # returns a new MarketData
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from marketData import MarketData


# ---------------------------------------------------------------------------
# Base cleaner
# ---------------------------------------------------------------------------
class BaseDataCleaner(ABC):
    """
    Abstract base class for data cleaning.

    Override individual ``clean_*`` methods for fine-grained control,
    or override ``clean_data`` to replace the whole pipeline.
    """

    def __init__(self, data: MarketData) -> None:
        self.data = data

    # -- per-frame hooks (override as needed) --------------------------------

    @abstractmethod
    def clean_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy of the prices frame."""
        ...

    @abstractmethod
    def clean_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy of the signals frame."""
        ...

    @abstractmethod
    def clean_volumes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy of the volumes frame."""
        ...

    @abstractmethod
    def clean_cash_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy of the cash-rate frame."""
        ...

    # -- orchestrator --------------------------------------------------------

    def clean_data(self) -> MarketData:
        """
        Run the full cleaning pipeline and return a new ``MarketData``.

        Override this to change the orchestration logic itself.
        """
        return MarketData(
            prices=self.clean_prices(self.data.prices.copy()),
            signals=self.clean_signals(self.data.signals.copy()),
            volumes=self.clean_volumes(self.data.volumes.copy()),
            cash_rate=self.clean_cash_rate(self.data.cash_rate.copy()),
        )


# ---------------------------------------------------------------------------
# Default cleaner (sensible defaults – override to customise)
# ---------------------------------------------------------------------------
class DefaultDataCleaner(BaseDataCleaner):
    """
    A ready-to-use cleaner with common-sense defaults:

    * Drop fully-empty rows/columns.
    * Forward-fill then backward-fill NaNs in prices & cash rates.
    * Fill remaining signal / volume NaNs with 0.
    * Remove duplicate index entries (keep last).
    """

    def __init__(self, data: MarketData, ffill_limit: int | None = 5) -> None:
        super().__init__(data)
        self.ffill_limit = ffill_limit

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _drop_empty(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows and columns that are entirely NaN."""
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")
        return df

    @staticmethod
    def _dedup_index(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate index entries, keeping the last occurrence."""
        return df[~df.index.duplicated(keep="last")]

    # -- per-frame implementations -------------------------------------------

    def clean_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._dedup_index(df)
        df = self._drop_empty(df)
        df = df.ffill(limit=self.ffill_limit).bfill(limit=self.ffill_limit)
        return df

    def clean_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._dedup_index(df)
        df = self._drop_empty(df)
        df = df.fillna(0.0)
        return df

    def clean_volumes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._dedup_index(df)
        df = self._drop_empty(df)
        df = df.fillna(0.0)
        return df

    def clean_cash_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._dedup_index(df)
        df = self._drop_empty(df)
        df = df.ffill(limit=self.ffill_limit).bfill(limit=self.ffill_limit)
        return df