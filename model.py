"""
Base model class for the Algothon portfolio pipeline.

Subclass BaseModel and implement:
  - expected_returns()  -> dict[str, float]   (asset -> expected return)
  - covariance_matrix() -> np.ndarray          (N x N covariance matrix)

The solver will consume these two outputs to compute optimal weights.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from marketData import MarketData
import numpy as np
import pandas as pd



# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
class BaseModel(ABC):
    """
    Abstract base class that every alpha / risk model must implement.

    Typical usage
    -------------
    >>> model = MyModel(data)
    >>> mu = model.expected_returns()        # dict[str, float]
    >>> cov = model.covariance_matrix()      # np.ndarray (N x N)
    >>> # feed mu, cov into the portfolio solver
    """

    def __init__(self, data: MarketData) -> None:
        self.data = data

    # -- required overrides --------------------------------------------------

    @abstractmethod
    def expected_returns(self) -> dict[str, float]:
        """
        Return a mapping  {asset_name: expected_return}  for the next horizon.

        The dictionary must contain one entry per asset in
        ``self.data.assets``.
        """
        ...

    @abstractmethod
    def covariance_matrix(self) -> np.ndarray:
        """
        Return an (N x N) numpy covariance matrix ordered consistently with
        ``self.data.assets``.
        """
        ...

    # -- convenience helpers -------------------------------------------------

    def expected_returns_array(self) -> np.ndarray:
        """Expected returns as an (N,) numpy array aligned with self.data.assets."""
        mu = self.expected_returns()
        return np.array([mu[a] for a in self.data.assets])

    def risk_free_rate(self, tenor: str = "3mo") -> float:
        """Latest available risk-free rate for the given tenor (annualised)."""
        col = tenor
        rates = self.data.cash_rate[col].dropna()
        if rates.empty:
            return 0.0
        return float(rates.iloc[-1]) / 100.0   # CSV values are in %

    def validate(self) -> None:
        """Basic sanity checks on model outputs."""
        mu = self.expected_returns()
        cov = self.covariance_matrix()

        # --- expected returns ---
        assert set(mu.keys()) == set(self.data.assets), (
            "expected_returns keys must match data.assets"
        )

        # --- covariance matrix ---
        n = self.data.n_assets
        assert cov.shape == (n, n), f"Covariance must be ({n}, {n})"
        assert np.allclose(cov, cov.T), "Covariance matrix must be symmetric"
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-8), "Covariance matrix must be PSD"
