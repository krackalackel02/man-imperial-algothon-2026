"""
Covariance estimation from cleaned MarketData.

Refactored from ``dataCorr/dataClean.ipynb``.  Produces a robust
covariance matrix via:

1. Liquidity-aware return masking  (volumes > 0 at both endpoints)
2. MAD-based winsorization of extreme returns
3. EWMA covariance (recent data weighted more)
4. OAS shrinkage toward a diagonal target
5. PSD projection (eigenvalue clipping)

Usage
-----
>>> from covarianceEstimator import CovarianceEstimator
>>> est = CovarianceEstimator(clean_data)
>>> est.fit()
>>> cov = est.covariance_matrix          # daily, (N x N)
>>> corr = est.correlation_matrix        # (N x N)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from marketData import MarketData


# ---------------------------------------------------------------------------
# Pure helper functions (stateless, reusable)
# ---------------------------------------------------------------------------

def robust_winsorize(x: np.ndarray, k: float = 8.0) -> np.ndarray:
    """Column-wise MAD winsorization.  NaNs are preserved."""
    y = x.copy()
    for j in range(y.shape[1]):
        col = y[:, j]
        med = np.nanmedian(col)
        mad = np.nanmedian(np.abs(col - med))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        lo, hi = med - k * sigma, med + k * sigma
        y[:, j] = np.clip(col, lo, hi)
    return y


def ewma_covariance(x: np.ndarray, halflife: float = 60) -> np.ndarray:
    """EWMA covariance on a complete (no-NaN) return matrix *x*."""
    t, _ = x.shape
    lam = math.exp(math.log(0.5) / float(halflife))
    w = lam ** np.arange(t - 1, -1, -1, dtype=float)
    w /= w.sum()
    mu = (w[:, None] * x).sum(axis=0)
    xc = x - mu
    cov = (xc * w[:, None]).T @ xc
    return 0.5 * (cov + cov.T)


def oas_shrinkage_intensity(x: np.ndarray) -> float:
    """Oracle-Approximating Shrinkage intensity from a return matrix."""
    t, p = x.shape
    if t <= 1:
        return 1.0
    s = np.cov(x, rowvar=False, ddof=1)
    tr_s = np.trace(s)
    tr_s2 = np.trace(s @ s)
    num = (1.0 - 2.0 / p) * tr_s2 + tr_s ** 2
    den = (t + 1.0 - 2.0 / p) * (tr_s2 - tr_s ** 2 / p)
    if den <= 0:
        return 1.0
    return float(np.clip(num / den, 0.0, 1.0))


def nearest_psd(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone."""
    m = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, eps, None)
    psd = vecs @ np.diag(vals) @ vecs.T
    return 0.5 * (psd + psd.T)


def covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), 1e-20, None))
    corr = cov / np.outer(d, d)
    return np.clip(corr, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------
@dataclass
class CovarianceConfig:
    """Tunable knobs for the covariance pipeline."""
    winsorize_k: float = 8.0
    ewma_halflife: float = 60.0
    min_common_rows: int = 100
    psd_eps: float = 1e-10
    annualize_factor: float = 252.0


class CovarianceEstimator:
    """
    Computes a robust covariance matrix from cleaned ``MarketData``.

    Parameters
    ----------
    data : MarketData
        Cleaned market data (prices, volumes, etc.).
    config : CovarianceConfig, optional
        Override default hyper-parameters.
    """

    def __init__(
        self,
        data: MarketData,
        config: CovarianceConfig | None = None,
    ) -> None:
        self.data = data
        self.cfg = config or CovarianceConfig()
        self._fitted = False

        # Populated after .fit()
        self._instruments: list[str] = []
        self._log_returns_masked: np.ndarray | None = None
        self._log_returns_clean: np.ndarray | None = None
        self._common_liquid_returns: np.ndarray | None = None
        self._cov_daily: np.ndarray | None = None
        self._cov_annualized: np.ndarray | None = None
        self._corr: np.ndarray | None = None
        self._shrinkage_alpha: float | None = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def fit(self) -> CovarianceEstimator:
        """Run the full covariance estimation pipeline.  Returns *self*."""
        prices = self.data.prices.values.astype(float)
        volumes = self.data.volumes.values.astype(float)
        self._instruments = list(self.data.assets)

        # Align volume columns to price columns
        # (volume cols are INSTRUMENT_i_vol, price cols are INSTRUMENT_i)
        vol_col_order = [f"{c}_vol" for c in self._instruments]
        volumes_aligned = self.data.volumes[vol_col_order].values.astype(float)

        # 1) liquidity-aware mask
        vol_pos = volumes_aligned > 0
        vol_prev = np.vstack([
            np.zeros((1, volumes_aligned.shape[1]), dtype=bool),
            vol_pos[:-1],
        ])
        valid_mask = (vol_pos & vol_prev)[1:]          # skip first row

        # 2) log returns
        log_prices = np.log(prices)
        log_ret = log_prices[1:] - log_prices[:-1]

        ret_masked = log_ret.copy()
        ret_masked[~valid_mask] = np.nan
        self._log_returns_masked = ret_masked

        # 3) winsorize
        ret_clean = robust_winsorize(ret_masked, k=self.cfg.winsorize_k)
        self._log_returns_clean = ret_clean

        # 4) common-liquid panel
        common_rows = np.all(np.isfinite(ret_clean), axis=1)
        ret_common = ret_clean[common_rows]
        if ret_common.shape[0] < self.cfg.min_common_rows:
            raise ValueError(
                f"Only {ret_common.shape[0]} common-liquid rows; "
                f"need at least {self.cfg.min_common_rows} for stable "
                "covariance estimation."
            )
        self._common_liquid_returns = ret_common

        # 5) EWMA covariance
        cov_ewma = ewma_covariance(ret_common, halflife=self.cfg.ewma_halflife)

        # 6) OAS shrinkage
        alpha = oas_shrinkage_intensity(ret_common)
        diag_target = np.diag(np.diag(cov_ewma))
        cov_shrunk = (1.0 - alpha) * cov_ewma + alpha * diag_target
        self._shrinkage_alpha = alpha

        # 7) PSD projection
        self._cov_daily = nearest_psd(cov_shrunk, eps=self.cfg.psd_eps)
        self._cov_annualized = self._cov_daily * self.cfg.annualize_factor
        self._corr = covariance_to_correlation(self._cov_daily)

        self._fitted = True
        return self

    # -----------------------------------------------------------------
    # Properties (available after fit)
    # -----------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call .fit() before accessing results.")

    @property
    def instruments(self) -> list[str]:
        self._check_fitted()
        return self._instruments

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Daily covariance matrix (N x N)."""
        self._check_fitted()
        assert self._cov_daily is not None
        return self._cov_daily

    @property
    def covariance_matrix_annualized(self) -> np.ndarray:
        """Annualised covariance matrix (N x N)."""
        self._check_fitted()
        assert self._cov_annualized is not None
        return self._cov_annualized

    @property
    def correlation_matrix(self) -> np.ndarray:
        """Correlation matrix (N x N)."""
        self._check_fitted()
        assert self._corr is not None
        return self._corr

    @property
    def shrinkage_alpha(self) -> float:
        """OAS shrinkage intensity used."""
        self._check_fitted()
        assert self._shrinkage_alpha is not None
        return self._shrinkage_alpha

    @property
    def common_liquid_returns(self) -> np.ndarray:
        """Clean return panel used for estimation (T' x N)."""
        self._check_fitted()
        assert self._common_liquid_returns is not None
        return self._common_liquid_returns

    @property
    def log_returns_clean(self) -> np.ndarray:
        """Full winsorized return matrix (T-1 x N), NaN where illiquid."""
        self._check_fitted()
        assert self._log_returns_clean is not None
        return self._log_returns_clean
