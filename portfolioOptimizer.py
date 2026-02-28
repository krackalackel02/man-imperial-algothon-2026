"""
Portfolio optimizer for the Algothon pipeline.

Refactored from ``portfolio/portfolio.py``.  Takes expected returns
(from a model) and a covariance matrix (from CovarianceEstimator)
and produces optimal long-only weights via max-Sharpe projected
gradient ascent.

Usage
-----
>>> from portfolioOptimizer import PortfolioOptimizer
>>> opt = PortfolioOptimizer(expected_returns, covariance, assets)
>>> opt.solve()
>>> weights = opt.weights           # dict[str, float]
>>> opt.to_csv("portfolio.csv")     # export
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class PortfolioConfig:
    """Tunable knobs for the optimizer."""
    long_only: bool = True
    restarts: int = 24
    max_iter: int = 4000
    tol: float = 1e-12
    seed: int = 42
    psd_eps: float = 1e-10


# ---------------------------------------------------------------------------
# Math helpers (from portfolio/portfolio.py)
# ---------------------------------------------------------------------------

def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project *v* onto the probability simplex {w >= 0, sum(w) = 1}."""
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full(n, 1.0 / n)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def _nearest_psd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    sym = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, eps, None)
    psd = vecs @ np.diag(vals) @ vecs.T
    return 0.5 * (psd + psd.T)


def _sharpe_ratio(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    ret = float(mu @ w)
    var = float(w @ cov @ w)
    if var <= 0.0:
        return -np.inf
    return ret / np.sqrt(var)


def _sharpe_gradient(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov_w = cov @ w
    var = max(float(w @ cov_w), 1e-20)
    den = np.sqrt(var)
    ret = float(mu @ w)
    return mu / den - (ret / (var * den)) * cov_w


def _max_sharpe_long_only(
    mu: np.ndarray,
    cov: np.ndarray,
    cfg: PortfolioConfig,
) -> np.ndarray:
    """Long-only max-Sharpe via projected gradient ascent with restarts."""
    cov = _nearest_psd(cov, eps=cfg.psd_eps)
    n = mu.size
    rng = np.random.default_rng(cfg.seed)

    starts: list[np.ndarray] = [np.full(n, 1.0 / n)]
    starts.extend(np.eye(n))
    for _ in range(max(0, cfg.restarts - len(starts))):
        starts.append(rng.dirichlet(np.ones(n)))

    best_w = starts[0]
    best_sr = _sharpe_ratio(best_w, mu, cov)

    for w0 in starts:
        w = _project_to_simplex(w0)
        sr = _sharpe_ratio(w, mu, cov)
        step = 0.2

        for _ in range(cfg.max_iter):
            grad = _sharpe_gradient(w, mu, cov)
            improved = False
            step_try = step

            for _ in range(30):
                trial = _project_to_simplex(w + step_try * grad)
                trial_sr = _sharpe_ratio(trial, mu, cov)
                if trial_sr > sr + 1e-14:
                    improved = True
                    w_next = trial
                    sr_next = trial_sr
                    break
                step_try *= 0.5

            if not improved:
                break

            if abs(sr_next - sr) < cfg.tol:
                w = w_next
                sr = sr_next
                break

            w = w_next
            sr = sr_next
            step = min(step_try * 1.25, 5.0)

        if sr > best_sr:
            best_sr = sr
            best_w = w.copy()

    return _project_to_simplex(best_w)


# ---------------------------------------------------------------------------
# Name cleaning helper
# ---------------------------------------------------------------------------

def _clean_name(name: str) -> str:
    match = re.fullmatch(r"INSTRUMENT[_\s-]*(\d+)", name.strip(), flags=re.IGNORECASE)
    if match:
        return f"INSTRUMENT {match.group(1)}"
    return name.strip()


# ---------------------------------------------------------------------------
# PortfolioOptimizer
# ---------------------------------------------------------------------------
class PortfolioOptimizer:
    """
    Compute optimal portfolio weights from expected returns and a covariance
    matrix, then optionally export results to CSV.

    Parameters
    ----------
    expected_returns : dict[str, float]
        Mapping ``{asset_name: expected_return}``.
    covariance : np.ndarray
        (N x N) covariance matrix ordered consistently with *assets*.
    assets : list[str]
        Asset names in the same order as rows/cols of *covariance*.
    config : PortfolioConfig, optional
        Override default optimiser hyper-parameters.
    """

    def __init__(
        self,
        expected_returns: dict[str, float],
        covariance: np.ndarray,
        assets: list[str],
        config: PortfolioConfig | None = None,
    ) -> None:
        self.expected_returns = expected_returns
        self.covariance = np.asarray(covariance, dtype=float)
        self.assets = list(assets)
        self.cfg = config or PortfolioConfig()

        self._weights: dict[str, float] | None = None
        self._weights_array: np.ndarray | None = None
        self._sharpe: float | None = None
        self._solved = False

    # -----------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------

    def solve(self) -> PortfolioOptimizer:
        """Run the optimiser.  Returns *self*."""
        mu = np.array([self.expected_returns[a] for a in self.assets], dtype=float)

        w = _max_sharpe_long_only(mu, self.covariance, self.cfg)
        self._weights_array = w
        self._weights = {a: float(w[i]) for i, a in enumerate(self.assets)}
        self._sharpe = _sharpe_ratio(w, mu, _nearest_psd(self.covariance))
        self._solved = True
        return self

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    def _check_solved(self) -> None:
        if not self._solved:
            raise RuntimeError("Call .solve() before accessing results.")

    @property
    def weights(self) -> dict[str, float]:
        """Portfolio weights as ``{asset_name: weight}``."""
        self._check_solved()
        assert self._weights is not None
        return self._weights

    @property
    def weights_array(self) -> np.ndarray:
        """Portfolio weights as an (N,) numpy array aligned with *assets*."""
        self._check_solved()
        assert self._weights_array is not None
        return self._weights_array

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio of the optimal portfolio."""
        self._check_solved()
        assert self._sharpe is not None
        return self._sharpe

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------

    def to_csv(self, path: str | Path) -> Path:
        """Write weights to a CSV file.  Returns the resolved path."""
        self._check_solved()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            for asset in self.assets:
                name = _clean_name(asset)
                w = self._weights[asset]  # type: ignore[index]
                f.write(f"{name}: {w:.12f}\n")
        return path.resolve()

    def summary(self) -> str:
        """Human-readable summary of the portfolio."""
        self._check_solved()
        lines = ["=== Portfolio Weights ==="]
        for asset, w in sorted(self.weights.items(), key=lambda kv: -abs(kv[1])):
            lines.append(f"  {asset:>20s}  {w:+.12f}")
        lines.append(f"\n  Sum of weights : {sum(self.weights.values()):.6f}")
        lines.append(f"  Sharpe ratio   : {self.sharpe_ratio:.6f}")
        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Analytics / Metrics
    # -----------------------------------------------------------------

    @property
    def mean_return(self) -> float:
        """Expected mean return of the optimal portfolio."""
        self._check_solved()
        mu = np.array([self.expected_returns[a] for a in self.assets], dtype=float)
        return float(mu @ self.weights_array)

    @property
    def volatility(self) -> float:
        """Portfolio volatility (standard deviation, annualized if input is annualized)."""
        self._check_solved()
        cov = self.covariance
        w = self.weights_array
        return float(np.sqrt(w @ cov @ w))

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown of the cumulative return series (in-sample)."""
        self._check_solved()
        # Reconstruct in-sample returns
        mu = np.array([self.expected_returns[a] for a in self.assets], dtype=float)
        # Assume returns are daily, use weights as fixed allocation
        # (This is a simplification; for more accuracy, use actual historical returns)
        returns = mu @ self.weights_array
        # Simulate cumulative return
        cum = np.cumprod(1 + np.full(252, returns))  # 1 year
        peak = np.maximum.accumulate(cum)
        drawdown = (cum - peak) / peak
        return float(drawdown.min())

    def metrics(self) -> dict:
        """Return a dictionary of key portfolio metrics."""
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'mean_return': self.mean_return,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
        }

    def metrics_str(self) -> str:
        """Human-readable summary of key metrics."""
        m = self.metrics()
        return (f"Sharpe: {m['sharpe_ratio']:.4f}  "
                f"Mean: {m['mean_return']:.4f}  "
                f"Vol: {m['volatility']:.4f}  "
                f"MaxDD: {m['max_drawdown']:.4f}")
