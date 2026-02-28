"""
Linear regression model for the Algothon pipeline.

For each instrument, fits a linear regression:

    next-day log-return ~ trend4 + trend8 + trend16 + trend32

using the provided signals as features.  The fitted prediction is used
as the expected return for each asset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from marketData import MarketData
from covarianceEstimator import CovarianceEstimator
from model import BaseModel


class LinearRegressionModel(BaseModel):
    """
    Per-instrument linear regression of next-day log-returns on trend signals.

    Parameters
    ----------
    data : MarketData
        Cleaned market data.
    cov_estimator : CovarianceEstimator | None
        Pre-fitted covariance estimator (used by the pipeline).
    train_frac : float
        Fraction of history used for fitting (rest = validation).
    """

    def __init__(
        self,
        data: MarketData,
        cov_estimator: CovarianceEstimator | None = None,
        train_frac: float = 0.8,
    ) -> None:
        super().__init__(data, cov_estimator=cov_estimator)
        self.train_frac = train_frac
        self._models: dict[str, LinearRegression] = {}
        self._mu: dict[str, float] = {}

    # -----------------------------------------------------------------

    def train(self) -> None:
        prices = self.data.prices
        signals = self.data.signals
        log_ret = np.log(prices).diff()
        fwd_ret = log_ret.shift(-1)

        self._train_mse: dict[str, float] = {}
        self._val_mse: dict[str, float] = {}
        self._scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for asset in self.data.assets:
            sig_cols = [c for c in signals.columns if c.startswith(asset + "_")]
            if not sig_cols:
                self._mu[asset] = float(log_ret[asset].dropna().mean())
                self._train_mse[asset] = float('nan')
                self._val_mse[asset] = float('nan')
                continue

            X = signals[sig_cols]
            y = fwd_ret[asset]
            mask = X.notna().all(axis=1) & y.notna()
            X_valid = X.loc[mask].values
            y_valid = y.loc[mask].values

            if len(y_valid) < 20:
                self._mu[asset] = float(log_ret[asset].dropna().mean())
                self._train_mse[asset] = float('nan')
                self._val_mse[asset] = float('nan')
                continue

            # Standardize features (z-score)
            mean = X_valid.mean(axis=0)
            std = X_valid.std(axis=0) + 1e-12
            X_std = (X_valid - mean) / std
            self._scalers[asset] = (mean, std)

            # Train/validation split (time-ordered)
            split = int(len(y_valid) * self.train_frac)
            X_train, y_train = X_std[:split], y_valid[:split]
            X_val, y_val = X_std[split:], y_valid[split:]

            model = LinearRegression()
            model.fit(X_train, y_train)
            self._models[asset] = model

            # Predict expected return using latest available signals (standardized)
            latest_signals = signals[sig_cols].dropna().iloc[-1:].values
            latest_std = (latest_signals - mean) / std
            self._mu[asset] = float(model.predict(latest_std)[0])

            # Compute train/val MSE
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val) if len(y_val) > 0 else np.array([])
            self._train_mse[asset] = float(np.mean((y_train - y_train_pred) ** 2))
            self._val_mse[asset] = float(np.mean((y_val - y_val_pred) ** 2)) if len(y_val) > 0 else float('nan')

        self._trained = True
        self._weights = {a: 1.0 / self.data.n_assets for a in self.data.assets}

    # -----------------------------------------------------------------

    def expected_returns(self) -> dict[str, float]:
        if not self._trained:
            raise RuntimeError("Call .train() first.")
        return self._mu

    def covariance_matrix(self) -> np.ndarray:
        if self.cov_estimator is not None:
            return self.cov_estimator.covariance_matrix
        return self.data.prices.pct_change().dropna().cov().values

    # -----------------------------------------------------------------
    @property
    def mse_per_asset(self) -> dict[str, float]:
        """In-sample (train) mean squared error for each asset's regression."""
        return getattr(self, '_train_mse', {})

    @property
    def val_mse_per_asset(self) -> dict[str, float]:
        """Validation mean squared error for each asset's regression."""
        return getattr(self, '_val_mse', {})

    @property
    def avg_mse(self) -> float:
        """Average in-sample (train) MSE across all assets."""
        mses = [v for v in self.mse_per_asset.values() if np.isfinite(v)]
        return float(np.mean(mses)) if mses else float('nan')

    @property
    def avg_val_mse(self) -> float:
        """Average validation MSE across all assets."""
        mses = [v for v in self.val_mse_per_asset.values() if np.isfinite(v)]
        return float(np.mean(mses)) if mses else float('nan')
