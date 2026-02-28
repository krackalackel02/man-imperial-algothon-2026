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

        # Log returns aligned to signals (return[t] = log(price[t]/price[t-1]))
        log_ret = np.log(prices).diff()

        # Shift returns by -1 so features at t predict return at t+1
        fwd_ret = log_ret.shift(-1)

        for asset in self.data.assets:
            # Gather signal columns for this asset (e.g. INSTRUMENT_1_trend4, ...)
            sig_cols = [c for c in signals.columns if c.startswith(asset + "_")]
            if not sig_cols:
                # Fallback: use mean return if no signals
                self._mu[asset] = float(log_ret[asset].dropna().mean())
                continue

            # Build feature / target frame, drop NaN rows
            X = signals[sig_cols]
            y = fwd_ret[asset]
            mask = X.notna().all(axis=1) & y.notna()
            X_valid = X.loc[mask].values
            y_valid = y.loc[mask].values

            if len(y_valid) < 20:
                self._mu[asset] = float(log_ret[asset].dropna().mean())
                continue

            # Train / validation split (time-ordered)
            split = int(len(y_valid) * self.train_frac)
            X_train, y_train = X_valid[:split], y_valid[:split]

            model = LinearRegression()
            model.fit(X_train, y_train)
            self._models[asset] = model

            # Predict expected return using latest available signals
            latest_signals = signals[sig_cols].dropna().iloc[-1:].values
            self._mu[asset] = float(model.predict(latest_signals)[0])

        self._trained = True
        # Weights are set by the PortfolioOptimizer, not the model itself
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
