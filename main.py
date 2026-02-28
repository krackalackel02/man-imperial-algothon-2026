"""
Algothon portfolio pipeline.

    Load  →  Clean  →  Covariance  →  Model  →  Optimize  →  Weights
"""

from __future__ import annotations

from pathlib import Path

from marketData import load_market_data, MarketData
from dataCleaner import DefaultDataCleaner, BaseDataCleaner
from covarianceEstimator import CovarianceEstimator, CovarianceConfig
from model import BaseModel
from portfolioOptimizer import PortfolioOptimizer, PortfolioConfig

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    data_dir: str = "data/2024-12-31",
    cleaner_cls: type[BaseDataCleaner] = DefaultDataCleaner,
    model_cls: type[BaseModel] | None = None,
    cov_config: CovarianceConfig | None = None,
    portfolio_config: PortfolioConfig | None = None,
    output_csv: str | Path | None = "portfolio.csv",
) -> dict[str, float]:
    """
    Execute the full pipeline and return portfolio weights.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing the CSV files.
    cleaner_cls : type[BaseDataCleaner]
        Data cleaner class to use (default: ``DefaultDataCleaner``).
    model_cls : type[BaseModel]
        Model class to instantiate and train.  Must be a concrete
        subclass of ``BaseModel``.
    cov_config : CovarianceConfig, optional
        Override covariance estimation hyper-parameters.
    portfolio_config : PortfolioConfig, optional
        Override portfolio optimiser hyper-parameters.
    output_csv : str | Path | None
        Path to write the final portfolio CSV.  ``None`` to skip.

    Returns
    -------
    dict[str, float]
        Final portfolio weights keyed by asset name.
    """
    if model_cls is None:
        raise ValueError("You must pass a concrete model_cls (subclass of BaseModel).")

    # 1. Load raw data
    print(f"[1/6] Loading market data from '{data_dir}' ...")
    raw_data: MarketData = load_market_data(data_dir)
    print(f"       {raw_data.n_assets} assets, "
          f"{len(raw_data.prices)} price rows loaded.")

    # 2. Clean data
    print(f"[2/6] Cleaning data with {cleaner_cls.__name__} ...")
    cleaner = cleaner_cls(raw_data)
    clean_data: MarketData = cleaner.clean_data()
    print(f"       {len(clean_data.prices)} price rows after cleaning.")

    # 3. Estimate covariance
    print("[3/6] Estimating covariance matrix ...")
    cov_estimator = CovarianceEstimator(clean_data, config=cov_config)
    cov_estimator.fit()
    print(f"       shrinkage alpha: {cov_estimator.shrinkage_alpha:.4f}, "
          f"common-liquid rows: {cov_estimator.common_liquid_returns.shape[0]}")

    # 4. Instantiate & train model (produces expected returns)
    print(f"[4/6] Building & training model ({model_cls.__name__}) ...")
    model: BaseModel = model_cls(clean_data, cov_estimator=cov_estimator)
    model.train()

    # 5. Optimize portfolio
    print("[5/6] Optimizing portfolio (max-Sharpe, long-only) ...")
    mu = model.expected_returns()
    cov = cov_estimator.covariance_matrix
    assets = clean_data.assets

    optimizer = PortfolioOptimizer(mu, cov, assets, config=portfolio_config)
    optimizer.solve()

    # 6. Output
    print("[6/6] Done.\n")
    print(optimizer.summary())

    if output_csv is not None:
        out_path = optimizer.to_csv(output_csv)
        print(f"\n  Wrote portfolio CSV to {out_path}")

    return optimizer.weights


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Replace `None` below with your concrete model class, e.g.:
    #
    #   from my_model import MyModel
    #   weights = run_pipeline(model_cls=MyModel)
    # -----------------------------------------------------------------
    print("Pipeline ready.  Provide a concrete BaseModel subclass to run.")
    print("Example:")
    print("  from main import run_pipeline")
    print("  from my_model import MyModel")
    print("  weights = run_pipeline(model_cls=MyModel)")
