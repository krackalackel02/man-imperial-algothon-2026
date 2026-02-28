"""
Algothon portfolio pipeline.

    Load  →  Clean  →  Model  →  Train  →  Weights
"""

from __future__ import annotations

from marketData import load_market_data, MarketData
from dataCleaner import DefaultDataCleaner, BaseDataCleaner
from covarianceEstimator import CovarianceEstimator, CovarianceConfig
from model import BaseModel

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

    Returns
    -------
    dict[str, float]
        Final portfolio weights keyed by asset name.
    """
    if model_cls is None:
        raise ValueError("You must pass a concrete model_cls (subclass of BaseModel).")

    # 1. Load raw data
    print(f"[1/5] Loading market data from '{data_dir}' ...")
    raw_data: MarketData = load_market_data(data_dir)
    print(f"       {raw_data.n_assets} assets, "
          f"{len(raw_data.prices)} price rows loaded.")

    # 2. Clean data
    print(f"[2/5] Cleaning data with {cleaner_cls.__name__} ...")
    cleaner = cleaner_cls(raw_data)
    clean_data: MarketData = cleaner.clean_data()
    print(f"       {len(clean_data.prices)} price rows after cleaning.")

    # 3. Estimate covariance
    print("[3/5] Estimating covariance matrix ...")
    cov_estimator = CovarianceEstimator(clean_data, config=cov_config)
    cov_estimator.fit()
    print(f"       shrinkage alpha: {cov_estimator.shrinkage_alpha:.4f}, "
          f"common-liquid rows: {cov_estimator.common_liquid_returns.shape[0]}")

    # 4. Instantiate model
    print(f"[4/5] Building model ({model_cls.__name__}) ...")
    model: BaseModel = model_cls(clean_data, cov_estimator=cov_estimator)

    # 5. Train
    print("[5/5] Training model ...")
    model.train()

    # 5. Output
    weights = model.weights
    print("\n=== Portfolio Weights ===")
    for asset, w in sorted(weights.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {asset:>20s}  {w:+.6f}")
    print(f"\n  Sum of weights: {sum(weights.values()):.6f}")

    return weights


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
