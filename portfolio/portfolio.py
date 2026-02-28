#!/usr/bin/env python3
"""Build the long-only maximum Sharpe-ratio portfolio from CSV inputs.

Usage:
    python portfolio.py --mu expected_returns.csv --cov cov_final_daily.csv --out portfolio.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _clean_name(name: Optional[str], idx: int) -> str:
    if not name:
        return f"INSTRUMENT {idx}"
    raw = name.strip()
    if not raw:
        return f"INSTRUMENT {idx}"
    match = re.fullmatch(r"INSTRUMENT[_\s-]*(\d+)", raw, flags=re.IGNORECASE)
    if match:
        return f"INSTRUMENT {match.group(1)}"
    return raw


def read_expected_returns(path: Path) -> Tuple[np.ndarray, List[Optional[str]]]:
    rows: List[List[str]] = []
    with path.open("r", newline="") as handle:
        for row in csv.reader(handle):
            stripped = [cell.strip() for cell in row]
            if any(cell != "" for cell in stripped):
                rows.append(stripped)

    values: List[float] = []
    names: List[Optional[str]] = []

    for row in rows:
        numeric_cells = [cell for cell in row if _is_number(cell)]
        if not numeric_cells:
            continue
        value = float(numeric_cells[-1])
        name = None
        if len(row) >= 2 and not _is_number(row[0]):
            name = row[0]
        values.append(value)
        names.append(name)

    if not values:
        raise ValueError(f"No numeric expected returns found in {path}")

    mu = np.asarray(values, dtype=float)
    return mu, names


def read_covariance(path: Path) -> Tuple[np.ndarray, Optional[List[str]]]:
    rows: List[List[str]] = []
    with path.open("r", newline="") as handle:
        for row in csv.reader(handle):
            stripped = [cell.strip() for cell in row]
            if any(cell != "" for cell in stripped):
                rows.append(stripped)

    if not rows:
        raise ValueError(f"Empty covariance file: {path}")

    first_row = rows[0]
    has_labeled_header = (
        len(first_row) > 1
        and not _is_number(first_row[0])
        and any(not _is_number(cell) for cell in first_row[1:])
    )

    if has_labeled_header:
        col_names = [cell for cell in first_row[1:]]
        matrix_rows = rows[1:]
        data: List[List[float]] = []
        row_names: List[str] = []

        for row in matrix_rows:
            if len(row) < len(col_names) + 1:
                raise ValueError("Covariance row has fewer columns than expected.")
            row_names.append(row[0])
            data.append([float(cell) for cell in row[1 : 1 + len(col_names)]])

        cov = np.asarray(data, dtype=float)
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance matrix must be square.")
        if len(row_names) == len(col_names) and row_names != col_names:
            # Keep column order as canonical if labels differ.
            pass
        return cov, col_names

    numeric_data: List[List[float]] = []
    for row in rows:
        numeric_cells = [cell for cell in row if _is_number(cell)]
        if not numeric_cells:
            continue
        numeric_data.append([float(cell) for cell in numeric_cells])

    cov = np.asarray(numeric_data, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be NxN.")
    return cov, None


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project v onto {w >= 0, sum(w)=1}."""
    if v.ndim != 1:
        raise ValueError("Simplex projection expects a 1D array.")
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


def nearest_psd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    sym = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, eps, None)
    psd = vecs @ np.diag(vals) @ vecs.T
    return 0.5 * (psd + psd.T)


def sharpe_ratio(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    ret = float(mu @ w)
    var = float(w @ cov @ w)
    if var <= 0.0:
        return -np.inf
    return ret / np.sqrt(var)


def sharpe_gradient(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov_w = cov @ w
    var = float(w @ cov_w)
    var = max(var, 1e-20)
    den = np.sqrt(var)
    ret = float(mu @ w)
    return mu / den - (ret / (var * den)) * cov_w


def max_sharpe_long_only(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    restarts: int = 24,
    max_iter: int = 4000,
    tol: float = 1e-12,
    seed: int = 42,
) -> np.ndarray:
    """Return long-only weights maximizing Sharpe ratio with sum(weights)=1."""
    mu = np.asarray(expected_returns, dtype=float).reshape(-1)
    cov = np.asarray(covariance, dtype=float)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be NxN")
    if mu.size != cov.shape[0]:
        raise ValueError("expected_returns length must match covariance dimensions")

    cov = nearest_psd(cov)
    n = mu.size
    rng = np.random.default_rng(seed)

    starts = [np.full(n, 1.0 / n)]
    starts.extend(np.eye(n))
    for _ in range(max(0, restarts - len(starts))):
        starts.append(rng.dirichlet(np.ones(n)))

    best_w = starts[0]
    best_sr = sharpe_ratio(best_w, mu, cov)

    for w0 in starts:
        w = project_to_simplex(w0)
        sr = sharpe_ratio(w, mu, cov)
        step = 0.2

        for _ in range(max_iter):
            grad = sharpe_gradient(w, mu, cov)
            improved = False
            step_try = step
            candidate = w
            candidate_sr = sr

            for _ in range(30):
                trial = project_to_simplex(w + step_try * grad)
                trial_sr = sharpe_ratio(trial, mu, cov)
                if trial_sr > sr + 1e-14:
                    candidate = trial
                    candidate_sr = trial_sr
                    improved = True
                    break
                step_try *= 0.5

            if not improved:
                break

            if abs(candidate_sr - sr) < tol:
                w = candidate
                sr = candidate_sr
                break

            w = candidate
            sr = candidate_sr
            step = min(step_try * 1.25, 5.0)

        if sr > best_sr:
            best_sr = sr
            best_w = w.copy()

    return project_to_simplex(best_w)


def write_portfolio_csv(path: Path, weights: np.ndarray, names: Optional[List[str]]) -> None:
    with path.open("w", newline="") as handle:
        for i, w in enumerate(weights, start=1):
            name = names[i - 1] if names is not None and i - 1 < len(names) else None
            handle.write(f"{_clean_name(name, i)}: {w:.12f}\n")


def align_names(
    mu: np.ndarray,
    mu_names: List[Optional[str]],
    cov_names: Optional[List[str]],
) -> Tuple[np.ndarray, List[str]]:
    n = mu.size
    if cov_names is None:
        out_names = [_clean_name(name, i + 1) for i, name in enumerate(mu_names)]
        return mu, out_names

    if len(cov_names) != n:
        raise ValueError("Expected returns length does not match covariance labels.")

    mu_has_names = all(name is not None and name.strip() != "" for name in mu_names)
    if not mu_has_names:
        return mu, [_clean_name(name, i + 1) for i, name in enumerate(cov_names)]

    mu_name_to_idx = {str(name).strip(): i for i, name in enumerate(mu_names)}
    if all(name in mu_name_to_idx for name in cov_names):
        mu_aligned = np.asarray([mu[mu_name_to_idx[name]] for name in cov_names], dtype=float)
        return mu_aligned, [_clean_name(name, i + 1) for i, name in enumerate(cov_names)]

    # Fallback to positional alignment if names are present but inconsistent.
    return mu, [_clean_name(name, i + 1) for i, name in enumerate(cov_names)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-only maximum Sharpe portfolio optimizer.")
    parser.add_argument("--mu", required=True, type=Path, help="Path to Nx1 expected returns CSV.")
    parser.add_argument("--cov", required=True, type=Path, help="Path to NxN covariance CSV.")
    parser.add_argument(
        "--out",
        default=Path("portfolio.csv"),
        type=Path,
        help="Output portfolio CSV path (default: portfolio.csv).",
    )
    parser.add_argument(
        "--restarts",
        default=24,
        type=int,
        help="Number of optimization restarts (default: 24).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mu, mu_names = read_expected_returns(args.mu)
    cov, cov_names = read_covariance(args.cov)

    if mu.size != cov.shape[0]:
        raise ValueError(
            f"Dimension mismatch: expected returns length {mu.size}, covariance shape {cov.shape}"
        )

    mu_aligned, out_names = align_names(mu, mu_names, cov_names)
    weights = max_sharpe_long_only(mu_aligned, cov, restarts=max(1, args.restarts))
    write_portfolio_csv(args.out, weights, out_names)

    print(f"Wrote portfolio weights to {args.out.resolve()}")


if __name__ == "__main__":
    main()
