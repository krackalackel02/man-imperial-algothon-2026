# signals.csv

## What this file is showing
- Precomputed trend features for each instrument:
  - `trend4`, `trend8`, `trend16`, `trend32`
  - for `INSTRUMENT_1` ... `INSTRUMENT_10` (total `40` signal columns).
- Coverage: `2017-01-03` to `2024-12-31` (`2851` rows).

## Data quality notes
- Warm-up NaNs are expected:
  - each `trend4`: `16` missing rows
  - each `trend8`: `32` missing rows
  - each `trend16`: `64` missing rows
  - each `trend32`: `128` missing rows
- After warm-up, columns are broadly populated through the sample.
- Construction formula is not documented here, so exact interpretation (return horizon, smoothing, normalization) is uncertain.

## How useful this is for trading
**Usefulness: High for prototyping, Medium-High for production**

- Very useful for quickly bootstrapping trend-following models and reducing feature-engineering time.
- Good candidate input for cross-sectional ranking or regime-aware models.
- Before production use, validate there is no look-ahead leakage and confirm signal timing assumptions.
