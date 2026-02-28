# volumes.csv

## What this file is showing
- Daily traded volume series for the same 10 instruments:
  - `INSTRUMENT_1_vol` ... `INSTRUMENT_10_vol`
- Coverage: `2017-01-03` to `2024-12-31` (`2851` rows).
- No missing values at the CSV level.

## Data behavior notes
- Instruments `1`-`8` each have `839` zero-volume rows (likely non-trading days or closed sessions).
- Instruments `9`-`10` each have only `157` zero-volume rows, suggesting a different trading calendar/market microstructure.
- Zero values should not be blindly treated as missing; they may encode market closures.

## How useful this is for trading
**Usefulness: High for execution/risk, Medium for standalone alpha**

- Important for liquidity filters, turnover constraints, slippage modeling, and capacity control.
- Helps prevent unrealistic fills in backtests.
- On its own, volume is usually a supporting feature rather than a complete signal.
