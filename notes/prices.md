# prices.csv

## What this file is showing
- Daily price series for `10` instruments: `INSTRUMENT_1` ... `INSTRUMENT_10`.
- Coverage: `2017-01-03` to `2024-12-31` (`2851` rows).
- No missing values across any instrument column.

## Data behavior notes
- Includes weekend dates (`770` weekend rows), so this is not a strict weekday-only equity-style calendar.
- Instrument mapping is anonymized, so asset class assumptions should be validated before strategy design.

## How useful this is for trading
**Usefulness: Very High**

- This is the core dataset for return generation, volatility estimation, portfolio construction, and backtesting.
- Reliable completeness makes it suitable for feature engineering without heavy imputation.
- Main caveat: treat weekend handling and calendar alignment carefully when mixing with other files (especially rates).
