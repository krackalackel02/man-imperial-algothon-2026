# cash_rate.csv

## What this file is showing
- Daily USD yield-curve style rates by tenor: `1mo`, `1.5month`, `2mo`, `3mo`, `4mo`, `6mo`, `1yr`, `2yr`, `3yr`, `5yr`, `7yr`, `10yr`, `20yr`, `30yr`.
- Coverage: `1990-01-02` to `2024-12-31` (`8757` rows).
- It is macro/rates context data, not directly instrument prices.

## Data quality notes
- `1.5month` is fully empty (`0/8757` populated).
- `4mo` is very sparse (`550/8757` populated).
- `2mo` is partially populated (`1552/8757`).
- `20yr` and `30yr` have long historical gaps (early-period missingness).

## How useful this is for trading
**Usefulness: Medium-High**

- Strong for financing assumptions, carry/roll features, discounting, and macro regime features (e.g., curve slope/inversion).
- Weak as a standalone alpha source for short-horizon directional trading.
- Best used as a conditioning/context feature alongside prices and signals.
