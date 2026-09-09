# options-backtest-lab

## Purpose
Historical, analytical, batch-oriented project. Given a strategy definition
(e.g. short strangle on QQQ, weekly, with an adjustment rule), simulate how it
would have performed over the last N years using **synthetic historical
option prices derived from Black-Scholes**, not real historical options data.

This repo does NOT place trades and does NOT connect to any broker. It is
read/compute only.

## Reference material
- Sheldon Natenberg, *Option Volatility and Pricing* (2nd ed.) — conceptual
  grounding for pricing model (Ch. 5, 18), volatility (Ch. 6, 20), risk
  measures/greeks (Ch. 7, 9), volatility spreads & adjustments (Ch. 11),
  volatility skew (Ch. 24).
- `../options-research/notes/` — strategy-to-ETF matching notes, decisions on
  which tickers/parameters to test. Reference this before adding a new
  ticker or strategy; don't duplicate research reasoning into this repo.

## Architecture
1. **Raw zone (S3)** — daily OHLC per ticker (yfinance/Stooq), plus a
   volatility proxy series (rolling realized vol and/or VXN-style index).
2. **Processed zone (Parquet)** — synthetic historical option chains:
   one row per (date, symbol, expiration, strike, right) with Black-Scholes
   theoretical value and full greeks, generated with NumPy + Polars.
3. **Backtest engine** — walks the processed zone day-by-day, applies a
   strategy + adjustment rule, produces a results table (P&L, Sharpe, max
   drawdown, win rate) per parameter set. Sequential by nature: tomorrow's
   position depends on today's.
4. **Analysis** — DuckDB over the results Parquet, plus Jupyter notebooks in
   `notebooks/`, for comparison across strategies and instruments.

## Tech stack
- Python 3.11+
- **NumPy** — pricing core (pure, vectorized, no I/O)
- **Polars** — chain generation and dataframe work
- **DuckDB** — analytical queries over results Parquet, locally or on S3
- **joblib** — parallelism across parameter-sweep runs
- Storage: Parquet on local disk; S3 optional for archival/sharing
- **No Spark.** A full 10-year chain for one symbol is ~806k contracts and
  prices in 0.09s on a single core (~78 MB). Spark's job startup alone is
  10-30s, and the backtest engine is a sequential state machine — the worst
  fit for its execution model. The real parallelism is across independent
  sweep runs, which is a job-queue problem, not a data-parallelism one.
  Revisit only if this ever moves to tick-level data.

## Conventions
- Zone layout (same tree locally under `data/` or under an `s3://<bucket>/`
  prefix): `raw/<symbol>/...`, `processed/chains/<symbol>/...`,
  `results/<strategy>/<run_id>/...`
- All option pricing goes through a single shared module
  (`obl/pricing/black_scholes.py`) — do not reimplement BS math elsewhere.
  It stays pure NumPy: no I/O, no config, no dataframe library.
- **The engine contains no ticker symbols and no strategy names.** Strategy
  and instrument layers meet only at a daily chain-snapshot interface;
  strategies are declarative specs, not engine branches.
- Pricing changes must pass the QuantLib cross-validation suite
  (`tests/test_quantlib_reference.py`) as well as the internal checks —
  internal consistency alone cannot catch a shared-formula error.
- Every backtest run is parameterized and logged with a `run_id`; no
  in-place mutation of prior results.
- Validate any new pricing output against known real option prices/put-call
  parity before trusting it in a backtest (see Ch. 3-5 sanity checks).

## Shared with options-live-validator
- `obl/timebase.py` is **shared code**, imported by `options-live-validator`
  from a pinned tag of this package. Changing a clock's numbers changes strike
  selection in both repos and invalidates recorded residuals, so treat it as a
  published interface: keep the `id` values stable, and if a clock's output
  moves, bump its `id` version rather than editing in place.
- It must stay dependency-free (stdlib only). Sessions are injected via
  `SessionSource`; do not import a market-calendar library into it.

## Out of scope for this repo
- Live/streaming data, broker connections, real order placement → that's
  `options-live-validator`.
- Strategy selection research / ETF characteristic analysis → that's
  `options-research`.
