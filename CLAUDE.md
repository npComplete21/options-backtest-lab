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
2. **Processed zone (S3, Parquet)** — synthetic historical option chains:
   one row per (date, ticker, expiration, strike, type) with Black-Scholes
   theoretical value, delta/gamma/theta/vega, computed via PySpark.
3. **Backtest engine (PySpark)** — walks the processed zone day-by-day,
   applies a strategy + adjustment rule, produces a results table
   (P&L, Sharpe, max drawdown, win rate) per parameter set.
4. **Analysis** — Jupyter notebooks in `notebooks/` reading results from
   Athena/S3 for comparison across strategies and tickers.

## Tech stack
- Python 3.11, PySpark
- AWS: S3 (raw + processed zones), EMR Serverless for Spark jobs, Athena for
  querying results, IAM roles scoped to this project only
- Local dev: run Spark in local mode against a small date range before
  submitting full EMR jobs

## Conventions
- S3 bucket layout: `s3://<bucket>/raw/<ticker>/...`,
  `s3://<bucket>/processed/chains/<ticker>/...`,
  `s3://<bucket>/results/<strategy>/<run_id>/...`
- All option pricing goes through a single shared module
  (`src/pricing/black_scholes.py`) — do not reimplement BS math elsewhere.
- Every backtest run is parameterized and logged with a `run_id`; no
  in-place mutation of prior results.
- Validate any new pricing output against known real option prices/put-call
  parity before trusting it in a backtest (see Ch. 3-5 sanity checks).

## Out of scope for this repo
- Live/streaming data, broker connections, real order placement → that's
  `options-live-validator`.
- Strategy selection research / ETF characteristic analysis → that's
  `options-research`.
