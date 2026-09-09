"""Market data adapters.

Sources sit behind a protocol so the pipeline is not welded to one vendor.
yfinance is the v1 implementation: it is the only free source providing OHLC,
dividends, splits and volatility indices together. Alpha Vantage's REST API is
a drop-in replacement if a subscription is added - note that the Alpha Vantage
*MCP tools* are available to an assistant in a chat session, not to this
pipeline, so they cannot be the runtime source.

Price convention
----------------
The underlying path uses **raw close: split-adjusted but NOT
dividend-adjusted**. Verified against AAPL's 4:1 split on 2020-08-31, where
raw close is smooth (124.81 -> 129.04, no 4x jump) while ``Adj Close`` sits
about $2.35 below it across 2024.

This matters: dividends enter the model once, as the continuous yield ``q`` in
Black-Scholes. Feeding a dividend-adjusted series into a pricer that also
carries ``q`` would count them twice and quietly bias every call and put.
``adj_close`` is stored for reference and must not be used as the spot path.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol

import polars as pl

BAR_SCHEMA = {
    "symbol": pl.String,
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,  # split-adjusted, dividend-inclusive: the spot path
    "adj_close": pl.Float64,  # also dividend-adjusted: reference only
    "volume": pl.Int64,
}

DIVIDEND_SCHEMA = {"symbol": pl.String, "date": pl.Date, "amount": pl.Float64}


class DataUnavailableError(RuntimeError):
    """A source returned nothing for a symbol.

    Fatal rather than empty-defaulted: a silently empty vol index would make
    an instrument fall back to assumed volatility without anything saying so.
    """


class BarSource(Protocol):
    def daily_bars(self, symbol: str, start: dt.date, end: dt.date) -> pl.DataFrame: ...
    def dividends(self, symbol: str) -> pl.DataFrame: ...


class YFinanceSource:
    """yfinance-backed implementation of :class:`BarSource`."""

    def daily_bars(self, symbol: str, start: dt.date, end: dt.date) -> pl.DataFrame:
        import yfinance as yf

        raw = yf.Ticker(symbol).history(
            start=start.isoformat(),
            end=(end + dt.timedelta(days=1)).isoformat(),
            auto_adjust=False,  # keep splits applied, dividends NOT applied
            actions=False,
        )
        if raw.empty:
            raise DataUnavailableError(
                f"no bars for {symbol!r} between {start} and {end}. If this is a "
                "volatility index, it may have been delisted from this source "
                "(^RVX behaves this way) - check the instrument's vol_model."
            )
        raw = raw.reset_index()
        out = pl.DataFrame(
            {
                "symbol": [symbol] * len(raw),
                "date": [d.date() for d in raw["Date"]],
                "open": raw["Open"].to_numpy(),
                "high": raw["High"].to_numpy(),
                "low": raw["Low"].to_numpy(),
                "close": raw["Close"].to_numpy(),
                "adj_close": (
                    raw["Adj Close"].to_numpy() if "Adj Close" in raw else raw["Close"].to_numpy()
                ),
                "volume": raw["Volume"].fillna(0).astype("int64").to_numpy(),
            },
            schema=BAR_SCHEMA,
        )
        return out.sort("date")

    def dividends(self, symbol: str) -> pl.DataFrame:
        import yfinance as yf

        series = yf.Ticker(symbol).dividends
        if series is None or len(series) == 0:
            return pl.DataFrame(schema=DIVIDEND_SCHEMA)
        return pl.DataFrame(
            {
                "symbol": [symbol] * len(series),
                "date": [i.date() for i in series.index],
                "amount": series.to_numpy(),
            },
            schema=DIVIDEND_SCHEMA,
        ).sort("date")
