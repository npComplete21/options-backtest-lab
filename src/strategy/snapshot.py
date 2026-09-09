"""The one interface between instrument data and strategy logic.

A :class:`ChainSnapshot` is everything a strategy may know on a given day.
Strategies read it and never touch a symbol, a data source, or a vol model;
the instrument layer builds it and never learns which strategy is running.
See ``CLAUDE.md``: the engine contains no ticker symbols and no strategy names.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

from src.timebase import DEFAULT_CLOCK, TauClock, at_midnight

CONTRACT_COLUMNS = ("expiration", "strike", "right", "theo", "delta", "gamma", "vega", "sigma")


class NoContractsError(LookupError):
    """No contract satisfies a selector. Fatal rather than silently skipped:
    a strategy that quietly opens fewer legs than it declared is no longer the
    strategy under test."""


@dataclass(frozen=True)
class ChainSnapshot:
    """One day's option chain for one underlying.

    ``contracts`` carries greeks computed from the volatility *surface*, not
    from a single flat vol. This matters for strike selection: deltas from a
    flat vol are wrong by roughly 3-5 points on the wings, so a 16-delta
    selector fed flat-vol deltas picks a materially different strike than the
    one intended. Skew is an input here, never a residual.
    """

    as_of: dt.date
    symbol: str
    spot: float
    contracts: pl.DataFrame
    clock: TauClock = DEFAULT_CLOCK

    def __post_init__(self) -> None:
        missing = set(CONTRACT_COLUMNS) - set(self.contracts.columns)
        if missing:
            raise ValueError(f"chain snapshot missing columns: {sorted(missing)}")

    def expirations(self) -> list[dt.date]:
        return sorted(self.contracts["expiration"].unique().to_list())

    def dte(self, expiry: dt.date) -> int:
        return (expiry - self.as_of).days

    def tau(self, expiry: dt.date) -> float:
        """Year fraction under this snapshot's clock.

        Read through the clock rather than computed inline: the convention is
        a recorded input, and hardcoding calendar days here would move strike
        selection without leaving a trace in run metadata.

        A snapshot is one *day*, so both endpoints are normalised to the same
        wall time and the timezone cancels - this returns exactly the
        ``days/365`` a date-resolution clock did. Intraday work does not go
        through here: it holds real instants already and calls the clock
        directly, which is the whole reason the clock protocol takes instants.
        """
        return self.clock.year_fraction(at_midnight(self.as_of), at_midnight(expiry))

    def slice(self, expiry: dt.date, right: str) -> pl.DataFrame:
        out = self.contracts.filter(
            (pl.col("expiration") == expiry) & (pl.col("right") == right.upper())
        )
        if out.is_empty():
            raise NoContractsError(
                f"{self.symbol}: no {right.upper()} contracts expiring {expiry} on {self.as_of}"
            )
        return out.sort("strike")

    def atm_sigma(self, expiry: dt.date) -> float:
        """Implied vol nearest the money, used by standard-deviation selectors."""
        calls = self.slice(expiry, "C")
        idx = (calls["strike"] - self.spot).abs().arg_min()
        return float(calls["sigma"][idx])
