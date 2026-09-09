"""Risk-free rate handling.

``^IRX`` (13-week T-bill) is the single short-rate series. A full curve is not
built yet: rho is small at the 7-90 DTE the first strategies target, so a
single short rate is accurate enough, and ``^TNX`` is available to interpolate
a term structure later if longer-dated strategies arrive.
"""

from __future__ import annotations

import numpy as np
import polars as pl

RATE_SYMBOL = "^IRX"


def to_continuous(quoted_percent: np.ndarray | float) -> np.ndarray:
    """Convert a quoted annual percentage to a continuously-compounded rate.

    ``^IRX`` quotes percent (4.35 meaning 4.35%). Black-Scholes takes a
    continuously-compounded decimal, so ``r_cc = ln(1 + r)``. The gap is small
    at current levels (4.35% -> 4.26%) but it is free to be exact, and it
    matters for put-call parity, which is asserted to 1e-10 in the test suite.
    """
    r = np.asarray(quoted_percent, dtype=float) / 100.0
    return np.log1p(r)


def rate_frame(bars: pl.DataFrame) -> pl.DataFrame:
    """Convert a ``^IRX`` bar frame to ``(date, rate)`` in continuous decimal."""
    return bars.select(
        pl.col("date"),
        pl.col("close").map_batches(lambda s: pl.Series(to_continuous(s.to_numpy()))).alias("rate"),
    ).drop_nulls()
