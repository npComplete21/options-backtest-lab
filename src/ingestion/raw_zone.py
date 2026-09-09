"""Raw zone: cached, idempotent local storage for ingested market data.

Layout (mirrors the S3 prefix in ``CLAUDE.md``, so the same tree works either
place)::

    data/raw/bars/<symbol>.parquet
    data/raw/dividends/<symbol>.parquet

Re-running ingestion is cheap and safe: a cached range that already covers the
request is served from disk, and a wider request refetches and rewrites the
union rather than appending duplicates.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import polars as pl

from src.ingestion.sources import BAR_SCHEMA, DIVIDEND_SCHEMA, BarSource, YFinanceSource

DEFAULT_ROOT = Path("data/raw")


def _safe_name(symbol: str) -> str:
    """Filesystem-safe filename for a symbol (``^VXN`` -> ``_VXN``).

    The true symbol is always stored as a column, so this is presentational
    only and never needs reversing.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", symbol)


class RawZone:
    def __init__(self, root: Path | str = DEFAULT_ROOT, source: BarSource | None = None):
        self.root = Path(root)
        self.source = source or YFinanceSource()

    def _path(self, kind: str, symbol: str) -> Path:
        return self.root / kind / f"{_safe_name(symbol)}.parquet"

    def _meta_path(self, kind: str, symbol: str) -> Path:
        return self.root / kind / f"{_safe_name(symbol)}.meta.json"

    def _coverage(self, kind: str, symbol: str) -> tuple[dt.date, dt.date] | None:
        """The widest range ever *requested* for this symbol.

        Coverage is tracked explicitly rather than inferred from the min/max
        dates present in the data, because those are not the same thing. A
        request ending on a Sunday caches data ending the prior Friday;
        inferring coverage from the data would conclude the cache fell short
        and refetch on every call. Quarter and year boundaries land on
        weekends routinely, so that is the common case, not an edge case.
        """
        meta = self._meta_path(kind, symbol)
        if not meta.exists():
            return None
        raw = json.loads(meta.read_text())
        return dt.date.fromisoformat(raw["start"]), dt.date.fromisoformat(raw["end"])

    def _write_coverage(self, kind: str, symbol: str, start: dt.date, end: dt.date) -> None:
        self._meta_path(kind, symbol).write_text(
            json.dumps({"start": start.isoformat(), "end": end.isoformat()})
        )

    def bars(
        self, symbol: str, start: dt.date, end: dt.date, *, refresh: bool = False
    ) -> pl.DataFrame:
        """Daily bars for ``symbol``, fetching only what the cache lacks."""
        path = self._path("bars", symbol)
        cached = pl.read_parquet(path) if path.exists() and not refresh else None
        have = None if refresh else self._coverage("bars", symbol)

        if cached is not None and have is not None:
            have_lo, have_hi = have
            if have_lo <= start and have_hi >= end:
                return cached.filter(pl.col("date").is_between(start, end))
            start, end = min(start, have_lo), max(end, have_hi)

        fetched = self.source.daily_bars(symbol, start, end)
        merged = (
            (pl.concat([cached, fetched]) if cached is not None and len(cached) else fetched)
            .unique(subset=["date"], keep="last")
            .sort("date")
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        merged.write_parquet(path)
        self._write_coverage("bars", symbol, start, end)
        return merged.filter(pl.col("date").is_between(start, end))

    def dividends(self, symbol: str, *, refresh: bool = False) -> pl.DataFrame:
        path = self._path("dividends", symbol)
        if path.exists() and not refresh:
            return pl.read_parquet(path)
        frame = self.source.dividends(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        return frame

    def trailing_dividend_yield(
        self, symbol: str, bars: pl.DataFrame, *, window_days: int = 365
    ) -> pl.DataFrame:
        """Continuous dividend yield ``q`` per date, from trailing cash dividends.

        ``q = (trailing 12m dividends) / spot``. For ETFs this is a good fit.
        For single stocks it approximates a discrete stream as continuous,
        which is a known limitation (plan section 3) affecting early-exercise
        behaviour around ex-dividend dates.
        """
        divs = self.dividends(symbol)
        if divs.is_empty():
            return bars.select(pl.col("date"), pl.lit(0.0).alias("div_yield"))

        d_dates = divs["date"].to_list()
        d_amts = divs["amount"].to_list()
        out = []
        for day, close in zip(bars["date"].to_list(), bars["close"].to_list(), strict=True):
            floor = day - dt.timedelta(days=window_days)
            ttm = sum(a for dd, a in zip(d_dates, d_amts, strict=True) if floor < dd <= day)
            out.append(ttm / close if close else 0.0)
        return bars.select(pl.col("date")).with_columns(pl.Series("div_yield", out))


__all__ = ["BAR_SCHEMA", "DIVIDEND_SCHEMA", "DEFAULT_ROOT", "RawZone"]
