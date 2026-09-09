"""Raw zone caching, dividend yield, and rate conversion.

Hermetic by default: a fake source stands in for yfinance so the suite never
depends on network or on a vendor's uptime. The live-source check is marked
``network`` and deselected unless asked for - it exists to catch yfinance API
drift, which is the main risk of an unofficial data source.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from src.ingestion.rates import to_continuous
from src.ingestion.raw_zone import RawZone
from src.ingestion.sources import BAR_SCHEMA, DIVIDEND_SCHEMA, DataUnavailableError


class FakeSource:
    """Deterministic bars, counting fetches so caching can be asserted."""

    def __init__(self, dividends: list[tuple[dt.date, float]] | None = None):
        self.calls: list[tuple[str, dt.date, dt.date]] = []
        self._dividends = dividends or []

    def daily_bars(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
        days = [d for d in days if d.weekday() < 5]
        if not days:
            raise DataUnavailableError(f"no bars for {symbol!r}")
        n = len(days)
        return pl.DataFrame(
            {
                "symbol": [symbol] * n,
                "date": days,
                "open": np.full(n, 100.0),
                "high": np.full(n, 101.0),
                "low": np.full(n, 99.0),
                "close": np.full(n, 100.0),
                "adj_close": np.full(n, 98.0),
                "volume": np.full(n, 1_000, dtype="int64"),
            },
            schema=BAR_SCHEMA,
        )

    def dividends(self, symbol):
        if not self._dividends:
            return pl.DataFrame(schema=DIVIDEND_SCHEMA)
        return pl.DataFrame(
            {
                "symbol": [symbol] * len(self._dividends),
                "date": [d for d, _ in self._dividends],
                "amount": [a for _, a in self._dividends],
            },
            schema=DIVIDEND_SCHEMA,
        )


@pytest.fixture
def zone(tmp_path):
    return RawZone(root=tmp_path, source=FakeSource())


def test_bars_are_fetched_then_served_from_cache(zone):
    rng = (dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    first = zone.bars("QQQ", *rng)
    second = zone.bars("QQQ", *rng)
    assert len(zone.source.calls) == 1, "second call must hit the cache"
    assert first.equals(second)


def test_widening_the_range_refetches_and_keeps_the_union(zone):
    zone.bars("QQQ", dt.date(2024, 2, 1), dt.date(2024, 2, 29))
    wide = zone.bars("QQQ", dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    assert len(zone.source.calls) == 2
    assert wide["date"].min() == dt.date(2024, 1, 1)
    assert wide["date"].max() <= dt.date(2024, 3, 31)


def test_narrower_range_does_not_refetch(zone):
    zone.bars("QQQ", dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    narrow = zone.bars("QQQ", dt.date(2024, 2, 1), dt.date(2024, 2, 15))
    assert len(zone.source.calls) == 1
    assert narrow["date"].min() >= dt.date(2024, 2, 1)


def test_merging_never_duplicates_dates(zone):
    zone.bars("QQQ", dt.date(2024, 2, 1), dt.date(2024, 2, 29))
    merged = zone.bars("QQQ", dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    assert merged["date"].n_unique() == len(merged)


def test_refresh_forces_a_refetch(zone):
    rng = (dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    zone.bars("QQQ", *rng)
    zone.bars("QQQ", *rng, refresh=True)
    assert len(zone.source.calls) == 2


def test_index_symbols_get_filesystem_safe_names(zone, tmp_path):
    zone.bars("^VXN", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert (tmp_path / "bars" / "_VXN.parquet").exists()


def test_symbol_column_preserves_the_true_symbol(zone):
    bars = zone.bars("^VXN", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert bars["symbol"].unique().to_list() == ["^VXN"]


def test_empty_source_raises_rather_than_returning_nothing(zone):
    """A silently empty vol index would downgrade an instrument invisibly."""
    with pytest.raises(DataUnavailableError):
        zone.bars("QQQ", dt.date(2024, 1, 6), dt.date(2024, 1, 7))  # weekend only


def test_trailing_dividend_yield(tmp_path):
    z = RawZone(
        root=tmp_path,
        source=FakeSource(
            dividends=[
                (dt.date(2023, 6, 1), 1.0),
                (dt.date(2023, 12, 1), 1.0),
                (dt.date(2024, 6, 1), 1.0),
            ]
        ),
    )
    bars = z.bars("TEST", dt.date(2024, 7, 1), dt.date(2024, 7, 31))
    q = z.trailing_dividend_yield("TEST", bars)
    # Trailing 365d from July 2024 covers the Dec 2023 and Jun 2024 payments only.
    assert q["div_yield"].max() == pytest.approx(2.0 / 100.0)


def test_zero_dividend_symbol_yields_zero(zone):
    bars = zone.bars("QQQ", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    q = zone.trailing_dividend_yield("QQQ", bars)
    assert q["div_yield"].to_numpy().max() == 0.0


@pytest.mark.parametrize("quoted,expected", [(4.35, 0.042580), (0.05, 0.00049988), (0.0, 0.0)])
def test_quoted_percent_to_continuous_rate(quoted, expected):
    assert float(to_continuous(quoted)) == pytest.approx(expected, abs=1e-6)


def test_continuous_rate_is_below_simple_rate():
    """Sanity: continuous compounding needs a lower rate for the same growth."""
    assert float(to_continuous(5.0)) < 0.05


@pytest.mark.network
def test_live_source_still_matches_expected_schema():
    """Opt-in: catches yfinance API drift. Run with `pytest -m network`."""
    z = RawZone(root="data/raw")
    bars = z.bars("QQQ", dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    assert len(bars) > 50
    assert set(bars.columns) == set(BAR_SCHEMA)
    assert bars["close"].min() > 0


def test_range_ending_on_a_weekend_does_not_refetch_forever(zone):
    """Regression: 2024-03-31 is a Sunday, so no bar exists on the end date.

    Inferring cache coverage from the data's max date would decide the cache
    fell short and refetch on every call. Quarter and year ends land on
    weekends routinely, so this would have hit constantly in normal use.
    """
    rng = (dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    zone.bars("QQQ", *rng)
    zone.bars("QQQ", *rng)
    zone.bars("QQQ", *rng)
    assert len(zone.source.calls) == 1


def test_range_starting_on_a_weekend_does_not_refetch(zone):
    rng = (dt.date(2024, 1, 6), dt.date(2024, 1, 31))  # 1/6 is a Saturday
    zone.bars("QQQ", *rng)
    zone.bars("QQQ", *rng)
    assert len(zone.source.calls) == 1


def test_coverage_survives_a_new_zone_instance(tmp_path):
    """Cache must work across processes, not just within one object's lifetime."""
    src = FakeSource()
    rng = (dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    RawZone(root=tmp_path, source=src).bars("QQQ", *rng)
    RawZone(root=tmp_path, source=src).bars("QQQ", *rng)
    assert len(src.calls) == 1
