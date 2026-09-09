"""Trading sessions and availability-aware expiration generation."""

from __future__ import annotations

import datetime as dt

from src.ingestion import calendar as cal
from src.instruments import registry
from src.instruments.base import ExpirationType


def _valid(start, end):
    return frozenset(cal.sessions(start - dt.timedelta(days=10), end))


def test_sessions_exclude_weekends_and_holidays():
    days = cal.sessions(dt.date(2025, 12, 22), dt.date(2025, 12, 28))
    assert dt.date(2025, 12, 25) not in days  # Christmas
    assert dt.date(2025, 12, 27) not in days  # Saturday
    assert dt.date(2025, 12, 26) in days


def test_monthly_expirations_are_third_fridays():
    start, end = dt.date(2025, 1, 1), dt.date(2025, 6, 30)
    got = sorted(cal.monthly_expirations(start, end, _valid(start, end)))
    assert got == [
        dt.date(2025, 1, 17),
        dt.date(2025, 2, 21),
        dt.date(2025, 3, 21),
        dt.date(2025, 4, 17),  # Good Friday 4/18 -> rolls back to Thursday
        dt.date(2025, 5, 16),
        dt.date(2025, 6, 20),
    ]


def test_holiday_expiration_rolls_back_to_prior_session():
    """April 2025's third Friday was Good Friday; the last tradeable day is Thursday."""
    start, end = dt.date(2025, 4, 1), dt.date(2025, 4, 30)
    got = cal.monthly_expirations(start, end, _valid(start, end))
    assert dt.date(2025, 4, 18) not in got
    assert dt.date(2025, 4, 17) in got


def test_weekly_expirations_land_on_the_requested_weekday():
    start, end = dt.date(2025, 3, 1), dt.date(2025, 3, 31)
    got = cal.weekly_expirations(2, start, end, _valid(start, end))  # Wednesday
    assert got and all(d.weekday() == 2 for d in got)


def test_qqq_2015_has_no_midweek_expirations():
    """Regression for the failure this module exists to prevent.

    Emitting the modern five-weekday grid across a 2015 window would backtest
    contracts that did not exist, producing plausible and meaningless output.
    """
    start, end = dt.date(2015, 1, 1), dt.date(2015, 3, 31)
    got = cal.expirations_for(registry.get("QQQ"), start, end, as_of=start)
    assert got, "2015 should still have Friday and monthly expirations"
    assert {d.weekday() for d in got} == {4}, "only Fridays existed for QQQ in 2015"


def test_qqq_today_has_all_five_weekdays():
    start, end = dt.date(2026, 1, 1), dt.date(2026, 3, 31)
    got = cal.expirations_for(registry.get("QQQ"), start, end, as_of=start)
    assert {d.weekday() for d in got} == {0, 1, 2, 3, 4}


def test_expirations_are_deduplicated_across_overlapping_series():
    """A third Friday is both a monthly and a weekly - it must appear once."""
    start, end = dt.date(2026, 1, 1), dt.date(2026, 3, 31)
    got = cal.expirations_for(registry.get("QQQ"), start, end, as_of=start)
    assert len(got) == len(set(got))


def test_as_of_controls_the_listing_regime_not_the_range():
    """Same range, different quote dates, different available series."""
    start, end = dt.date(2026, 1, 1), dt.date(2026, 3, 31)
    qqq = registry.get("QQQ")
    modern = cal.expirations_for(qqq, start, end, as_of=dt.date(2026, 1, 1))
    legacy = cal.expirations_for(qqq, start, end, as_of=dt.date(2015, 1, 1))
    assert len(legacy) < len(modern)
    assert {d.weekday() for d in legacy} == {4}


def test_single_stock_gets_fridays_only():
    start, end = dt.date(2026, 1, 1), dt.date(2026, 3, 31)
    got = cal.expirations_for(registry.get("AAPL"), start, end, as_of=start)
    assert {d.weekday() for d in got} == {4}


def test_no_expirations_when_nothing_was_listed_yet():
    start, end = dt.date(1990, 1, 1), dt.date(1990, 3, 31)
    assert cal.expirations_for(registry.get("QQQ"), start, end, as_of=start) == []


def test_every_expiration_is_a_trading_session():
    start, end = dt.date(2025, 1, 1), dt.date(2025, 12, 31)
    got = cal.expirations_for(registry.get("QQQ"), start, end, as_of=start)
    valid = set(cal.sessions(start, end))
    assert got and set(got) <= valid


def test_expiration_types_map_to_correct_weekdays():
    assert ExpirationType.MONTHLY.weekday == 4
    assert ExpirationType.WEEKLY_MON.weekday == 0
    assert ExpirationType.WEEKLY_THU.weekday == 3
