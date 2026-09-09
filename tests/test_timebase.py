"""Tau clock conventions.

The clock is the largest single modelling lever in the system: at 09:45 on
expiry day calendar and trading-hours tau differ by 5.3x, and 2.3x in
sigma*sqrt(tau), which is what strike selectors scale with.
"""

from __future__ import annotations

import datetime as dt

import pytest
from src.ingestion import calendar as cal
from src.timebase import (
    CalendarClock,
    IntradayClock,
    TradingDayClock,
    get_clock,
)


@pytest.fixture(scope="module")
def sessions():
    return frozenset(cal.sessions(dt.date(2026, 1, 1), dt.date(2027, 1, 1)))


@pytest.mark.parametrize(
    "start,expiry,expected_days",
    [
        (dt.date(2026, 1, 1), dt.date(2026, 1, 31), 30),
        (dt.date(2026, 9, 9), dt.date(2026, 10, 16), 37),
        (dt.date(2026, 1, 1), dt.date(2027, 1, 1), 365),
        (dt.date(2026, 9, 9), dt.date(2026, 9, 9), 0),  # 0DTE under a daily clock
    ],
)
def test_calendar_clock_is_days_over_365(start, expiry, expected_days):
    assert CalendarClock().tau(start, expiry) == pytest.approx(expected_days / 365)


def test_expired_option_has_zero_tau_not_negative():
    assert CalendarClock().tau(dt.date(2026, 9, 10), dt.date(2026, 9, 9)) == 0.0


def test_datetime_and_date_agree_for_a_daily_clock():
    d, e = dt.date(2026, 9, 9), dt.date(2026, 10, 16)
    assert CalendarClock().tau(dt.datetime(2026, 9, 9, 14, 30), e) == CalendarClock().tau(d, e)


def test_trading_clock_differs_from_calendar(sessions):
    """If these agreed, one of them would be mislabelled."""
    c, t = CalendarClock(), TradingDayClock(sessions)
    a, e = dt.date(2026, 9, 9), dt.date(2026, 10, 16)
    assert t.tau(a, e) != pytest.approx(c.tau(a, e), rel=1e-3)


def test_trading_clock_requires_sessions():
    """A 252/365 approximation collapses to calendar/365 exactly, so a
    'convenient' default would hand back a calendar clock wearing the wrong
    label - the silent units error this module exists to prevent."""
    with pytest.raises(ValueError, match="requires the trading sessions"):
        TradingDayClock(frozenset())


def test_clock_ids_are_stable_and_resolvable():
    """A run's clock must round-trip through recorded metadata, or the run
    cannot be reproduced."""
    assert CalendarClock().id == "calendar-365/v1"
    assert get_clock("calendar-365/v1").id == "calendar-365/v1"


def test_unknown_clock_id_raises():
    with pytest.raises(KeyError, match="unknown tau clock"):
        get_clock("wall-clock/v9")


def test_intraday_clock_refuses_rather_than_guessing():
    """0DTE needs a session weighting curve measured from recorded data.
    Substituting a daily clock would price every 0DTE contract at intrinsic."""
    with pytest.raises(NotImplementedError, match="measured from recorded"):
        IntradayClock().tau(dt.date(2026, 9, 9), dt.date(2026, 9, 9))
