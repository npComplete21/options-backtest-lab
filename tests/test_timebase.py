"""Tau clock conventions.

The clock is the largest single modelling lever in the system: at 09:45 on
expiry day calendar and trading-hours tau differ by 5.3x, and 2.3x in
sigma*sqrt(tau), which is what strike selectors scale with.

This module is shared with ``options-live-validator``, which pins a tag of this
package rather than keeping its own clock. ``test_the_headline_ratio_holds`` is
the conformance test that fact rests on - if it ever changes value, every
recorded residual in both repos changes with it.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest
from src.ingestion import calendar as cal
from src.timebase import (
    CalendarClock,
    Session,
    ShortDatedTauError,
    TradingDayClock,
    TradingHoursClock,
    VolWeightedClock,
    at_midnight,
    clock_ratio,
    get_clock,
)

ET = ZoneInfo("America/New_York")


class _Sessions:
    """A SessionSource over regular 09:30-16:00 sessions, with early closes
    supplied explicitly. Stands in for each repo's real market calendar; the
    shared module takes sessions injected precisely so it needs neither."""

    def __init__(self, dates, early_closes=None):
        self._early = early_closes or {}
        self._by_date = {
            d: Session(
                date=d,
                open=dt.datetime.combine(d, dt.time(9, 30), tzinfo=ET),
                close=dt.datetime.combine(d, self._early.get(d, dt.time(16, 0)), tzinfo=ET),
            )
            for d in dates
        }

    def sessions_between(self, start: dt.date, end: dt.date) -> tuple[Session, ...]:
        return tuple(s for d, s in sorted(self._by_date.items()) if start <= d <= end)


@pytest.fixture(scope="module")
def session_dates():
    return frozenset(cal.sessions(dt.date(2026, 1, 1), dt.date(2027, 1, 1)))


@pytest.fixture(scope="module")
def hours(session_dates):
    return TradingHoursClock(_Sessions(session_dates))


# --- calendar clock: the old date-resolution numbers must survive ------------


@pytest.mark.parametrize(
    "start,expiry,expected_days",
    [
        (dt.date(2026, 1, 1), dt.date(2026, 1, 31), 30),
        (dt.date(2026, 9, 9), dt.date(2026, 10, 16), 37),
        (dt.date(2026, 1, 1), dt.date(2027, 1, 1), 365),
        (dt.date(2026, 9, 9), dt.date(2026, 9, 9), 0),
    ],
)
def test_calendar_clock_is_days_over_365(start, expiry, expected_days):
    """Normalised to midnight, the instant-based clock returns exactly what the
    date-based one did. This is the compatibility guarantee that let the
    protocol move to instants without restating any backtest result."""
    tau = CalendarClock().year_fraction(at_midnight(start), at_midnight(expiry))
    assert tau == pytest.approx(expected_days / 365)


def test_midnight_normalisation_is_timezone_independent():
    """Both endpoints shift together, so the zone cancels."""
    d, e = dt.date(2026, 9, 9), dt.date(2026, 10, 16)
    utc = CalendarClock().year_fraction(at_midnight(d), at_midnight(e))
    et = CalendarClock().year_fraction(at_midnight(d, ET), at_midnight(e, ET))
    assert utc == pytest.approx(et)


def test_expired_option_has_zero_tau_not_negative():
    c = CalendarClock()
    assert (
        c.year_fraction(at_midnight(dt.date(2026, 9, 10)), at_midnight(dt.date(2026, 9, 9))) == 0.0
    )


def test_intraday_time_now_changes_tau():
    """The date-resolution clock truncated the time of day, which is exactly
    what made it unusable at 0DTE: every instant within expiry day answered the
    same tau. Losing that truncation is the point of the change, so it is
    pinned rather than left to be rediscovered as a regression."""
    expiry = dt.datetime(2026, 10, 16, 16, 0, tzinfo=ET)
    morning = CalendarClock().year_fraction(dt.datetime(2026, 9, 9, 9, 45, tzinfo=ET), expiry)
    midnight = CalendarClock().year_fraction(at_midnight(dt.date(2026, 9, 9), ET), expiry)
    assert morning < midnight


def test_naive_datetimes_are_refused():
    """Assuming UTC or local for a naive input is a units error waiting to
    happen, and this module exists to prevent exactly that class of bug."""
    with pytest.raises(ValueError, match="timezone-aware"):
        CalendarClock().year_fraction(
            dt.datetime(2026, 9, 9, 9, 45), dt.datetime(2026, 9, 9, 16, 0, tzinfo=ET)
        )


# --- the headline: the two clocks disagree by a factor that moves strikes ----


def test_the_headline_ratio_holds(hours):
    """At 09:45 on expiry day: 5.3x in tau, 2.31x in sigma*sqrt(tau).

    Both repos' plans quote these numbers and both size strikes with them. If
    this test changes value, every strike selector in the program moves and
    every recorded residual is invalidated - so it is pinned here rather than
    restated as prose in two places.
    """
    now = dt.datetime(2026, 9, 9, 9, 45, tzinfo=ET)
    expiry = dt.datetime(2026, 9, 9, 16, 0, tzinfo=ET)
    calendar = CalendarClock()

    assert calendar.year_fraction(now, expiry) == pytest.approx(0.000713, abs=1e-6)
    assert hours.year_fraction(now, expiry) == pytest.approx(0.003815, abs=1e-6)
    assert hours.year_fraction(now, expiry) / calendar.year_fraction(now, expiry) == pytest.approx(
        5.35, abs=0.01
    )
    assert clock_ratio(now, expiry, hours, calendar) == pytest.approx(2.31, abs=0.01)


def test_trading_hours_clock_ignores_the_overnight(hours):
    """Close to next open contributes nothing, which is the entire reason a
    session clock exists."""
    close = dt.datetime(2026, 9, 9, 16, 0, tzinfo=ET)
    next_open = dt.datetime(2026, 9, 10, 9, 30, tzinfo=ET)
    assert hours.year_fraction(close, next_open) == pytest.approx(0.0, abs=1e-12)


def test_trading_hours_clock_is_timezone_agnostic(hours):
    """Same instants expressed in UTC must give the same tau; the session date
    is resolved with a day of padding so an instant that is 'tomorrow' in UTC
    does not silently drop a session."""
    now = dt.datetime(2026, 9, 9, 9, 45, tzinfo=ET)
    expiry = dt.datetime(2026, 9, 9, 16, 0, tzinfo=ET)
    assert hours.year_fraction(now, expiry) == pytest.approx(
        hours.year_fraction(now.astimezone(dt.UTC), expiry.astimezone(dt.UTC))
    )


def test_early_close_shortens_tau(session_dates):
    """A 0DTE strategy on a 13:00 close has barely half its usual time to
    expiry; a clock assuming 16:00 overstates tau by roughly 2x on exactly the
    days liquidity is worst."""
    half_day = dt.date(2026, 11, 27)
    full = TradingHoursClock(_Sessions(session_dates))
    early = TradingHoursClock(_Sessions(session_dates, {half_day: dt.time(13, 0)}))
    now = dt.datetime(2026, 11, 27, 9, 45, tzinfo=ET)
    expiry = dt.datetime(2026, 11, 27, 16, 0, tzinfo=ET)
    assert early.year_fraction(now, expiry) < full.year_fraction(now, expiry)


# --- whole-day clock: still offered, still refuses to fake intraday ----------


def test_trading_day_clock_differs_from_calendar(session_dates):
    """If these agreed, one of them would be mislabelled."""
    c, t = CalendarClock(), TradingDayClock(session_dates)
    a = at_midnight(dt.date(2026, 9, 9))
    e = at_midnight(dt.date(2026, 10, 16))
    assert t.year_fraction(a, e) != pytest.approx(c.year_fraction(a, e), rel=1e-3)


def test_trading_day_clock_requires_sessions():
    """A 252/365 approximation collapses to calendar/365 exactly, so a
    'convenient' default would hand back a calendar clock wearing the wrong
    label - the silent units error this module exists to prevent."""
    with pytest.raises(ValueError, match="requires the trading sessions"):
        TradingDayClock(frozenset())


def test_trading_day_clock_refuses_same_session_expiry(session_dates):
    """Counting whole sessions leaves no intraday resolution, so at 0DTE it can
    only answer zero - which would price every contract at intrinsic. It raises
    instead, and that refusal is why TradingHoursClock exists."""
    t = TradingDayClock(session_dates)
    now = dt.datetime(2026, 9, 9, 9, 45, tzinfo=ET)
    expiry = dt.datetime(2026, 9, 9, 16, 0, tzinfo=ET)
    with pytest.raises(ShortDatedTauError, match="whole sessions"):
        t.year_fraction(now, expiry)


# --- identity: a run's clock must round-trip through recorded metadata -------


def test_clock_ids_are_stable_and_resolvable():
    assert CalendarClock().id == "calendar-365/v1"
    assert get_clock("calendar-365/v1").id == "calendar-365/v1"


def test_session_based_clocks_carry_stable_ids(hours, session_dates):
    """Not in the registry - they need sessions injected - but the id still has
    to round-trip, or a run using one cannot be reproduced."""
    assert hours.id == "trading-hours-252x6.5/v1"
    assert TradingDayClock(session_dates).id == "trading-252/v1"


def test_unknown_clock_id_raises():
    with pytest.raises(KeyError, match="unknown tau clock"):
        get_clock("wall-clock/v9")


def test_vol_weighted_clock_refuses_rather_than_guessing():
    """Flat session time (TradingHoursClock) is the v1 choice. The U-shaped
    weighting on top of it has to be measured from recorded data, which the
    live-validator recorder produces; guessing it would be the invented
    assumption this program exists to avoid."""
    with pytest.raises(NotImplementedError, match="measured from recorded"):
        VolWeightedClock().year_fraction(
            dt.datetime(2026, 9, 9, 9, 45, tzinfo=ET),
            dt.datetime(2026, 9, 9, 16, 0, tzinfo=ET),
        )


def test_clock_ratio_refuses_a_degenerate_denominator(hours):
    """sqrt(tau_a/tau_b) is meaningless once tau_b collapses, and returning a
    huge number would look like a finding rather than a division by nothing."""
    instant = dt.datetime(2026, 9, 9, 16, 0, tzinfo=ET)
    with pytest.raises(ValueError, match="degenerate tau"):
        clock_ratio(instant, instant, CalendarClock(), hours)
