"""Trading sessions and expiration dates.

Expiration generation is availability-aware: it emits only the series that
were listable for a symbol on the quote date. QQQ has expirations on all five
weekdays today but had only monthlies and Friday weeklies in 2015, and a
generator that ignores that would produce a backtest trading contracts which
never existed - output that looks entirely plausible and means nothing.
"""

from __future__ import annotations

import datetime as dt
import functools

import pandas_market_calendars as mcal

from src.instruments.base import ExpirationType, Instrument

_CALENDAR = "NYSE"


@functools.lru_cache(maxsize=8)
def _sessions_cached(start: dt.date, end: dt.date) -> tuple[dt.date, ...]:
    sched = mcal.get_calendar(_CALENDAR).schedule(start_date=start, end_date=end)
    return tuple(d.date() for d in sched.index)


def sessions(start: dt.date, end: dt.date) -> list[dt.date]:
    """NYSE trading days in ``[start, end]``."""
    return list(_sessions_cached(start, end))


def _roll_back_to_session(day: dt.date, valid: frozenset[dt.date]) -> dt.date | None:
    """Move an expiration off a holiday to the prior trading day.

    A holiday expiration is settled on the preceding session, so the last
    tradeable date - the one a backtest actually needs - is that session.
    """
    for _ in range(7):
        if day in valid:
            return day
        day -= dt.timedelta(days=1)
    return None


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The ``n``-th ``weekday`` of a month (Monday=0)."""
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def monthly_expirations(start: dt.date, end: dt.date, valid: frozenset[dt.date]) -> set[dt.date]:
    """Third Friday of each month, rolled back off holidays.

    Before February 2015 the contract technically expired the Saturday after
    the third Friday, but the last *trading* day was the third Friday in both
    regimes, so this needs no era switch.
    """
    out: set[dt.date] = set()
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        day = _roll_back_to_session(_nth_weekday(year, month, 4, 3), valid)
        if day and start <= day <= end:
            out.add(day)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def weekly_expirations(
    weekday: int, start: dt.date, end: dt.date, valid: frozenset[dt.date]
) -> set[dt.date]:
    """Every occurrence of ``weekday`` in range, rolled back off holidays."""
    out: set[dt.date] = set()
    day = start + dt.timedelta(days=(weekday - start.weekday()) % 7)
    while day <= end:
        rolled = _roll_back_to_session(day, valid)
        if rolled and start <= rolled <= end:
            out.add(rolled)
        day += dt.timedelta(days=7)
    return out


def expirations_for(
    instrument: Instrument,
    start: dt.date,
    end: dt.date,
    *,
    as_of: dt.date | None = None,
) -> list[dt.date]:
    """Expirations listable for ``instrument`` between ``start`` and ``end``.

    Parameters
    ----------
    as_of
        The quote date whose listing regime applies. Defaults to ``start``.
        Passing the quote date is what prevents a 2015 backtest from seeing
        expiration series that were not introduced until 2022.
    """
    as_of = as_of or start
    available = set(instrument.expirations_available_on(as_of))
    if not available:
        return []

    valid = frozenset(sessions(start - dt.timedelta(days=10), end))
    out: set[dt.date] = set()
    for exp_type in available:
        if exp_type is ExpirationType.MONTHLY:
            out |= monthly_expirations(start, end, valid)
        else:
            out |= weekly_expirations(exp_type.weekday, start, end, valid)
    return sorted(out)
