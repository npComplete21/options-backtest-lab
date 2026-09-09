"""Time-to-expiry conventions. **Shared with ``options-live-validator``.**

Separated from both pricing and strategy code because the two must agree, and
because both repos in this program must resolve tau through *this* module.
``options-live-validator`` imports it from a pinned tag of this package rather
than keeping its own copy; see that repo's ``docs/IMPLEMENTATION_PLAN.md``
section 3, which makes an identical clock a hard requirement.

Nothing here imports from the rest of the project, and nothing here imports a
market-calendar library. Sessions are **injected** (:class:`SessionSource`), so
this module stays pure-stdlib and can be depended on without dragging
``pandas_market_calendars``, numpy or polars behind it. Each repo builds its
own sessions and hands them in.

Why this is its own module
--------------------------
The tau convention is the largest single modelling lever in the system, not a
formatting detail. At 09:45 on expiry day, calendar tau = 0.000713 while
trading-hours tau = 0.003815 - **5.3x in tau, 2.3x in sigma*sqrt(tau)**. Strike
selectors are expressed in deltas and standard deviations, both of which scale
with ``sigma*sqrt(tau)``, so the clock choice moves *which strikes get sold* by
better than a factor of two.

If two repos pick clocks independently, no live-versus-backtest comparison
means anything, and the discrepancy shows up looking like a market finding
rather than a units error. So a clock is an **identified, recorded input**: it
carries a stable ``id`` that belongs in ``params.json`` and in the ``run_id``
hash, never an implicit default buried at a call site.

Resolution: instants, not dates
-------------------------------
Every clock here answers over **timezone-aware instants**. An earlier version
of this module was date-resolution, which was survivable while this repo only
priced multi-week tenors and fatal the moment 0DTE entered the program: on
expiry day a date-resolution clock can only answer zero, which prices every
0DTE contract at intrinsic.

Date-granular callers normalise with :func:`at_midnight` and get exactly the
old numbers back - ``days/365`` is unchanged when both endpoints sit at the
same wall time. Naive datetimes are rejected outright rather than assumed to be
UTC or local, because that assumption is a units error waiting to happen.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

#: Standard market convention: 365 calendar days.
CALENDAR_DAYS_PER_YEAR = 365.0
CALENDAR_SECONDS_PER_YEAR = CALENDAR_DAYS_PER_YEAR * 24 * 3600

#: 252 sessions, 6.5 hours each. The denominator for trading-time tau.
TRADING_DAYS_PER_YEAR = 252.0
REGULAR_SESSION_HOURS = 6.5
TRADING_SECONDS_PER_YEAR = TRADING_DAYS_PER_YEAR * REGULAR_SESSION_HOURS * 3600

#: Below this, Black-Scholes is numerically degenerate: delta becomes a step
#: function and vega collapses. Live positions must be flat before this is
#: reached, so hitting it means a bug, not a trade.
MIN_TAU = 1e-8

UTC = dt.UTC


class ShortDatedTauError(ValueError):
    """Raised when a clock is asked for a tau it cannot represent honestly.

    A session-*counting* clock has no intraday resolution, so on expiry day it
    can only answer zero. For 0DTE that is not a rounding error - it is the
    entire quantity being modelled. Failing loudly beats returning a number
    that silently prices every 0DTE contract at intrinsic.
    """


def at_midnight(value: dt.date | dt.datetime, tz: dt.tzinfo = UTC) -> dt.datetime:
    """Normalise a date to an instant for date-granular callers.

    Both endpoints of a tau calculation must be normalised the same way, and
    then the timezone cancels: ``days/365`` is identical whichever ``tz`` is
    used. It is a parameter only so a caller working in market time is not
    forced to round-trip through UTC.
    """
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=tz)
    return dt.datetime.combine(value, dt.time.min, tzinfo=tz)


@dataclass(frozen=True)
class Session:
    """One trading day, as an inclusive-open/exclusive-close instant pair.

    Carrying real open and close instants rather than a bare date is what lets
    early closes shorten tau correctly. Half-days matter more than they sound:
    a 0DTE strategy on a 13:00 close has barely half its usual time to expiry,
    and a clock assuming 16:00 would overstate tau by roughly 2x on exactly the
    days liquidity is worst.
    """

    date: dt.date
    open: dt.datetime
    close: dt.datetime

    @property
    def seconds(self) -> float:
        return (self.close - self.open).total_seconds()

    def overlap_seconds(self, start: dt.datetime, end: dt.datetime) -> float:
        """Seconds of this session lying inside ``[start, end)``."""
        lo = max(self.open, start)
        hi = min(self.close, end)
        return max(0.0, (hi - lo).total_seconds())


@runtime_checkable
class SessionSource(Protocol):
    """Supplies exchange sessions. Implemented by each repo over its own
    calendar, so this module needs no market-calendar dependency."""

    def sessions_between(self, start: dt.date, end: dt.date) -> tuple[Session, ...]:
        """Sessions intersecting ``[start, end]``, inclusive of both dates."""
        ...


@runtime_checkable
class TauClock(Protocol):
    @property
    def id(self) -> str:
        """Stable identifier recorded in run metadata, e.g. ``calendar-365/v1``."""

    def year_fraction(self, now: dt.datetime, expiry: dt.datetime) -> float:
        """Year fraction between two timezone-aware instants."""
        ...


def _validate(now: dt.datetime, expiry: dt.datetime) -> None:
    if now.tzinfo is None or expiry.tzinfo is None:
        raise ValueError(
            "tau clocks require timezone-aware instants; naive input is ambiguous. "
            "Date-granular callers should use at_midnight()."
        )


class CalendarClock:
    """Wall-clock seconds over 365 days - the market convention for quoted vol.

    Matches how implied volatility is quoted and keeps put-call parity
    consistent with the rate, which is why it remains the default here. Its
    known artifact is that value decays over weekends, when no trading occurs;
    at 0DTE the same artifact becomes severe, pricing six hours of market risk
    as if spread across six hours of a sleeping world. It is kept for 0DTE work
    only as the baseline the 2.3x comparison is made against.
    """

    id = "calendar-365/v1"

    def year_fraction(self, now: dt.datetime, expiry: dt.datetime) -> float:
        _validate(now, expiry)
        return max((expiry - now).total_seconds() / CALENDAR_SECONDS_PER_YEAR, 0.0)


class TradingHoursClock:
    """Seconds the market is actually open, over 252 x 6.5 hours.

    The only clock here with intraday resolution, and therefore the only one
    usable at 0DTE. Overnights, weekends and holidays contribute nothing, and
    early closes shorten the day correctly because :class:`Session` carries
    real close instants.

    Sessions must be supplied; there is deliberately no default calendar, for
    the same reason :class:`TradingDayClock` has none.
    """

    id = "trading-hours-252x6.5/v1"

    def __init__(self, sessions: SessionSource):
        self._sessions = sessions

    def trading_seconds(self, now: dt.datetime, expiry: dt.datetime) -> float:
        _validate(now, expiry)
        if expiry <= now:
            return 0.0
        # Widened by a day at each end because ``now`` may be expressed in any
        # timezone: 20:00 in market time is already tomorrow in UTC, and
        # resolving the session date from the raw instant would drop a session.
        # This module deliberately does not know the market's timezone, so it
        # over-fetches instead; a session outside the interval overlaps by zero
        # and contributes nothing.
        pad = dt.timedelta(days=1)
        sessions = self._sessions.sessions_between((now - pad).date(), (expiry + pad).date())
        return sum(s.overlap_seconds(now, expiry) for s in sessions)

    def year_fraction(self, now: dt.datetime, expiry: dt.datetime) -> float:
        return self.trading_seconds(now, expiry) / TRADING_SECONDS_PER_YEAR


class TradingDayClock:
    """Whole trading days / 252 - decay follows sessions, not the wall calendar.

    Closer to how theta is actually realised, but inconsistent with quoted
    implied vol, so mixing it with an unadjusted vol surface double-counts the
    weekend effect. Offered for comparison, not as the default.

    Sessions must be supplied. There is deliberately no fallback: scaling
    calendar days by 252/365 and then dividing by 252 is algebraically
    identical to calendar/365, so a "convenient" default would hand back a
    calendar clock wearing a trading-day label - the exact class of silent
    units error this module exists to prevent.

    Counting whole sessions leaves it with no intraday resolution, so it raises
    :class:`ShortDatedTauError` rather than returning zero for a same-session
    expiry. That refusal is the point: it is the reason
    :class:`TradingHoursClock` exists.
    """

    id = "trading-252/v1"

    def __init__(self, sessions: frozenset[dt.date]):
        if not sessions:
            raise ValueError(
                "TradingDayClock requires the trading sessions it should count. "
                "Without them it can only approximate calendar days, which "
                "collapses to calendar/365 while still claiming to be a "
                "trading-day clock."
            )
        self._sessions = frozenset(sessions)

    def year_fraction(self, now: dt.datetime, expiry: dt.datetime) -> float:
        _validate(now, expiry)
        if expiry <= now:
            return 0.0
        start, end = now.date(), expiry.date()
        count = sum(1 for d in self._sessions if start < d <= end)
        if count == 0:
            raise ShortDatedTauError(
                f"{self.id} counts whole sessions and found none between "
                f"{now.isoformat()} and {expiry.isoformat()}, so it can only answer "
                "zero for an expiry that has not yet arrived. Use TradingHoursClock "
                "for intraday tau; substituting a whole-day clock at 0DTE prices "
                "every contract at intrinsic."
            )
        return count / TRADING_DAYS_PER_YEAR


class VolWeightedClock:
    """Placeholder for the vol-weighted intraday clock. Not implemented.

    :class:`TradingHoursClock` measures intraday tau in *flat* session seconds,
    which is already enough for 0DTE and is the v1 choice. What is still
    missing is the weighting: intraday volatility is U-shaped, heavy at the
    open and into the close, so flat session time still misstates tau through
    the session.

    That curve cannot be written from first principles - it has to be
    *measured* from recorded market data, and ``options-live-validator``'s
    recorder is what produces it (its plan section 3). This class exists so the
    gap stays visible in the type system rather than being discovered as a
    residual nobody can explain.
    """

    id = "vol-weighted-measured/v0-unimplemented"

    def year_fraction(self, now: dt.datetime, expiry: dt.datetime) -> float:
        raise NotImplementedError(
            "Vol-weighted intraday tau requires a session weighting curve measured "
            "from recorded market data, which does not exist yet. See the "
            "options-live-validator recorder. Use TradingHoursClock for flat "
            "intraday time; do not substitute a calendar or whole-day clock at "
            "0DTE, where they differ by 5.3x in tau and 2.3x in sigma*sqrt(tau)."
        )


def clock_ratio(now: dt.datetime, expiry: dt.datetime, a: TauClock, b: TauClock) -> float:
    """``sqrt(tau_a / tau_b)`` - how much the clock choice moves sigma*sqrt(tau).

    This is the quantity that matters, because strike selectors work in
    standard deviations. Reported alongside the surface so the convention's
    effect stays visible rather than buried in a constant.
    """
    tau_b = b.year_fraction(now, expiry)
    if tau_b <= MIN_TAU:
        raise ValueError(f"denominator clock {b.id} returned a degenerate tau ({tau_b})")
    return (a.year_fraction(now, expiry) / tau_b) ** 0.5


DEFAULT_CLOCK = CalendarClock()

# Only clocks constructible without external data are pre-registered. The
# session-based clocks need sessions injected, so a run using one must supply
# the instance; its id still round-trips through run metadata.
_REGISTRY: dict[str, TauClock] = {c.id: c for c in (CalendarClock(),)}


def get_clock(clock_id: str) -> TauClock:
    """Resolve a clock by its recorded id, so a run can be reproduced exactly."""
    try:
        return _REGISTRY[clock_id]
    except KeyError:
        raise KeyError(
            f"unknown tau clock {clock_id!r}. Known: {sorted(_REGISTRY)}. "
            "A run's clock must be resolvable from its recorded id or the run "
            "cannot be reproduced."
        ) from None
