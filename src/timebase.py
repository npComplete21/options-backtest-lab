"""Time-to-expiry conventions.

Separated from both pricing and strategy code because the two must agree, and
because this module is the intended extraction point for a package shared with
``options-live-validator``. Nothing here imports from the rest of the project.

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
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, runtime_checkable

CALENDAR_DAYS_PER_YEAR = 365.0
TRADING_DAYS_PER_YEAR = 252.0


class ShortDatedTauError(ValueError):
    """Raised when a clock is asked for a tau it cannot represent honestly.

    A daily clock has no intraday resolution, so on expiry day it can only
    answer zero. For 0DTE that is not a rounding error - it is the entire
    quantity being modelled. Failing loudly beats returning a number that
    silently prices every 0DTE contract at intrinsic.
    """


@runtime_checkable
class TauClock(Protocol):
    @property
    def id(self) -> str:
        """Stable identifier recorded in run metadata, e.g. ``calendar-365/v1``."""

    def tau(self, as_of: dt.date | dt.datetime, expiry: dt.date) -> float:
        """Year fraction from ``as_of`` to ``expiry``."""


def _as_date(value: dt.date | dt.datetime) -> dt.date:
    return value.date() if isinstance(value, dt.datetime) else value


class CalendarClock:
    """Calendar days / 365 - the market convention for quoted vol.

    Matches how implied volatility is quoted and keeps put-call parity
    consistent with the rate, which is why it is the default. Its known
    artifact is that value decays over weekends, when no trading occurs.
    """

    id = "calendar-365/v1"

    def tau(self, as_of: dt.date | dt.datetime, expiry: dt.date) -> float:
        days = (expiry - _as_date(as_of)).days
        return max(days, 0) / CALENDAR_DAYS_PER_YEAR


class TradingDayClock:
    """Trading days / 252 - decay follows sessions, not the wall calendar.

    Closer to how theta is actually realised, but inconsistent with quoted
    implied vol, so mixing it with an unadjusted vol surface double-counts the
    weekend effect. Offered for comparison, not as the default.

    Sessions must be supplied. There is deliberately no fallback: scaling
    calendar days by 252/365 and then dividing by 252 is algebraically
    identical to calendar/365, so a "convenient" default would hand back a
    calendar clock wearing a trading-day label - the exact class of silent
    units error this module exists to prevent.
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

    def tau(self, as_of: dt.date | dt.datetime, expiry: dt.date) -> float:
        start = _as_date(as_of)
        if expiry <= start:
            return 0.0
        count = sum(1 for d in self._sessions if start < d <= expiry)
        return count / TRADING_DAYS_PER_YEAR


class IntradayClock:
    """Placeholder for the 0DTE-capable clock. Not implemented.

    A correct intraday clock cannot be written from first principles: intraday
    volatility is U-shaped, so even a trading-hours clock misstates tau through
    the session, and the weighting curve has to be *measured* from recorded
    market data. That data does not exist yet - ``options-live-validator``'s
    recorder is what produces it.

    This class exists so the gap is visible in the type system rather than
    being discovered when a 0DTE backtest silently returns intrinsic values.
    """

    id = "intraday-measured/v0-unimplemented"

    def tau(self, as_of: dt.date | dt.datetime, expiry: dt.date) -> float:
        raise NotImplementedError(
            "Intraday tau requires a session weighting curve measured from recorded "
            "market data, which does not exist yet. See the options-live-validator "
            "recorder. Do not substitute a calendar or trading-day clock for 0DTE "
            "work: at 09:45 on expiry day they differ by 5.3x in tau and 2.3x in "
            "sigma*sqrt(tau), which moves strike selection by more than a factor of two."
        )


DEFAULT_CLOCK = CalendarClock()

# Only clocks constructible without external data are pre-registered.
# TradingDayClock needs sessions injected, so a run using it must supply the
# instance; its id still round-trips through run metadata.
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
