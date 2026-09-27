"""Entry triggers and management rules - the second half of the strategy DSL.

Legs say *what* to trade; these say *when to open it* and *what to do while it
is open*. Both halves live in YAML so that adding a strategy stays a config
change, and both are **validated at load time**, before any compute starts.

Why these are typed rather than passthrough dicts
-------------------------------------------------
They began as ``list[dict[str, Any]]``, which loads anything and means nothing.
Two things force real modelling:

*   **Natenberg's first rule is specified as a load-time rejection.** Never
    adjust by adding to a losing structure. A rule you can only catch at
    runtime is a rule the engine discovers mid-session, with a position on. So
    the action vocabulary is a closed set (:data:`SAFE_ACTIONS`), and anything
    that could increase short quantity in an open position is refused here,
    with an explicit and deliberately ugly override for the case where someone
    really means it.
*   **Two repos share this vocabulary.** ``options-live-validator`` imports it
    from a pinned tag. If a rule is an opaque dict, the two repos agree on its
    spelling and not its meaning, which is worse than disagreeing openly.

Intraday additions
------------------
The multi-week strategies this repo started with need no clock finer than a
day. 0DTE does, so five rule types arrive with it (see
``options-live-validator`` plan section 4): :class:`TimeOfDayTrigger`,
:class:`TimeOfDayExit`, :class:`UnderlyingTouch`, :class:`AbsPortfolioDelta`,
and the ``hedge_underlying`` action.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

#: Actions permitted without an explicit override. All of them either reduce or
#: neutralise exposure; none can increase short quantity in an open position.
#: Natenberg, and section 2 of the live-validator plan.
SAFE_ACTIONS = ("close", "close_partial", "force_flat", "hedge_underlying")

#: Opting out of the rule above. Named to be impossible to type by accident and
#: impossible to miss in review.
UNSAFE_OVERRIDE = "unsafe_allow_adding_to_losing_position"

MARKET_TZ_DEFAULT = "America/New_York"

#: Keys that modify a rule rather than naming one, so they do not count toward
#: the single-key requirement. ``{time_of_day: "09:45", tz: ...}`` names one
#: rule with a modifier, not two rules.
MODIFIER_KEYS = ("tz", "action")


class RuleError(ValueError):
    """A rule is malformed or forbidden. Raised at load time."""


def _parse_time(value: Any, field: str) -> dt.time:
    if isinstance(value, dt.time):
        return value
    try:
        return dt.time.fromisoformat(str(value))
    except (TypeError, ValueError):
        raise RuleError(f"{field} must be a time like '09:45', got {value!r}") from None


def _parse_tz(value: Any) -> ZoneInfo:
    try:
        return ZoneInfo(str(value))
    except (ZoneInfoNotFoundError, ValueError):
        raise RuleError(f"unknown timezone {value!r}") from None


# --- entry triggers ---------------------------------------------------------


@dataclass(frozen=True)
class TimeOfDayTrigger:
    """Open at a wall-clock time. Intraday entry, for 0DTE.

    Timezone-aware by construction and defaulting to market time: a naive
    "09:45" is ambiguous, and a strategy that entered at 09:45 UTC would be
    trading four hours before the open.
    """

    at: dt.time
    tz: ZoneInfo

    @classmethod
    def build(cls, block: Any, outer: dict[str, Any] | None = None) -> TimeOfDayTrigger:
        outer = outer or {}
        if isinstance(block, dict):
            at = block.get("time_of_day", block.get("at"))
            tz = block.get("tz") or outer.get("tz")
        else:
            at, tz = block, outer.get("tz")
        return cls(at=_parse_time(at, "time_of_day"), tz=_parse_tz(tz or MARKET_TZ_DEFAULT))


@dataclass(frozen=True)
class CalendarTrigger:
    """Open on a recurring calendar slot, e.g. every Friday."""

    every: str
    on: str | None = None

    @classmethod
    def build(cls, block: Any, outer: dict[str, Any] | None = None) -> CalendarTrigger:
        if not isinstance(block, dict) or "every" not in block:
            raise RuleError(f"calendar trigger needs an 'every', got {block!r}")
        return cls(every=str(block["every"]), on=block.get("on"))


@dataclass(frozen=True)
class NoOpenPosition:
    """Refuse to open while the strategy already holds one."""

    required: bool = True

    @classmethod
    def build(cls, block: Any, outer: dict[str, Any] | None = None) -> NoOpenPosition:
        return cls(required=bool(block) if block is not None else True)


# --- management conditions --------------------------------------------------


@dataclass(frozen=True)
class ProfitTarget:
    pct_of_credit: float

    @classmethod
    def build(cls, block: Any, outer: dict[str, Any] | None = None) -> ProfitTarget:
        pct = block.get("pct_of_credit") if isinstance(block, dict) else block
        if not isinstance(pct, int | float) or not 0 < float(pct) <= 1:
            raise RuleError(f"profit_target.pct_of_credit must be in (0, 1], got {pct!r}")
        return cls(pct_of_credit=float(pct))


@dataclass(frozen=True)
class StopLoss:
    multiple_of_credit: float

    @classmethod
    def build(cls, block: Any, outer: dict[str, Any] | None = None) -> StopLoss:
        mult = block.get("multiple_of_credit") if isinstance(block, dict) else block
        if not isinstance(mult, int | float) or float(mult) <= 0:
            raise RuleError(f"stop_loss.multiple_of_credit must be > 0, got {mult!r}")
        return cls(multiple_of_credit=float(mult))


@dataclass(frozen=True)
class DTEExit:
    at: int

    @classmethod
    def build(cls, block: Any, outer: dict[str, Any] | None = None) -> DTEExit:
        at = block.get("at") if isinstance(block, dict) else block
        if not isinstance(at, int) or at < 0:
            raise RuleError(f"dte_exit.at must be a non-negative integer, got {at!r}")
        return cls(at=at)


@dataclass(frozen=True)
class TimeOfDayExit:
    """Act at a wall-clock time. The mandatory pre-close flatten is this rule
    paired with ``force_flat``."""

    at: dt.time
    tz: ZoneInfo

    @classmethod
    def build(cls, block: Any, outer: dict[str, Any] | None = None) -> TimeOfDayExit:
        trigger = TimeOfDayTrigger.build(block, outer)
        return cls(at=trigger.at, tz=trigger.tz)


@dataclass(frozen=True)
class UnderlyingTouch:
    """Trigger when the underlying trades through a named leg's strike.

    A price level, deliberately, not a delta threshold. As tau approaches zero
    delta becomes a step function, so a per-leg delta stop fires erratically in
    the last hour - exactly when it matters most. See live-validator plan
    section 3.
    """

    leg: str

    @classmethod
    def build(cls, block: Any, outer: dict[str, Any] | None = None) -> UnderlyingTouch:
        leg = block.get("leg") if isinstance(block, dict) else block
        if not isinstance(leg, str) or not leg:
            raise RuleError(f"underlying_touch needs a leg id, got {leg!r}")
        return cls(leg=leg)


@dataclass(frozen=True)
class AbsPortfolioDelta:
    """Trigger on the position's *net* delta crossing a threshold.

    Portfolio-level rather than per-leg, which is what lets it survive tau to
    zero: individual leg deltas become meaningless near expiry, but their sum
    is still the position's real directional exposure.
    """

    threshold: float

    @classmethod
    def build(cls, block: Any, outer: dict[str, Any] | None = None) -> AbsPortfolioDelta:
        value = block.get("threshold") if isinstance(block, dict) else block
        if not isinstance(value, int | float) or float(value) <= 0:
            raise RuleError(f"abs_portfolio_delta.threshold must be > 0, got {value!r}")
        return cls(threshold=float(value))


TRIGGERS: dict[str, Any] = {"time_of_day": TimeOfDayTrigger, "calendar": CalendarTrigger}
FILTERS: dict[str, Any] = {"no_open_position": NoOpenPosition}
CONDITIONS: dict[str, Any] = {
    "profit_target": ProfitTarget,
    "stop_loss": StopLoss,
    "dte_exit": DTEExit,
    "time_of_day": TimeOfDayExit,
    "underlying_touch": UnderlyingTouch,
    "abs_portfolio_delta": AbsPortfolioDelta,
}


@dataclass(frozen=True)
class ManagementRule:
    """One condition paired with one action."""

    condition: Any
    action: str

    @property
    def kind(self) -> str:
        return type(self.condition).__name__


@dataclass(frozen=True)
class EntrySpec:
    """When a strategy may open. A trigger plus zero or more filters."""

    trigger: Any | None
    filters: tuple[Any, ...] = ()


def _build_one(block: dict[str, Any], table: dict[str, Any], kind: str, *, skip=()) -> Any:
    """Resolve a rule mapping against a registry.

    Exactly one key must name a rule; modifiers like ``tz`` and ``action`` ride
    alongside it and are handed to the builder rather than counted as rules.
    """
    keys = [k for k in block if k not in skip and k not in MODIFIER_KEYS]
    if len(keys) != 1:
        raise RuleError(
            f"{kind} must name exactly one of {sorted(table)}, got {sorted(keys) or 'nothing'}"
        )
    (name,) = keys
    try:
        cls = table[name]
    except KeyError:
        raise RuleError(f"unknown {kind} {name!r}; known: {sorted(table)}") from None
    return cls.build(block[name], block)


def build_entry(raw: Any) -> EntrySpec:
    """Parse an ``entry:`` block.

    Two shapes are accepted, because the repo already had one and 0DTE wanted
    the other. A list is the original form - a trigger and its filters, flat.
    A mapping with ``trigger``/``filters`` is the explicit form. Normalising
    both here means neither dialect leaks past the loader.
    """
    if not raw:
        return EntrySpec(trigger=None)

    if isinstance(raw, dict) and ("trigger" in raw or "filters" in raw):
        trigger = (
            _build_one(raw["trigger"], TRIGGERS, "entry trigger") if raw.get("trigger") else None
        )
        filters = tuple(_build_one(f, FILTERS, "entry filter") for f in (raw.get("filters") or []))
        return EntrySpec(trigger=trigger, filters=filters)

    if not isinstance(raw, list):
        raise RuleError(
            f"entry must be a list or a trigger/filters mapping, got {type(raw).__name__}"
        )

    trigger, filters = None, []
    for item in raw:
        if not isinstance(item, dict) or len(item) != 1:
            raise RuleError(f"entry item must be a single-key mapping, got {item!r}")
        (name,) = item
        if name in TRIGGERS:
            if trigger is not None:
                raise RuleError("entry declares more than one trigger")
            trigger = _build_one(item, TRIGGERS, "entry trigger")
        elif name in FILTERS:
            filters.append(_build_one(item, FILTERS, "entry filter"))
        else:
            raise RuleError(
                f"unknown entry rule {name!r}; triggers: {sorted(TRIGGERS)}, "
                f"filters: {sorted(FILTERS)}"
            )
    return EntrySpec(trigger=trigger, filters=tuple(filters))


def build_management(
    raw: Any, *, leg_ids: frozenset[str] = frozenset(), allow_unsafe: bool = False
) -> tuple[ManagementRule, ...]:
    """Parse a ``management:`` block, refusing anything that could add risk.

    ``leg_ids`` cross-checks rules that name a leg. A stop pointing at a leg
    that does not exist is a typo the engine would discover only when the stop
    should have fired, which is the worst possible moment to learn about it.
    """
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise RuleError(f"management must be a list, got {type(raw).__name__}")

    rules = []
    for item in raw:
        if not isinstance(item, dict):
            raise RuleError(f"management rule must be a mapping, got {item!r}")
        action = item.get("action")
        if not action:
            raise RuleError(f"management rule {item!r} declares no action")
        if action not in SAFE_ACTIONS and not allow_unsafe:
            raise RuleError(
                f"action {action!r} is not permitted. Management is restricted to "
                f"{sorted(SAFE_ACTIONS)} because anything that can increase short "
                f"quantity in an open position is adding to a loser. If that is "
                f"genuinely intended, set {UNSAFE_OVERRIDE}: true on the strategy."
            )
        condition = _build_one(item, CONDITIONS, "management condition", skip=("action",))
        if isinstance(condition, UnderlyingTouch) and leg_ids and condition.leg not in leg_ids:
            raise RuleError(
                f"underlying_touch names leg {condition.leg!r}, which this strategy "
                f"does not declare. Known legs: {sorted(leg_ids)}"
            )
        rules.append(ManagementRule(condition=condition, action=str(action)))
    return tuple(rules)
