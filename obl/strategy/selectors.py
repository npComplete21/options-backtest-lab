"""Strike and expiry selectors - the shared vocabulary of every strategy.

Every option strategy reduces to the same primitive: select N legs, each a
``(right, expiry selector, strike selector, quantity)``. Strategies differ in
how many legs they have and how strikes are chosen, not in the mechanics of
choosing them. That is what lets most strategies be data rather than code.

``OffsetStrike`` is what makes this genuinely general rather than a pile of
special cases: an iron condor's wings are *defined relative to* its shorts, so
leg resolution is a small dependency DAG rather than an independent lookup per
leg. Two differing expiry selectors in one position give calendars and
diagonals for free.

Every selection records what was **requested** and what was **realized**. A
16-delta request lands on a listed strike that might be 17.3 delta, and silent
snapping is how a backtest stops describing a tradeable strategy.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import polars as pl

from obl.strategy.snapshot import ChainSnapshot, NoContractsError


@dataclass(frozen=True)
class Selection:
    """A chosen strike, with the miss against what was asked for."""

    strike: float
    criterion: str
    requested: float
    realized: float

    @property
    def miss(self) -> float:
        return abs(self.realized - self.requested)


@dataclass(frozen=True)
class ResolvedLeg:
    leg_id: str
    right: str
    qty: int
    expiration: dt.date
    strike: float
    selection: Selection

    @property
    def is_short(self) -> bool:
        return self.qty < 0


# --- expiry selectors ----------------------------------------------------


class ExpirySelector(Protocol):
    def depends_on(self) -> set[str]: ...
    def select(self, snap: ChainSnapshot, resolved: dict[str, ResolvedLeg]) -> dt.date: ...


@dataclass(frozen=True)
class DTEExpiry:
    """Expiry closest to ``target`` days out, within ``tolerance``."""

    target: int
    tolerance: int = 7

    def depends_on(self) -> set[str]:
        return set()

    def select(self, snap: ChainSnapshot, resolved) -> dt.date:
        candidates = [
            e for e in snap.expirations() if abs(snap.dte(e) - self.target) <= self.tolerance
        ]
        if not candidates:
            available = sorted({snap.dte(e) for e in snap.expirations()})
            raise NoContractsError(
                f"no expiry within {self.tolerance}d of {self.target} DTE on {snap.as_of}; "
                f"available DTEs: {available[:12]}"
            )
        return min(candidates, key=lambda e: abs(snap.dte(e) - self.target))


@dataclass(frozen=True)
class NthExpiry:
    """The ``n``-th expiry from the front (0-indexed)."""

    n: int = 0

    def depends_on(self) -> set[str]:
        return set()

    def select(self, snap: ChainSnapshot, resolved) -> dt.date:
        exps = snap.expirations()
        if self.n >= len(exps):
            raise NoContractsError(f"expiry index {self.n} but only {len(exps)} listed")
        return exps[self.n]


@dataclass(frozen=True)
class SameExpiryAs:
    """Share another leg's expiry. Most multi-leg strategies use this."""

    leg: str

    def depends_on(self) -> set[str]:
        return {self.leg}

    def select(self, snap: ChainSnapshot, resolved) -> dt.date:
        return resolved[self.leg].expiration


# --- strike selectors ----------------------------------------------------


class StrikeSelector(Protocol):
    def depends_on(self) -> set[str]: ...
    def select(self, snap, expiry, right, resolved) -> Selection: ...


def _nearest(frame: pl.DataFrame, column: str, target: float) -> tuple[float, float]:
    """Row whose ``column`` is closest to ``target``; returns (strike, value)."""
    values = frame[column].to_numpy()
    idx = int(np.abs(values - target).argmin())
    return float(frame["strike"][idx]), float(values[idx])


@dataclass(frozen=True)
class DeltaStrike:
    """Strike whose absolute delta is closest to ``value``.

    Deltas come from the snapshot, which sources them from the volatility
    surface. They must not be recomputed here from a flat vol: that error is
    3-5 delta points on the wings, which is larger than the gap between
    adjacent strike selections.
    """

    value: float

    def depends_on(self) -> set[str]:
        return set()

    def select(self, snap, expiry, right, resolved) -> Selection:
        frame = snap.slice(expiry, right).with_columns(pl.col("delta").abs().alias("_absdelta"))
        strike, realized = _nearest(frame, "_absdelta", abs(self.value))
        return Selection(strike, "delta", abs(self.value), realized)


@dataclass(frozen=True)
class MoneynessStrike:
    """Strike closest to ``spot * (1 + pct)``. ``pct`` is signed."""

    pct: float

    def depends_on(self) -> set[str]:
        return set()

    def select(self, snap, expiry, right, resolved) -> Selection:
        target = snap.spot * (1.0 + self.pct)
        frame = snap.slice(expiry, right)
        strike, realized = _nearest(frame, "strike", target)
        return Selection(strike, "moneyness", target, realized)


@dataclass(frozen=True)
class StdDevStrike:
    """Strike ``n`` standard deviations from spot, in log space.

    Uses ``sigma * sqrt(tau)``, so this selector is directly sensitive to the
    tau clock - the reason the clock is a recorded input rather than a
    constant. Direction follows the right: puts down, calls up.
    """

    n: float

    def depends_on(self) -> set[str]:
        return set()

    def select(self, snap, expiry, right, resolved) -> Selection:
        sigma = snap.atm_sigma(expiry)
        move = sigma * np.sqrt(snap.tau(expiry))
        sign = -1.0 if right.upper() == "P" else 1.0
        target = snap.spot * float(np.exp(sign * abs(self.n) * move))
        frame = snap.slice(expiry, right)
        strike, realized = _nearest(frame, "strike", target)
        return Selection(strike, "stddev", target, realized)


@dataclass(frozen=True)
class ATMStrike:
    def depends_on(self) -> set[str]:
        return set()

    def select(self, snap, expiry, right, resolved) -> Selection:
        frame = snap.slice(expiry, right)
        strike, realized = _nearest(frame, "strike", snap.spot)
        return Selection(strike, "atm", snap.spot, realized)


@dataclass(frozen=True)
class PremiumStrike:
    """Strike whose theoretical value is closest to ``target``."""

    target: float

    def depends_on(self) -> set[str]:
        return set()

    def select(self, snap, expiry, right, resolved) -> Selection:
        frame = snap.slice(expiry, right)
        strike, realized = _nearest(frame, "theo", self.target)
        return Selection(strike, "premium", self.target, realized)


@dataclass(frozen=True)
class OffsetStrike:
    """Strike a fixed distance from another leg's strike.

    This is how defined-risk wings are expressed - an iron condor's long put
    sits ``width`` below its short put - and it is why leg resolution has to
    be ordered rather than independent.
    """

    base: str
    points: float

    def depends_on(self) -> set[str]:
        return {self.base}

    def select(self, snap, expiry, right, resolved) -> Selection:
        target = resolved[self.base].strike + self.points
        frame = snap.slice(expiry, right)
        strike, realized = _nearest(frame, "strike", target)
        return Selection(strike, "offset", target, realized)


# --- leg resolution ------------------------------------------------------


@dataclass(frozen=True)
class LegSpec:
    id: str
    right: str
    qty: int
    expiry: ExpirySelector
    strike: StrikeSelector

    def depends_on(self) -> set[str]:
        return self.expiry.depends_on() | self.strike.depends_on()


class CircularLegReferenceError(ValueError):
    """Legs reference each other in a cycle, so no resolution order exists."""


def resolution_order(specs: list[LegSpec]) -> list[LegSpec]:
    """Topologically sort legs so referenced legs resolve first."""
    by_id = {s.id: s for s in specs}
    if len(by_id) != len(specs):
        dupes = [s.id for s in specs if list(x.id for x in specs).count(s.id) > 1]
        raise ValueError(f"duplicate leg ids: {sorted(set(dupes))}")

    for spec in specs:
        unknown = spec.depends_on() - set(by_id)
        if unknown:
            raise ValueError(f"leg {spec.id!r} references unknown leg(s): {sorted(unknown)}")

    ordered: list[LegSpec] = []
    done: set[str] = set()
    visiting: set[str] = set()

    def visit(spec: LegSpec) -> None:
        if spec.id in done:
            return
        if spec.id in visiting:
            raise CircularLegReferenceError(
                f"leg {spec.id!r} is part of a reference cycle; wings must ultimately "
                "hang off a leg that stands on its own"
            )
        visiting.add(spec.id)
        for dep in sorted(spec.depends_on()):
            visit(by_id[dep])
        visiting.discard(spec.id)
        done.add(spec.id)
        ordered.append(spec)

    for spec in specs:
        visit(spec)
    return ordered


def resolve(specs: list[LegSpec], snap: ChainSnapshot) -> list[ResolvedLeg]:
    """Resolve every leg against a chain snapshot, honouring dependencies."""
    resolved: dict[str, ResolvedLeg] = {}
    for spec in resolution_order(specs):
        expiry = spec.expiry.select(snap, resolved)
        selection = spec.strike.select(snap, expiry, spec.right, resolved)
        resolved[spec.id] = ResolvedLeg(
            leg_id=spec.id,
            right=spec.right.upper(),
            qty=spec.qty,
            expiration=expiry,
            strike=selection.strike,
            selection=selection,
        )
    return [resolved[s.id] for s in specs]
