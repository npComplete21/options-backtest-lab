"""Strike/expiry selectors and dependency-ordered leg resolution."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from obl.pricing.black_scholes import price_and_greeks
from obl.strategy.selectors import (
    ATMStrike,
    CircularLegReferenceError,
    DeltaStrike,
    DTEExpiry,
    LegSpec,
    MoneynessStrike,
    NthExpiry,
    OffsetStrike,
    PremiumStrike,
    SameExpiryAs,
    StdDevStrike,
    resolution_order,
    resolve,
)
from obl.strategy.snapshot import ChainSnapshot, NoContractsError

AS_OF = dt.date(2026, 3, 9)
SPOT = 316.0
EXPIRIES = (dt.date(2026, 4, 17), dt.date(2026, 5, 15))


def _chain(sigma: float = 0.28, delta_override: float | None = None) -> pl.DataFrame:
    rows = []
    for exp in EXPIRIES:
        tau = (exp - AS_OF).days / 365
        for k in np.arange(260.0, 380.0, 2.5):
            for right in "CP":
                g = price_and_greeks(SPOT, k, tau, sigma, 0.04, 0.004, right)
                rows.append(
                    {
                        "expiration": exp,
                        "strike": float(k),
                        "right": right,
                        "theo": float(g["price"]),
                        "delta": (
                            delta_override if delta_override is not None else float(g["delta"])
                        ),
                        "gamma": float(g["gamma"]),
                        "vega": float(g["vega"]),
                        "sigma": sigma,
                    }
                )
    return pl.DataFrame(rows)


@pytest.fixture
def snap():
    return ChainSnapshot(as_of=AS_OF, symbol="TEST", spot=SPOT, contracts=_chain())


# --- snapshot ------------------------------------------------------------


def test_snapshot_rejects_a_chain_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        ChainSnapshot(AS_OF, "TEST", SPOT, pl.DataFrame({"strike": [1.0]}))


def test_missing_expiry_raises_rather_than_returning_empty(snap):
    """A strategy that silently opens fewer legs is no longer the strategy."""
    with pytest.raises(NoContractsError):
        snap.slice(dt.date(2030, 1, 1), "C")


# --- strike selectors ----------------------------------------------------


def test_delta_selector_picks_the_closest_absolute_delta(snap):
    sel = DeltaStrike(0.16).select(snap, EXPIRIES[0], "P", {})
    assert sel.criterion == "delta"
    assert sel.requested == 0.16
    assert sel.realized == pytest.approx(0.16, abs=0.03)
    assert sel.strike < SPOT  # a 16-delta put is out of the money


def test_delta_selector_uses_snapshot_deltas_not_recomputed_flat_vol():
    """Skew is an input, not a residual.

    Flat-vol deltas are wrong by 3-5 points on the wings, which is wider than
    the gap between adjacent strikes. Feeding the selector a chain whose
    deltas are deliberately uniform proves it reads them rather than deriving
    its own.
    """
    snap = ChainSnapshot(AS_OF, "TEST", SPOT, _chain(delta_override=0.5))
    sel = DeltaStrike(0.16).select(snap, EXPIRIES[0], "P", {})
    assert sel.realized == 0.5, "selector must read the chain's deltas verbatim"


def test_moneyness_selector(snap):
    sel = MoneynessStrike(-0.05).select(snap, EXPIRIES[0], "P", {})
    assert sel.strike == pytest.approx(SPOT * 0.95, abs=2.5)


def test_atm_selector_lands_next_to_spot(snap):
    sel = ATMStrike().select(snap, EXPIRIES[0], "C", {})
    assert abs(sel.strike - SPOT) <= 2.5


def test_premium_selector_targets_a_credit(snap):
    sel = PremiumStrike(2.0).select(snap, EXPIRIES[0], "P", {})
    assert sel.realized == pytest.approx(2.0, abs=1.0)


def test_stddev_selector_moves_with_the_tau_clock(snap):
    """Puts go down, calls go up, and both scale with sigma*sqrt(tau)."""
    put = StdDevStrike(1.0).select(snap, EXPIRIES[0], "P", {})
    call = StdDevStrike(1.0).select(snap, EXPIRIES[0], "C", {})
    assert put.strike < SPOT < call.strike
    near = StdDevStrike(1.0).select(snap, EXPIRIES[0], "C", {})
    far = StdDevStrike(1.0).select(snap, EXPIRIES[1], "C", {})
    assert far.strike > near.strike, "more time means a wider one-sigma move"


def test_offset_selector_hangs_off_another_leg(snap):
    legs = [
        LegSpec("short_put", "P", -1, NthExpiry(0), DeltaStrike(0.16)),
        LegSpec("long_put", "P", 1, SameExpiryAs("short_put"), OffsetStrike("short_put", -5.0)),
    ]
    short, long_ = resolve(legs, snap)
    assert long_.strike == pytest.approx(short.strike - 5.0)


# --- expiry selectors ----------------------------------------------------


def test_dte_selector_picks_the_nearest_target(snap):
    got = DTEExpiry(target=39, tolerance=7).select(snap, {})
    assert got == EXPIRIES[0]  # 39 days out


def test_dte_selector_raises_when_nothing_is_in_tolerance(snap):
    with pytest.raises(NoContractsError, match="available DTEs"):
        DTEExpiry(target=200, tolerance=3).select(snap, {})


def test_same_expiry_as_shares_the_referenced_leg(snap):
    legs = [
        LegSpec("a", "P", -1, DTEExpiry(39, 7), DeltaStrike(0.16)),
        LegSpec("b", "C", -1, SameExpiryAs("a"), DeltaStrike(0.16)),
    ]
    a, b = resolve(legs, snap)
    assert a.expiration == b.expiration


def test_legs_may_use_different_expiries(snap):
    """Calendars and diagonals fall out of the same abstraction for free."""
    legs = [
        LegSpec("front", "C", -1, NthExpiry(0), ATMStrike()),
        LegSpec("back", "C", 1, NthExpiry(1), ATMStrike()),
    ]
    front, back = resolve(legs, snap)
    assert front.expiration < back.expiration


# --- dependency resolution -----------------------------------------------


def test_resolution_order_puts_dependencies_first():
    legs = [
        LegSpec("wing", "P", 1, SameExpiryAs("body"), OffsetStrike("body", -5)),
        LegSpec("body", "P", -1, NthExpiry(0), DeltaStrike(0.16)),
    ]
    assert [s.id for s in resolution_order(legs)] == ["body", "wing"]


def test_circular_references_are_rejected():
    legs = [
        LegSpec("a", "P", -1, NthExpiry(0), OffsetStrike("b", -5)),
        LegSpec("b", "P", 1, NthExpiry(0), OffsetStrike("a", -5)),
    ]
    with pytest.raises(CircularLegReferenceError, match="reference cycle"):
        resolution_order(legs)


def test_unknown_leg_reference_is_rejected():
    legs = [LegSpec("a", "P", 1, NthExpiry(0), OffsetStrike("ghost", -5))]
    with pytest.raises(ValueError, match="unknown leg"):
        resolution_order(legs)


def test_duplicate_leg_ids_are_rejected():
    legs = [
        LegSpec("a", "P", -1, NthExpiry(0), ATMStrike()),
        LegSpec("a", "C", -1, NthExpiry(0), ATMStrike()),
    ]
    with pytest.raises(ValueError, match="duplicate leg ids"):
        resolution_order(legs)


def test_resolve_returns_legs_in_declared_order_not_resolution_order(snap):
    legs = [
        LegSpec("wing", "P", 1, SameExpiryAs("body"), OffsetStrike("body", -5)),
        LegSpec("body", "P", -1, NthExpiry(0), DeltaStrike(0.16)),
    ]
    assert [leg.leg_id for leg in resolve(legs, snap)] == ["wing", "body"]


def test_every_selection_records_requested_and_realized(snap):
    """Silent snapping is how a backtest stops describing a tradeable strategy."""
    legs = [LegSpec("p", "P", -1, NthExpiry(0), DeltaStrike(0.16))]
    (leg,) = resolve(legs, snap)
    assert leg.selection.requested == 0.16
    assert leg.selection.realized != leg.selection.requested  # landed on a listed strike
    assert leg.selection.miss > 0


def test_resolved_legs_carry_expiry_identity(snap):
    """Vega is not additive across expirations, so aggregation must be able to
    weight by expiry later. That requires expiry on the leg, not just a strike."""
    legs = [
        LegSpec("front", "C", -1, NthExpiry(0), ATMStrike()),
        LegSpec("back", "C", 1, NthExpiry(1), ATMStrike()),
    ]
    assert len({leg.expiration for leg in resolve(legs, snap)}) == 2
