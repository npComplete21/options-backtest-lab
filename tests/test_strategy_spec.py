"""Strategy spec loading, parameter binding, and the library files."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from obl.pricing.black_scholes import price_and_greeks
from obl.strategy import registry
from obl.strategy.selectors import DeltaStrike, OffsetStrike, SameExpiryAs, resolve
from obl.strategy.snapshot import ChainSnapshot
from obl.strategy.spec import StrategySpec, StrategySpecError

AS_OF, SPOT = dt.date(2026, 3, 9), 316.0


@pytest.fixture
def snap():
    rows = []
    for exp in (dt.date(2026, 4, 17), dt.date(2026, 5, 15)):
        tau = (exp - AS_OF).days / 365
        for k in np.arange(240.0, 400.0, 2.5):
            for right in "CP":
                g = price_and_greeks(SPOT, k, tau, 0.28, 0.04, 0.004, right)
                rows.append(
                    {
                        "expiration": exp,
                        "strike": float(k),
                        "right": right,
                        "theo": float(g["price"]),
                        "delta": float(g["delta"]),
                        "gamma": float(g["gamma"]),
                        "vega": float(g["vega"]),
                        "sigma": 0.28,
                    }
                )
    return ChainSnapshot(AS_OF, "AAPL", SPOT, pl.DataFrame(rows))


def _spec(**over):
    base = {
        "name": "t",
        "params": {"d": {"type": "float", "default": 0.16, "min": 0.01, "max": 0.5}},
        "legs": [
            {"id": "p", "right": "P", "qty": -1, "expiry": {"nth": 0}, "strike": {"delta": "{{d}}"}}
        ],
    }
    base.update(over)
    return StrategySpec(base)


# --- parameter schema ----------------------------------------------------


def test_defaults_apply_when_unspecified():
    assert _spec().bind().params["d"] == 0.16


def test_overrides_are_validated_against_bounds():
    with pytest.raises(ValidationError):
        _spec().bind(d=0.99)
    with pytest.raises(ValidationError):
        _spec().bind(d=0.0)


def test_unknown_param_type_is_rejected_at_load():
    with pytest.raises(StrategySpecError, match="unknown type"):
        _spec(params={"x": {"type": "complex"}})


def test_param_without_type_is_rejected():
    with pytest.raises(StrategySpecError, match="needs a 'type'"):
        _spec(params={"x": {"default": 1}})


def test_missing_legs_key_is_rejected():
    with pytest.raises(StrategySpecError, match="missing required key"):
        StrategySpec({"name": "t"})


def test_empty_legs_is_rejected():
    with pytest.raises(StrategySpecError, match="declares no legs"):
        _spec(legs=[])


# --- placeholder substitution --------------------------------------------


def test_whole_placeholder_keeps_native_type():
    bound = _spec().bind(d=0.25)
    assert isinstance(bound.legs[0].strike, DeltaStrike)
    assert bound.legs[0].strike.value == 0.25
    assert isinstance(bound.legs[0].strike.value, float)


def test_embedded_placeholder_supports_negation():
    """`points: "-{{wing_width}}"` is how a downside wing is expressed."""
    spec = StrategySpec(
        {
            "name": "t",
            "params": {"w": {"type": "float", "default": 5.0}},
            "legs": [
                {
                    "id": "a",
                    "right": "P",
                    "qty": -1,
                    "expiry": {"nth": 0},
                    "strike": {"delta": 0.16},
                },
                {
                    "id": "b",
                    "right": "P",
                    "qty": 1,
                    "expiry": {"same_as": "a"},
                    "strike": {"offset": {"base": "a", "points": "-{{w}}"}},
                },
            ],
        }
    )
    assert spec.bind(w=7.5).legs[1].strike.points == -7.5


def test_unknown_placeholder_is_rejected():
    with pytest.raises(StrategySpecError, match="unknown parameter"):
        _spec(
            legs=[
                {
                    "id": "p",
                    "right": "P",
                    "qty": -1,
                    "expiry": {"nth": 0},
                    "strike": {"delta": "{{nope}}"},
                }
            ]
        ).bind()


# --- leg and selector validation -----------------------------------------


def test_unknown_selector_names_are_rejected():
    with pytest.raises(StrategySpecError, match="unknown strike selector"):
        _spec(
            legs=[
                {"id": "p", "right": "P", "qty": -1, "expiry": {"nth": 0}, "strike": {"vibes": 1}}
            ]
        ).bind()


def test_bad_right_is_rejected():
    with pytest.raises(StrategySpecError, match="right must be C or P"):
        _spec(
            legs=[
                {
                    "id": "p",
                    "right": "X",
                    "qty": -1,
                    "expiry": {"nth": 0},
                    "strike": {"delta": 0.16},
                }
            ]
        ).bind()


def test_zero_quantity_leg_is_rejected():
    with pytest.raises(StrategySpecError, match="has qty 0"):
        _spec(
            legs=[
                {"id": "p", "right": "P", "qty": 0, "expiry": {"nth": 0}, "strike": {"delta": 0.16}}
            ]
        ).bind()


# --- run identity --------------------------------------------------------


def test_param_hash_is_stable_and_order_independent():
    a = registry.get("iron_condor").bind(dte_target=45, wing_width=5.0)
    b = registry.get("iron_condor").bind(wing_width=5.0, dte_target=45)
    assert a.param_hash() == b.param_hash()


def test_param_hash_changes_with_params():
    spec = registry.get("iron_condor")
    assert spec.bind(wing_width=5.0).param_hash() != spec.bind(wing_width=10.0).param_hash()


# --- the library ---------------------------------------------------------


def test_registry_lists_the_library():
    assert set(registry.all_strategies()) >= {"iron_condor", "short_strangle"}


def test_unknown_strategy_says_how_to_add_one():
    with pytest.raises(registry.UnknownStrategyError, match="no code change is needed"):
        registry.get("butterfly")


@pytest.mark.parametrize("name", ["iron_condor", "short_strangle"])
def test_library_files_load_and_bind(name):
    bound = registry.get(name).bind()
    assert bound.legs and bound.name == name


def test_iron_condor_resolves_on_a_single_stock(snap):
    """The original ask: iron condor on AAPL with specific thresholds."""
    bound = registry.get("iron_condor").bind(
        dte_target=45, short_put_delta=0.16, short_call_delta=0.16, wing_width=5.0
    )
    legs = {leg.leg_id: leg for leg in resolve(bound.legs, snap)}
    assert set(legs) == {"short_put", "long_put", "short_call", "long_call"}
    assert legs["long_put"].strike == pytest.approx(legs["short_put"].strike - 5.0)
    assert legs["long_call"].strike == pytest.approx(legs["short_call"].strike + 5.0)
    assert legs["short_put"].strike < SPOT < legs["short_call"].strike
    assert len({leg.expiration for leg in legs.values()}) == 1
    assert sum(leg.qty for leg in legs.values()) == 0  # balanced


def test_wing_width_actually_widens_the_wings(snap):
    """A parameter that does not move the position is not a parameter."""
    spec = registry.get("iron_condor")
    narrow = {leg.leg_id: leg for leg in resolve(spec.bind(wing_width=5.0).legs, snap)}
    wide = {leg.leg_id: leg for leg in resolve(spec.bind(wing_width=20.0).legs, snap)}
    assert wide["long_put"].strike < narrow["long_put"].strike
    assert wide["long_call"].strike > narrow["long_call"].strike


def test_short_strangle_is_the_condor_without_wings(snap):
    """The genericity test: two strategies, one code path, zero engine changes."""
    strangle = resolve(registry.get("short_strangle").bind().legs, snap)
    condor = resolve(registry.get("iron_condor").bind().legs, snap)
    assert len(strangle) == 2 and len(condor) == 4
    assert all(leg.qty < 0 for leg in strangle)
    shorts = {leg.strike for leg in condor if leg.qty < 0}
    assert {leg.strike for leg in strangle} == shorts


def test_iron_condor_uses_offset_and_same_expiry_selectors():
    """Guards the abstraction: wings must reference the shorts, not be
    independently selected, or the structure is not really a condor."""
    bound = registry.get("iron_condor").bind()
    by_id = {leg.id: leg for leg in bound.legs}
    assert isinstance(by_id["long_put"].strike, OffsetStrike)
    assert by_id["long_put"].strike.base == "short_put"
    assert isinstance(by_id["long_call"].expiry, SameExpiryAs)
