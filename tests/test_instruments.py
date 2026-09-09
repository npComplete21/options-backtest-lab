"""Instrument registry and contract-spec behaviour."""

from __future__ import annotations

import datetime as dt

import pytest

from obl.instruments import registry
from obl.instruments.base import (
    AssetClass,
    ContractSpec,
    ExpirationRule,
    ExpirationType,
    Instrument,
    VolModelSpec,
)


def test_registry_resolves_etf_and_single_stock():
    """Phase 2 gate: both instrument classes resolve from config."""
    qqq, aapl = registry.get("QQQ"), registry.get("AAPL")
    assert qqq.asset_class is AssetClass.ETF
    assert aapl.asset_class is AssetClass.SINGLE_STOCK


def test_registry_is_case_insensitive():
    assert registry.get("qqq").symbol == registry.get("QQQ").symbol


def test_unknown_symbol_is_fatal_and_says_what_is_missing():
    """Defaulting an unknown symbol would produce a clean run describing nothing."""
    with pytest.raises(registry.UnknownInstrumentError, match="not in the instrument registry"):
        registry.get("NVDA")


def test_etf_is_observed_iv_and_single_stock_is_not():
    """The confidence split is the whole point of the vol model spec."""
    assert registry.get("QQQ").vol_model.confidence == "observed_iv"
    assert registry.get("AAPL").vol_model.confidence == "directional_only"


def test_iwm_falls_back_because_rvx_is_unavailable():
    """^RVX returns empty from yfinance, so IWM cannot claim observed IV."""
    iwm = registry.get("IWM")
    assert iwm.vol_model.kind == "single_stock"
    assert iwm.vol_model.confidence == "directional_only"


def test_vol_index_model_requires_an_index_symbol():
    with pytest.raises(ValueError, match="requires a vol_index symbol"):
        VolModelSpec(kind="vol_index")


def test_single_stock_must_declare_earnings_sensitivity():
    with pytest.raises(ValueError, match="earnings_sensitive"):
        Instrument(
            symbol="TEST",
            asset_class=AssetClass.SINGLE_STOCK,
            contract_spec=ContractSpec(strike_increment=1.0),
            vol_model=VolModelSpec(kind="single_stock"),
            expirations=[ExpirationRule(type=ExpirationType.MONTHLY, available_from="2000-01-01")],
        )


@pytest.mark.parametrize(
    "increment,raw,expected",
    [(1.0, 718.36, 718.0), (2.5, 316.22, 315.0), (5.0, 718.36, 720.0), (5.0, 716.2, 715.0)],
)
def test_snap_strike_lands_on_the_listed_grid(increment, raw, expected):
    assert ContractSpec(strike_increment=increment).snap_strike(raw) == expected


def test_expiration_availability_is_time_dependent():
    """The core guard: QQQ had no Tue/Wed/Thu expirations in 2015."""
    qqq = registry.get("QQQ")
    in_2015 = set(qqq.expirations_available_on(dt.date(2015, 6, 1)))
    today = set(qqq.expirations_available_on(dt.date(2026, 9, 8)))

    assert in_2015 == {ExpirationType.MONTHLY, ExpirationType.WEEKLY_FRI}
    assert ExpirationType.WEEKLY_TUE in today
    assert in_2015 < today


def test_no_expirations_before_the_first_listing_date():
    assert registry.get("QQQ").expirations_available_on(dt.date(1990, 1, 1)) == []


def test_unverified_rules_are_surfaced_not_hidden():
    """Estimated listing dates must be reportable into run metadata."""
    unverified = registry.get("QQQ").unverified_expiration_rules()
    assert unverified, "QQQ has estimated weekly listing dates"
    assert all(not r.verified for r in unverified)
