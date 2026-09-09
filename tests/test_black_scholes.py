"""Validation harness for the pricing module.

``CLAUDE.md`` requires pricing output to be validated against put-call parity
and known real prices before any backtest is allowed to trust it. These are
checks 1-4 of ``docs/IMPLEMENTATION_PLAN.md`` section 9; check 6 (against
live market quotes) needs network access and lives separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from obl.pricing.black_scholes import (
    d1_d2,
    forward,
    implied_vol,
    price,
    price_and_greeks,
    to_trader_greeks,
)

# A grid spanning deep ITM to deep OTM, one week to two years, 5% to 90% vol,
# including a negative rate and a zero-dividend case.
SPOTS = [50.0, 100.0, 250.0]
STRIKES = [60.0, 95.0, 100.0, 105.0, 400.0]
TAUS = [7 / 365, 0.25, 1.0, 2.0]
SIGMAS = [0.05, 0.20, 0.90]
RATES = [-0.005, 0.0, 0.05]
DIVS = [0.0, 0.02]


def _grid():
    for S in SPOTS:
        for K in STRIKES:
            for tau in TAUS:
                for sigma in SIGMAS:
                    for r in RATES:
                        for q in DIVS:
                            yield S, K, tau, sigma, r, q


# --- 1. Put-call parity -------------------------------------------------


def test_put_call_parity():
    """C - P == S*exp(-q*tau) - K*exp(-r*tau) across the whole grid."""
    for S, K, tau, sigma, r, q in _grid():
        c = float(price(S, K, tau, sigma, r, q, "C"))
        p = float(price(S, K, tau, sigma, r, q, "P"))
        expected = S * np.exp(-q * tau) - K * np.exp(-r * tau)
        assert abs((c - p) - expected) < 1e-10, f"parity broke at {(S, K, tau, sigma, r, q)}"


@pytest.mark.parametrize("tau,sigma", [(0.0, 0.2), (-1.0, 0.2), (0.5, 0.0), (0.0, 0.0)])
def test_put_call_parity_degenerate(tau, sigma):
    """Parity must survive the tau<=0 and sigma<=0 branches, not just the smooth one."""
    for S, K, _, _, r, q in _grid():
        c = float(price(S, K, tau, sigma, r, q, "C"))
        p = float(price(S, K, tau, sigma, r, q, "P"))
        t = max(tau, 0.0)
        expected = S * np.exp(-q * t) - K * np.exp(-r * t)
        assert abs((c - p) - expected) < 1e-10


# --- 2. Greeks vs. finite differences ------------------------------------


def test_delta_matches_finite_difference():
    h = 1e-5
    for S, K, tau, sigma, r, q in _grid():
        for right in ("C", "P"):
            up = float(price(S + h, K, tau, sigma, r, q, right))
            dn = float(price(S - h, K, tau, sigma, r, q, right))
            analytic = float(price_and_greeks(S, K, tau, sigma, r, q, right)["delta"])
            assert analytic == pytest.approx((up - dn) / (2 * h), abs=1e-5)


def test_gamma_matches_finite_difference():
    h = 1e-3
    for S, K, tau, sigma, r, q in _grid():
        mid = float(price(S, K, tau, sigma, r, q, "C"))
        up = float(price(S + h, K, tau, sigma, r, q, "C"))
        dn = float(price(S - h, K, tau, sigma, r, q, "C"))
        analytic = float(price_and_greeks(S, K, tau, sigma, r, q, "C")["gamma"])
        assert analytic == pytest.approx((up - 2 * mid + dn) / h**2, abs=1e-4)


def test_vega_matches_finite_difference():
    h = 1e-6
    for S, K, tau, sigma, r, q in _grid():
        up = float(price(S, K, tau, sigma + h, r, q, "C"))
        dn = float(price(S, K, tau, sigma - h, r, q, "C"))
        analytic = float(price_and_greeks(S, K, tau, sigma, r, q, "C")["vega"])
        assert analytic == pytest.approx((up - dn) / (2 * h), rel=1e-4, abs=1e-4)


def test_theta_matches_finite_difference():
    """theta = -dV/dtau, so shrinking tau must move value the other way."""
    h = 1e-6
    for S, K, tau, sigma, r, q in _grid():
        for right in ("C", "P"):
            up = float(price(S, K, tau + h, sigma, r, q, right))
            dn = float(price(S, K, tau - h, sigma, r, q, right))
            analytic = float(price_and_greeks(S, K, tau, sigma, r, q, right)["theta"])
            assert analytic == pytest.approx(-(up - dn) / (2 * h), rel=1e-3, abs=1e-3)


def test_rho_matches_finite_difference():
    h = 1e-7
    for S, K, tau, sigma, r, q in _grid():
        for right in ("C", "P"):
            up = float(price(S, K, tau, sigma, r + h, q, right))
            dn = float(price(S, K, tau, sigma, r - h, q, right))
            analytic = float(price_and_greeks(S, K, tau, sigma, r, q, right)["rho"])
            assert analytic == pytest.approx((up - dn) / (2 * h), rel=1e-4, abs=1e-4)


# --- 3. Known reference values -------------------------------------------


def test_hull_worked_example():
    """Hull, *Options, Futures and Other Derivatives*: S=42, K=40, r=10%,
    sigma=20%, T=0.5 -> call 4.76, put 0.81."""
    assert float(price(42, 40, 0.5, 0.20, 0.10, 0.0, "C")) == pytest.approx(4.76, abs=0.005)
    assert float(price(42, 40, 0.5, 0.20, 0.10, 0.0, "P")) == pytest.approx(0.81, abs=0.005)


def test_atm_forward_call_put_equal():
    """Struck at the forward, a call and a put are worth the same (parity, K=F)."""
    S, tau, sigma, r, q = 100.0, 1.0, 0.25, 0.04, 0.01
    K = float(forward(S, tau, r, q))
    c = float(price(S, K, tau, sigma, r, q, "C"))
    p = float(price(S, K, tau, sigma, r, q, "P"))
    assert c == pytest.approx(p, abs=1e-10)


# --- 4. Monotonicity and no-arbitrage bounds ------------------------------


def test_price_increases_with_volatility():
    for S, K, tau, _, r, q in _grid():
        vals = [float(price(S, K, tau, s, r, q, "C")) for s in (0.05, 0.15, 0.30, 0.60)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:], strict=False))


def test_call_decreases_and_put_increases_in_strike():
    S, tau, sigma, r, q = 100.0, 0.5, 0.25, 0.03, 0.01
    strikes = np.arange(60.0, 141.0, 5.0)
    calls = price(S, strikes, tau, sigma, r, q, "C")
    puts = price(S, strikes, tau, sigma, r, q, "P")
    assert np.all(np.diff(calls) <= 1e-12)
    assert np.all(np.diff(puts) >= -1e-12)


def test_no_arbitrage_bounds():
    """max(Se^-qt - Ke^-rt, 0) <= C <= Se^-qt, and the put mirror."""
    for S, K, tau, sigma, r, q in _grid():
        disc_s, disc_k = S * np.exp(-q * tau), K * np.exp(-r * tau)
        c = float(price(S, K, tau, sigma, r, q, "C"))
        p = float(price(S, K, tau, sigma, r, q, "P"))
        assert max(disc_s - disc_k, 0.0) - 1e-10 <= c <= disc_s + 1e-10
        assert max(disc_k - disc_s, 0.0) - 1e-10 <= p <= disc_k + 1e-10


def test_call_convex_in_strike():
    """Butterfly no-arbitrage: call prices must be convex in strike."""
    S, tau, sigma, r, q = 100.0, 0.5, 0.25, 0.03, 0.01
    strikes = np.arange(70.0, 131.0, 1.0)
    calls = price(S, strikes, tau, sigma, r, q, "C")
    assert np.all(np.diff(calls, 2) >= -1e-10)


# --- 5. Degenerate inputs -------------------------------------------------


@pytest.mark.parametrize("right,expected", [("C", 10.0), ("P", 0.0)])
def test_expiry_is_intrinsic(right, expected):
    assert float(price(110.0, 100.0, 0.0, 0.25, 0.05, 0.01, right)) == pytest.approx(expected)


def test_zero_vol_is_discounted_forward_intrinsic():
    S, K, tau, r, q = 100.0, 90.0, 1.0, 0.05, 0.02
    expected = S * np.exp(-q * tau) - K * np.exp(-r * tau)
    assert float(price(S, K, tau, 0.0, r, q, "C")) == pytest.approx(expected, abs=1e-12)


def test_degenerate_greeks_are_finite_and_sane():
    """A backtest walks to expiry every cycle - this path must never produce NaN."""
    for tau, sigma in [(0.0, 0.25), (0.5, 0.0), (0.0, 0.0)]:
        g = price_and_greeks([110.0, 90.0], 100.0, tau, sigma, 0.05, 0.01, "C")
        for name, arr in g.items():
            assert np.all(np.isfinite(arr)), f"{name} not finite at tau={tau}, sigma={sigma}"
        assert np.all(g["gamma"] == 0.0)
        assert np.all(g["vega"] == 0.0)


def test_d1_d2_nan_only_in_degenerate_cells():
    d1, d2 = d1_d2([100.0, 100.0], 100.0, [0.5, 0.0], 0.25, 0.03, 0.0)
    assert np.isfinite(d1[0]) and np.isfinite(d2[0])
    assert np.isnan(d1[1]) and np.isnan(d2[1])


# --- 6. Vectorization -----------------------------------------------------


def test_broadcasting_matches_scalar_loop():
    strikes = np.array([90.0, 100.0, 110.0])
    taus = np.array([[0.25], [1.0]])
    rights = np.array(["C", "P", "C"])
    got = price_and_greeks(100.0, strikes, taus, 0.2, 0.03, 0.01, rights)
    assert got["price"].shape == (2, 3)
    for i, tau in enumerate([0.25, 1.0]):
        for j, (K, right) in enumerate(zip(strikes, rights, strict=True)):
            assert got["price"][i, j] == pytest.approx(
                float(price(100.0, K, tau, 0.2, 0.03, 0.01, right))
            )


def test_lowercase_rights_accepted_and_bad_rights_rejected():
    assert float(price(100, 100, 1.0, 0.2, 0.0, 0.0, "c")) == pytest.approx(
        float(price(100, 100, 1.0, 0.2, 0.0, 0.0, "C"))
    )
    with pytest.raises(ValueError, match="must be 'C' or 'P'"):
        price(100, 100, 1.0, 0.2, 0.0, 0.0, "X")


# --- 7. Implied vol round-trip -------------------------------------------


def test_implied_vol_recovers_input():
    """Round-trip wherever the price carries volatility information.

    Identifiability is a property of vega, not of price size: a deep ITM put
    can be worth $10 and still price bit-identically across a wide vol range.
    """
    checked = 0
    for S, K, tau, sigma, r, q in _grid():
        for right in ("C", "P"):
            theo = float(price(S, K, tau, sigma, r, q, right))
            vega = float(price_and_greeks(S, K, tau, sigma, r, q, right)["vega"])
            if vega < 1e-6:  # price does not respond to vol - nothing to recover
                continue
            got = float(implied_vol(theo, S, K, tau, r, q, right))
            assert got == pytest.approx(sigma, abs=1e-4), (S, K, tau, sigma, r, q, right)
            checked += 1
    assert checked > 500, "grid should exercise many identifiable cases"


def test_implied_vol_returns_nan_outside_bounds():
    # Above the max possible call value - no root exists.
    assert np.isnan(float(implied_vol(999.0, 100.0, 100.0, 1.0, 0.03, 0.0, "C")))
    assert np.isnan(float(implied_vol(1.0, 100.0, 100.0, 0.0, 0.03, 0.0, "C")))


def test_implied_vol_nan_when_price_carries_no_vol_information():
    """A 7-day deep ITM put is pure intrinsic to machine precision.

    Returning the bracket edge (1e-6) here would look like a real measurement
    and would silently poison any surface calibrated from real chain data,
    where deep ITM contracts are routine.
    """
    S, K, tau, r, q = 50.0, 60.0, 7 / 365, 0.0, 0.0
    theo = float(price(S, K, tau, 0.05, r, q, "P"))
    assert float(price(S, K, tau, 0.01, r, q, "P")) == theo  # bit-identical
    assert np.isnan(float(implied_vol(theo, S, K, tau, r, q, "P")))


# --- 8. Convention helpers ------------------------------------------------


def test_trader_greek_conversions():
    raw = price_and_greeks(100.0, 100.0, 0.5, 0.25, 0.03, 0.01, "C")
    conv = to_trader_greeks(raw)
    assert conv["vega"] == pytest.approx(raw["vega"] / 100.0)
    assert conv["theta"] == pytest.approx(raw["theta"] / 365.0)
    assert conv["rho"] == pytest.approx(raw["rho"] / 10000.0)
    assert conv["delta"] == pytest.approx(raw["delta"])
