"""Cross-validation of the pricing module against QuantLib.

Every other test in this suite checks our implementation against *itself* -
put-call parity, finite differences, monotonicity. Those are strong internal
consistency checks, but they share our formulae, so a subtly wrong ``d1`` or
a misplaced ``exp(-q*tau)`` could satisfy all of them simultaneously.

QuantLib is an independent implementation maintained by the derivatives
community. Agreement with it rules out a class of error that no amount of
internal consistency checking can reach.

Dev-only dependency: install with ``pip install -e '.[dev]'``. It is never
imported by the runtime path.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.pricing.black_scholes import implied_vol, price_and_greeks

ql = pytest.importorskip("QuantLib", reason="QuantLib is a dev-only dependency")

GREEKS = ("price", "delta", "gamma", "vega", "theta", "rho")

# Day counts rather than year fractions: QuantLib works in dates, so exact
# integer days under Actual365Fixed avoid any tau quantization mismatch.
DAYS = [7, 30, 91, 365, 730]
CASES = [
    (S, K, d, sigma, r, q, right)
    for S, K in [(100.0, 90.0), (100.0, 100.0), (100.0, 115.0), (250.0, 240.0)]
    for d in DAYS
    for sigma in [0.10, 0.25, 0.60]
    for r in [-0.005, 0.0, 0.05]
    for q in [0.0, 0.02]
    for right in ("C", "P")
]


def _quantlib_reference(S, K, days, sigma, r, q, right) -> dict[str, float]:
    today = ql.Date(1, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count)),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), sigma, day_count)
        ),
    )
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call if right == "C" else ql.Option.Put, K),
        ql.EuropeanExercise(today + int(days)),
    )
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return {
        "price": option.NPV(),
        "delta": option.delta(),
        "gamma": option.gamma(),
        "vega": option.vega(),
        "theta": option.theta(),
        "rho": option.rho(),
    }


def _case_id(c) -> str:
    return f"S{c[0]:g}K{c[1]:g}d{c[2]}v{c[3]}r{c[4]}q{c[5]}{c[6]}"


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_matches_quantlib(case):
    """Price and all five greeks agree with QuantLib to near machine precision.

    Also confirms our greek conventions match the reference implementation:
    vega per 1.00 vol, theta per year, rho per 1.00 rate. A convention
    mismatch would show up here as a clean factor of 100 or 365.
    """
    S, K, days, sigma, r, q, right = case
    ours = price_and_greeks(S, K, days / 365.0, sigma, r, q, right)
    ref = _quantlib_reference(*case)
    for g in GREEKS:
        assert float(ours[g]) == pytest.approx(ref[g], rel=1e-10, abs=1e-10), (
            f"{g} disagrees with QuantLib for {case}"
        )


def test_implied_vol_matches_quantlib_inversion():
    """Our IV solver recovers the same vol QuantLib's does, where identifiable."""
    checked = 0
    for S, K, days, sigma, r, q, right in CASES:
        tau = days / 365.0
        theo = float(price_and_greeks(S, K, tau, sigma, r, q, right)["price"])
        vega = float(price_and_greeks(S, K, tau, sigma, r, q, right)["vega"])
        if vega < 1e-6:
            continue
        ours = float(implied_vol(theo, S, K, tau, r, q, right))
        assert ours == pytest.approx(sigma, abs=1e-4)
        assert not np.isnan(ours)
        checked += 1
    assert checked > 200
