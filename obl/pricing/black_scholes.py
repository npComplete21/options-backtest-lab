"""Generalized Black-Scholes-Merton pricing and greeks.

This is the single shared pricing module for the project. Per ``CLAUDE.md``,
no other module may reimplement Black-Scholes math.

Deliberately dependency-light: pure vectorized NumPy/SciPy with no I/O, no
config, and no Spark. That keeps it trivially unit-testable and lets the same
code run unchanged in a notebook, in a pandas prototype, and inside a Spark
UDF.

Model
-----
Generalized BSM with a continuous dividend yield ``q`` (cost of carry
``b = r - q``). SPY/QQQ/IWM and single stocks all pay meaningful dividends;
ignoring ``q`` breaks put-call parity and biases calls against puts.

**European exercise.** US equity and ETF options are American in reality.
For the OTM premium-selling strategies this repo targets first, the
approximation is mild, but early exercise on ITM calls before an ex-dividend
date is a real effect. See ``docs/IMPLEMENTATION_PLAN.md`` section 3 - this
is a known limitation, not an oversight, and an American pricer is an
additive extension behind the same interface.

Conventions
-----------
All greeks are returned in **raw per-unit terms**, not scaled to trader
conventions, so there is exactly one unambiguous definition in the codebase:

===========  ==========================================  ====================
Greek        Raw unit returned here                      Trader convention
===========  ==========================================  ====================
``delta``    per 1.00 change in spot                     same
``gamma``    per 1.00^2 change in spot                   same
``vega``     per 1.00 (=100 vol points) change in vol    ``vega / 100``
``theta``    per 1.0 year                                ``theta / 365``
``rho``      per 1.00 (=10000bp) change in rate          ``rho / 100``
===========  ==========================================  ====================

Use :func:`to_trader_greeks` to convert. Callers must not apply these
divisors ad hoc.

Degenerate inputs
-----------------
``tau <= 0`` (expiry) and ``sigma <= 0`` are handled explicitly rather than
producing ``NaN`` from a division by ``sigma * sqrt(tau)``. A backtest walks
a position to expiry on every cycle, so this path is hot, not exotic: both
collapse to the discounted-forward intrinsic value with indicator deltas and
zero gamma/vega.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

__all__ = [
    "CALL",
    "PUT",
    "d1_d2",
    "forward",
    "implied_vol",
    "price",
    "price_and_greeks",
    "to_trader_greeks",
]

CALL = "C"
PUT = "P"

# Bracket for implied-vol root-finding. 1e-6 to 500% annualized comfortably
# spans anything a listed option can quote at.
_IV_LOW = 1e-6
_IV_HIGH = 5.0

# Identifiability guard for implied_vol: probe volatility by one basis point
# and require the price to move by more than a few ulps of rounding noise.
_IV_PROBE = 1e-4
_IDENTIFIABILITY_EPS = 8 * np.finfo(float).eps


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _as_call_mask(right) -> np.ndarray:
    """Normalize a right specifier to a boolean 'is a call' mask."""
    arr = np.asarray(right)
    if arr.dtype.kind in "US":
        upper = np.char.upper(arr.astype(str))
        is_call = upper == CALL
        if not np.all(is_call | (upper == PUT)):
            bad = np.unique(upper[~(is_call | (upper == PUT))])
            raise ValueError(f"right must be 'C' or 'P', got {bad.tolist()}")
        return is_call
    return arr.astype(bool)


def forward(S, tau, r, q):
    """Forward price ``F = S * exp((r - q) * tau)``."""
    S, tau, r, q = np.broadcast_arrays(*map(np.asarray, (S, tau, r, q)))
    return S * np.exp((r - q) * tau)


def d1_d2(S, K, tau, sigma, r, q):
    """Black-Scholes ``d1`` and ``d2``.

    Returns ``NaN`` in degenerate cells (``tau <= 0`` or ``sigma <= 0``);
    callers are expected to mask those out rather than use the values.
    """
    S, K, tau, sigma, r, q = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, tau, sigma, r, q))
    )
    ok = (tau > 0) & (sigma > 0) & (S > 0) & (K > 0)
    # Guard the divisor so the masked-out cells cannot raise or warn.
    vol_sqrt_tau = np.where(ok, sigma * np.sqrt(np.where(ok, tau, 1.0)), 1.0)
    log_sk = np.log(np.where(ok, S / np.where(K > 0, K, 1.0), 1.0))
    d1 = (log_sk + (r - q + 0.5 * sigma**2) * tau) / vol_sqrt_tau
    d2 = d1 - vol_sqrt_tau
    return np.where(ok, d1, np.nan), np.where(ok, d2, np.nan)


def _deterministic(S, K, tau, r, q, is_call):
    """Value and greeks when ``tau <= 0`` or ``sigma <= 0``.

    With no volatility the forward is known, so the option is just a
    discounted intrinsic claim: ``max(F - K, 0) * exp(-r*tau)`` for a call,
    which simplifies to ``S*exp(-q*tau) - K*exp(-r*tau)`` when in the money.
    This is the continuous limit of the general formulae with
    ``N(d1), N(d2) -> 1{F > K}``, so parity and greek continuity are preserved.

    ``tau < 0`` is clamped to zero: an expired option is expired, and must not
    be discounted backwards into a larger value.
    """
    tau = np.maximum(tau, 0.0)
    disc_s = S * np.exp(-q * tau)
    disc_k = K * np.exp(-r * tau)
    itm = np.where(is_call, disc_s > disc_k, disc_s < disc_k)

    sign = np.where(is_call, 1.0, -1.0)
    value = np.where(itm, sign * (disc_s - disc_k), 0.0)
    delta = np.where(itm, sign * np.exp(-q * tau), 0.0)
    theta = np.where(itm, q * disc_s - r * disc_k, 0.0) * sign
    rho = np.where(itm, sign * K * tau * np.exp(-r * tau), 0.0)
    zero = np.zeros_like(value)
    return value, delta, zero, zero, theta, rho


def price_and_greeks(S, K, tau, sigma, r, q=0.0, right=CALL) -> dict[str, np.ndarray]:
    """Price and all five greeks in one pass.

    Computes ``d1``/``d2`` and the normal CDF/PDF once and reuses them, which
    matters because chain generation prices hundreds of thousands of contracts
    and needs every greek for each.

    Parameters
    ----------
    S, K, tau, sigma, r, q
        Spot, strike, year fraction to expiry, annualized volatility,
        continuously-compounded risk-free rate, continuous dividend yield.
        Broadcast against each other, so any may be scalar or array.
    right
        ``'C'``/``'P'`` (arrays accepted), or a boolean 'is a call' mask.

    Returns
    -------
    dict
        ``price``, ``delta``, ``gamma``, ``vega``, ``theta``, ``rho`` -
        all in the raw units documented in the module docstring.
    """
    S, K, tau, sigma, r, q = (np.asarray(x, dtype=float) for x in (S, K, tau, sigma, r, q))
    is_call = _as_call_mask(right)
    S, K, tau, sigma, r, q, is_call = np.broadcast_arrays(S, K, tau, sigma, r, q, is_call)

    stochastic = (tau > 0) & (sigma > 0) & (S > 0) & (K > 0)
    d1, d2 = d1_d2(S, K, tau, sigma, r, q)
    # Neutral fill so the vectorized math below never sees NaN.
    d1 = np.where(stochastic, d1, 0.0)
    d2 = np.where(stochastic, d2, 0.0)

    disc_r = np.exp(-r * tau)
    disc_q = np.exp(-q * tau)
    sqrt_tau = np.sqrt(np.where(tau > 0, tau, 0.0))
    pdf_d1 = _norm_pdf(d1)

    sign = np.where(is_call, 1.0, -1.0)
    # N(d) for calls, N(-d) for puts - one expression instead of two branches.
    nd1 = ndtr(sign * d1)
    nd2 = ndtr(sign * d2)

    value = sign * (S * disc_q * nd1 - K * disc_r * nd2)
    delta = sign * disc_q * nd1
    # Guard the divisor: cells where it is zero are overwritten below anyway.
    gamma_den = np.where(stochastic, S * sigma * sqrt_tau, 1.0)
    gamma = disc_q * pdf_d1 / gamma_den
    vega = S * disc_q * pdf_d1 * sqrt_tau
    theta = (
        -(S * disc_q * pdf_d1 * sigma) / (2.0 * np.where(sqrt_tau > 0, sqrt_tau, 1.0))
        - sign * r * K * disc_r * nd2
        + sign * q * S * disc_q * nd1
    )
    rho = sign * K * tau * disc_r * nd2

    det = _deterministic(S, K, tau, r, q, is_call)
    out = {}
    for name, stoch_val, det_val in zip(
        ("price", "delta", "gamma", "vega", "theta", "rho"),
        (value, delta, gamma, vega, theta, rho),
        det,
        strict=True,
    ):
        out[name] = np.where(stochastic, stoch_val, det_val)
    return out


def price(S, K, tau, sigma, r, q=0.0, right=CALL) -> np.ndarray:
    """Theoretical value only. See :func:`price_and_greeks` for parameters."""
    return price_and_greeks(S, K, tau, sigma, r, q, right)["price"]


def to_trader_greeks(greeks: dict[str, np.ndarray], days_per_year: float = 365.0) -> dict:
    """Rescale raw greeks to the conventions traders quote.

    ``vega`` per vol point, ``theta`` per day, ``rho`` per basis point.
    ``delta`` and ``gamma`` are unchanged.
    """
    out = dict(greeks)
    if "vega" in out:
        out["vega"] = out["vega"] / 100.0
    if "theta" in out:
        out["theta"] = out["theta"] / days_per_year
    if "rho" in out:
        out["rho"] = out["rho"] / 10000.0
    return out


def implied_vol(target, S, K, tau, r, q=0.0, right=CALL, *, tol=1e-8) -> np.ndarray:
    """Invert :func:`price` for volatility via Brent's method.

    Returns ``NaN`` in two distinct cases:

    1. **No solution in** ``[1e-6, 5.0]`` - a target price outside the
       no-arbitrage bounds, common in real quote data with stale or crossed
       markets.
    2. **Volatility is not identifiable from the price.** Deep ITM and deep
       OTM contracts are pure intrinsic value to machine precision - a 7-day
       50/60 put prices bit-identically for every vol from 1e-6 to 0.07. The
       inverse problem has no unique solution there, and a root-finder will
       happily return the edge of its bracket. Reporting that as a volatility
       would silently poison anything calibrated from it, so we return ``NaN``
       instead.

    Not used by chain generation (which goes the other way); this exists for
    validating our pricer against real market quotes, and for a future
    real-chain vol model.
    """
    target, S, K, tau, r, q = (np.asarray(x, dtype=float) for x in (target, S, K, tau, r, q))
    is_call = _as_call_mask(right)
    target, S, K, tau, r, q, is_call = np.broadcast_arrays(target, S, K, tau, r, q, is_call)

    out = np.full(target.shape, np.nan, dtype=float)
    for idx in np.ndindex(target.shape):
        tgt, s, k, t = target[idx], S[idx], K[idx], tau[idx]
        if not np.isfinite(tgt) or t <= 0 or s <= 0 or k <= 0:
            continue

        def objective(vol, _s=s, _k=k, _t=t, _i=idx, _tgt=tgt):
            return float(price(_s, _k, _t, vol, r[_i], q[_i], is_call[_i])) - _tgt

        try:
            lo, hi = objective(_IV_LOW), objective(_IV_HIGH)
            if lo * hi > 0:  # target outside achievable range - no root to bracket
                continue
            root = brentq(objective, _IV_LOW, _IV_HIGH, xtol=tol)
        except (ValueError, RuntimeError):
            continue

        # Does the price actually respond to volatility here? If a probe moves
        # it by no more than floating-point noise, the root is an artifact of
        # the bracket, not a measurement.
        up = objective(min(root + _IV_PROBE, _IV_HIGH))
        dn = objective(max(root - _IV_PROBE, _IV_LOW))
        if abs(up - dn) <= _IDENTIFIABILITY_EPS * max(1.0, abs(tgt)):
            continue
        out[idx] = root
    return out
