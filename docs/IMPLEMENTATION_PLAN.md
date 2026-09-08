# options-backtest-lab — Implementation Plan

Status: **draft for approval — no code written yet**
Revised: 2026-09-08 (v2 — generalized across strategy × instrument)

---

## 0. Design goal

The system must express:

> run `<strategy>` on `<instrument>` over `<date range>` with `<strategy-specific parameters>`

as **configuration, not code**. Adding "iron condor on AAPL for the last
6 months, 16-delta shorts, $5 wings" must not require touching the engine.

The v1 draft of this plan failed that test — it treated "QQQ short strangle"
as the shape of the system rather than as one row in it. This revision fixes
that. The concrete target:

```bash
python -m backtest run \
  --strategy iron_condor \
  --symbol AAPL \
  --start 2025-03-08 --end 2025-09-08 \
  --set dte_target=45 short_put_delta=0.16 short_call_delta=0.16 wing_width=5 \
  --capital 100000
```

### The architectural rule that makes this work

**The strategy layer never knows which instrument it is running on, and the
instrument layer never knows which strategy is running.** They meet at
exactly one interface: a daily *chain snapshot*. Every hardcoded "QQQ" or
"short strangle" assumption is a violation of this rule.

Three orthogonal axes, previously conflated:

| Axis | Varies | Owns |
|---|---|---|
| **Instrument** | SPY, QQQ, AAPL, SPX… | data sources, vol model, dividends, calendar, contract spec |
| **Strategy** | strangle, iron condor, calendar… | leg template, entry trigger, management rules |
| **Run** | dates, capital, costs, sweeps | execution config, `run_id`, results |

---

## 1. Strategy definition — declarative, with a code escape hatch

### The core insight

Every option strategy reduces to the same primitive: **select N legs, each a
`(right, expiry selector, strike selector, quantity)`**. Strategies differ in
*how many* legs and *how strikes are chosen* — not in the mechanics of
choosing them. So strike/expiry selection becomes the shared vocabulary, and
most strategies become data.

### Strike selectors

| Selector | Meaning | Used by |
|---|---|---|
| `delta(0.16)` | strike nearest 16-delta | strangle, IC, verticals |
| `moneyness(pct=-0.05)` | 5% OTM | simple % rules |
| `stddev(n=1.0)` | n × σ√τ from spot | vol-scaled selection |
| `atm()` | nearest to forward | calendars, straddles |
| `offset(base=<leg>, points=±X)` | **relative to another leg** | IC/butterfly wings |
| `premium(target=1.50)` | strike whose premium ≈ $1.50 | credit-targeted entries |

`offset` is the one that makes this genuinely general rather than a pile of
special cases: an iron condor's wings are *defined relative to* its shorts.
Leg resolution therefore becomes a small dependency DAG — topologically
sorted, cycles rejected at config-load time.

### Expiry selectors

`dte(target=45, tolerance=7)` · `nth_expiry(1)` · `fixed_date(...)` ·
`same_as(<leg>)` · constraints like `monthly_only`

Two *different* expiry selectors in one position is what makes calendar and
diagonal spreads fall out of the same abstraction for free.

### A strategy is then a YAML file

```yaml
name: iron_condor
legs:
  - {id: short_put,  right: P, qty: -1, expiry: {dte: {target: "{{dte_target}}", tolerance: 7}},
                                        strike: {delta: "{{short_put_delta}}"}}
  - {id: long_put,   right: P, qty: +1, expiry: {same_as: short_put},
                                        strike: {offset: {base: short_put,  points: "-{{wing_width}}"}}}
  - {id: short_call, right: C, qty: -1, expiry: {same_as: short_put},
                                        strike: {delta: "{{short_call_delta}}"}}
  - {id: long_call,  right: C, qty: +1, expiry: {same_as: short_put},
                                        strike: {offset: {base: short_call, points: "+{{wing_width}}"}}}

params:
  dte_target:       {type: int,   default: 45,   min: 1,    max: 365}
  short_put_delta:  {type: float, default: 0.16, min: 0.01, max: 0.50}
  short_call_delta: {type: float, default: 0.16, min: 0.01, max: 0.50}
  wing_width:       {type: float, default: 5.0,  min: 0.5}

entry:
  trigger: {calendar: {every: week, on: friday}}
  filters: [{no_open_position: true}]

management:
  - {profit_target: {pct_of_credit: 0.50}, action: close}
  - {stop_loss:     {multiple_of_credit: 2.0}, action: close}
  - {dte_exit:      {at: 21}, action: close}
  - {delta_breach:  {leg: short_put, abs_delta: 0.30}, action: roll_down}
```

**Short strangle is the same file with the two wing legs deleted.** Covered
call is one stock leg plus a short call. Calendar is two legs with different
expiry selectors. That is the test that the abstraction is real.

### Escape hatch

A `Strategy` ABC in Python for anything the DSL can't express (dynamic
ratios, path-dependent leg construction). **The DSL compiles down to the same
object the ABC defines**, so there is one execution path in the engine, not
two. Registered via entry point so third-party strategies drop in.

### Parameter schemas

Each strategy's `params` block generates a **Pydantic model** at load time.
This gives, from one declaration: validation with bounds, defaults,
`--set k=v` CLI coercion, JSON serialization into `params.json`, and the
canonical hash for `run_id`. Invalid parameter combinations fail before any
compute starts, not 40 minutes into a Spark job.

Discoverability:
```bash
python -m backtest strategies          # list registered strategies
python -m backtest params iron_condor  # schema, defaults, valid ranges
```

---

## 2. Instrument abstraction

```python
class Instrument:
    symbol: str
    asset_class: ETF | SINGLE_STOCK | INDEX
    def spot_series(start, end) -> Series
    def vol_model() -> VolModel            # differs by instrument — see §3
    def dividend_model() -> DividendModel  # continuous q vs discrete
    def calendar() -> Calendar             # trading days, expiries, earnings
    def contract_spec() -> ContractSpec
```

`ContractSpec` is easy to overlook and matters: **strike increment**
($1 for AAPL, $5 for SPX), **multiplier**, **exercise style** (American for
equity/ETF options, European for SPX/NDX), **settlement** (physical vs cash).
Generating synthetic strikes on an arbitrary grid would backtest contracts
that never existed — the spec is what keeps the synthetic chain plausible.

---

## 3. Vol model — where generalization gets genuinely hard

**This is the honest caveat, and it is not a plumbing problem.**

The v1 plan's central recommendation was: drive ATM vol from a real vol index
(`^VIX`/`^VXN`/`^RVX`) so that both the IV we sell *and* the path that
realizes are real historical observations, and the edge is inherited rather
than assumed. That argument still holds — **for the three ETFs it covers.**

For AAPL, there is no free vol index. So a single-stock run necessarily falls
back to an *assumed* IV premium, which is exactly the failure mode §1 of the
v1 plan warned about. Generalizing the framework is easy; generalizing the
**accuracy** is not.

### Per-instrument vol models

| Model | Applies to | IV source | Confidence |
|---|---|---|---|
| `VolIndexModel` | SPY, QQQ, IWM | observed `^VIX`/`^VXN`/`^RVX` | **high** — IV is real |
| `SingleStockModel` | AAPL, etc. | RV × premium multiplier calibrated on ETFs | **low** — premium assumed |
| `RealChainModel` | any covered symbol | real historical IV (premium data) | highest — *not currently available* |

**Runs using `SingleStockModel` are tagged `confidence: directional_only` in
`params.json` and in the results index**, so ETF and single-stock numbers can
never be silently compared side by side in a summary table.

### Five further single-stock effects, none of which are config

1. **Earnings.** IV ramps into an earnings date and crushes after. For an
   AAPL iron condor this is first-order — the risk profile changes four
   times a year. Requires historical earnings dates plus a term-structure
   bump model. Ignoring it makes short-premium backtests optimistic.
2. **Discrete dividends**, not continuous `q` — drives early exercise on ITM calls.
3. **American exercise / assignment risk** — materially more important than for OTM ETF strategies.
4. **Splits** — AAPL's 4:1 in 2020. Strike adjustment ≠ price adjustment.
5. **Jump risk.** Single stocks gap on news far more than index ETFs.
   Black-Scholes lognormality underprices tails, and short-premium strategies
   are precisely the ones that get hurt by tails. Expect optimistic results.

### Recommendation

Build the framework instrument-agnostic **now** (free if designed right),
ship `VolIndexModel` first, and ship `SingleStockModel` behind the
`directional_only` label. This fully supports "iron condor on AAPL for
6 months" — it just labels the output honestly instead of implying the
number is as trustworthy as an SPY one.

**Upgrade path:** Alpha Vantage `HISTORICAL_OPTIONS` returns 15+ years of
real chains with IV and greeks. I tested it on this key — it is
**premium-gated** (so is `REALTIME_OPTIONS`, which returns only a synthetic
sample schema). A subscription would enable `RealChainModel`, remove the
assumed-premium problem for single stocks entirely, and let us *calibrate*
the skew that §5 currently has to sweep. Worth pricing before Phase 4.

---

## 4. Module layout

```
src/
  pricing/
    black_scholes.py    # THE pricing module — pure vectorized numpy, no I/O
    volatility.py       # realized-vol estimators
    surface.py          # sigma(k, tau) + arbitrage checks
  instruments/
    base.py             # Instrument, ContractSpec
    registry.py         # symbol -> Instrument resolution
    vol_models.py       # VolIndexModel / SingleStockModel / RealChainModel
  ingestion/
    prices.py  vol_index.py  rates.py  dividends.py  calendar.py  earnings.py
  strategy/
    spec.py             # YAML schema -> Pydantic params
    selectors.py        # strike + expiry selectors (the shared vocabulary)
    base.py             # Strategy ABC (escape hatch)
    registry.py
    library/            # iron_condor.yaml, short_strangle.yaml, calendar.yaml...
  backtest/
    chains.py           # raw -> processed synthetic chains
    engine.py           # instrument- and strategy-agnostic day walker
    positions.py        # leg/position state, assignment, expiry
    fills.py            # bid/ask + slippage
    capital.py          # margin / sizing
    metrics.py
    run.py              # run_id, manifest, CLI
```

**Hard rule from `CLAUDE.md`:** `black_scholes.py` is pure numpy — no Spark,
no I/O, no config. Everything imports it; it imports nothing of ours.

**The engine contains zero strategy names and zero ticker symbols.** If a
grep for `QQQ` or `strangle` hits `backtest/engine.py`, the design has failed.

---

## 5. Volatility surface

`sigma(k, tau) = sigma_ATM(tau) · [1 + beta·k + gamma·k²]`, in log-moneyness
`k = ln(K/F)`. `beta` (skew slope, negative for equity) and `gamma`
(curvature) are **swept, not fixed** — with no historical chain data they are
unobservable, so the honest deliverable is a *sensitivity surface, not a point
estimate*. Deliberately not SVI/SABR: equally invented, but far more
authoritative-looking.

Every parameter set must pass no-arbitrage checks — total variance
non-decreasing in `tau` (no calendar arb), call prices convex in `K` (no
butterfly arb) — or be rejected rather than silently backtested.

---

## 6. Processed zone schema

One row per `(quote_date, symbol, expiration, strike, right)`:

`quote_date` · `symbol` · `expiration` · `strike` · `right` ·
`underlying_close` · `forward` · `dte_calendar` · `dte_trading` · `tau` ·
`rate` · `div_yield` · `sigma_quote` · `log_moneyness` · `theo_value` ·
`delta` `gamma` `theta` `vega` `rho` · `surface_id` · `vol_model_id`

**Partitioning:** `processed/chains/symbol=<S>/year=<YYYY>/month=<MM>/`, with
`quote_date` as a column. Daily partitions would mean ~2,500 tiny partitions
per symbol over 10 years — bad for Spark and Athena both.

`surface_id` and `vol_model_id` are load-bearing: the zone holds multiple
priced chains for the same date/strike under different assumptions, and these
are what stop them mixing.

---

## 7. `run_id`, results, and sweeps

```
run_id = <strategy>-<symbol>-<UTC timestamp>-<8-char param hash>
       = iron-condor-AAPL-20260908T143000Z-a3f9c1d2
```

Under `results/<strategy>/<run_id>/` per `CLAUDE.md`:

- `params.json` — strategy + params + symbol + window + `surface_id` +
  `vol_model_id` + **confidence tag** + git SHA. Enough to reproduce exactly.
- `trades.parquet` — one row per leg event (open/close/adjust/expire/assign),
  recording **requested vs. realized** selection (asked 16Δ, filled 17.3Δ)
- `daily.parquet` — per-date MTM, position greeks, cash, equity, margin
- `metrics.json` — Sharpe, max DD, win rate, CAGR, trade count

`results/_index/` — one row per run (strategy, symbol, params hash, window,
confidence, headline metrics). This is what makes cross-strategy and
cross-instrument comparison a query instead of a directory crawl, and it is
where the `directional_only` tag does its work.

Sweeps are the same machinery: a sweep config expands to a set of `run_id`s
sharing a `sweep_id`.

### Strike resolution honesty

A requested 16-delta strike may not exist on the contract grid. Policy: snap
to nearest valid strike per `ContractSpec`, **record both requested and
realized**, and fail loudly if the miss exceeds a tolerance. Silent snapping
is how backtests quietly stop describing a tradeable strategy.

---

## 8. Fills and capital — required before any P&L is believable

**Fills (§6 of v1, unchanged):** marking at BS theoretical mid overstates
returns badly — an iron condor is *four* legs crossed on entry and exit,
~50 cycles/year. Model half-spread in vol points, widening with moneyness and
shortening DTE, converted via vega, with a dollar tick floor, plus
per-contract commission. Sweep it: "how wide do spreads get before this stops
working?" is one of the more useful questions here.

**Capital:** Sharpe/CAGR/drawdown need a denominator, and a short strangle
has no natural cost basis. Reg-T-style margin recomputed daily, configurable
starting equity, max-utilization cap driving position size. Defined-risk
strategies (iron condor) margin at wing width — so this must be a
**per-strategy margin rule**, not one global formula.

---

## 9. Validation harness

Cheapest → strongest:

1. **Put-call parity** — `C − P = S·e^{−qτ} − K·e^{−rτ}`, to ~1e-10
2. **Greeks vs. finite differences** across a moneyness/τ grid
3. **Known reference values** — Hull worked examples
4. **Monotonicity & bounds** — price ↑ in σ; calls ↓ in K; `max(S−K,0) ≤ C ≤ S`
5. **Surface no-arbitrage** — calendar + butterfly (§5)
6. **Against real market quotes** — back out IV from a real chain's mid, feed
   it to our pricer, confirm we reproduce the price. Alpha Vantage is
   premium-gated, so use **yfinance `Ticker.option_chain()`** (free, live
   bid/ask/IV) for this.
7. **Strategy-level golden tests** — a hand-computed iron condor P&L over a
   short window, asserted exactly. This is what catches engine bugs that
   pricing tests cannot.

---

## 10. Build order

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | env, deps, pytest/ruff, calendar | `pytest` clean |
| 1 | `black_scholes.py` + validation 1–4 | parity 1e-10, greeks match FD |
| 2 | `Instrument` + registry + ingestion → local `data/` | ETF + single-stock symbol both resolve |
| 3 | selectors + strategy spec loader + registry | `strangle`/`iron_condor` YAML load & validate; selector unit tests |
| 4 | `surface.py` + `chains.py` (Spark local, short window) | validation 5–6 pass |
| 5 | engine + positions + fills + capital + metrics | validation 7: hand-checked IC run |
| 6 | second strategy + second instrument, **zero engine changes** | **the real test of the abstraction** |
| 7 | sweeps, S3, EMR Serverless, Athena, notebooks | sensitivity surface |

**Phase 6 is the acceptance test for this whole redesign.** If adding
strategy #2 on instrument #2 requires editing the engine, the abstraction
was wrong and it is much cheaper to find that out at Phase 6 than Phase 7.

---

## 11. Open decisions

**Framework (new — these drive the redesign):**

1. **Strategy DSL in YAML + Python escape hatch** (recommended), or
   Python-only strategy classes? YAML makes sweeps and diffs trivial; classes
   are simpler but push every new strategy through code review.
2. **First strategies to implement** — short strangle + iron condor covers
   undefined-risk and defined-risk. Add a calendar to prove the
   two-expiry path early, or defer?
3. **First instruments** — one ETF (QQQ, high-confidence vol model) plus one
   single stock (AAPL, `directional_only`) to force genericity from the
   start? Recommended: doing both early prevents ETF assumptions leaking in.
4. **Single-stock accuracy** (§3) — accept `directional_only` labelling, or
   restrict v1 to ETFs until real chain data is available?
5. **Alpha Vantage premium** — worth subscribing? It would remove the
   assumed-IV problem for single stocks and let us calibrate rather than
   sweep the skew. Changes §3 and §5 substantially.

**Carried over from v1:**

6. **European approximation** — accept for OTM ETF strategies (recommended),
   noting §3 makes this weaker for single stocks?
7. **Fill model** — vega-based vol-point spread (§8), or your own assumption?
8. **Capital model** — Reg-T margin with per-strategy rules (§8), or fixed notional?
9. **Backtest window** — 2015→now spans 2018 Volmageddon and 2020 COVID
   (good regime coverage, but the options market changed structurally).

---

## 12. Sizing note

One chain ≈ `2,500 days × ~40 strikes × 2 rights × ~4 expiries ≈ 800k
rows/symbol` — small; pandas would cope. PySpark earns its place at Phase 7,
where sweeps multiply that by every (surface × fill × strategy-param ×
instrument) combination. Keeping `black_scholes.py` pure numpy means the same
code serves both, so this costs nothing either way.

## 13. New dependencies

`pydantic` (param schemas) · `pandas_market_calendars` (trading days,
expiries) · `pytest`, `pytest-cov` · `scipy` (norm CDF/PDF, IV root-finding)
· `pyarrow` (Parquet outside Spark) · `pyyaml` (strategy specs)
