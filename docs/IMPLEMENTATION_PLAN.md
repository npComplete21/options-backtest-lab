# options-backtest-lab — Implementation Plan

Status: **draft for approval — no code written yet**
Revised: 2026-09-09 (v6 — tau clock is now the shared module, instant-resolution)

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
compute starts, not 40 minutes into a chain-generation run.

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

**Expiration availability is time-varying, and this is load-bearing.** A live
probe on 2026-09-08 showed QQQ listing expirations on all five weekdays
(Fri:18, Wed:5, Thu:4, Mon:2, Tue:1); in 2015 it had only monthlies and Friday
weeklies, with daily expirations arriving around 2022. A generator emitting
the modern grid across a 2015 window backtests contracts that never existed —
the same silent-but-plausible failure mode as pricing off realized vol. Each
instrument therefore carries per-series `available_from` dates, and the
generator filters by the quote date's listing regime. Listing dates are hard
to source, so rules default to `verified: false` and surface as estimates in
run metadata rather than passing as fact.

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

## 3b. Scope limit: this repo cannot backtest 0DTE

The program targets **QQQ 0DTE** (decided 2026-09-08). This repo cannot
serve that goal, and sweeping parameters does not fix it: `sigma_ATM` comes
from `^VXN` (30-day) with term shape borrowed from the SPX complex, whose
shortest anchor is `^VIX9D` — **nine days**. Reaching `tau ~ 0` from there is
invention, not interpolation, and 0DTE is precisely where term structure is
most non-linear and where implied must collapse into realised intraday.

So for 0DTE the repo relationship **inverts**: `options-live-validator`'s
recorder is the primary data source, and a 0DTE backtest becomes possible
only later, by replay. Unrecorded sessions are permanently absent from that
dataset. This repo remains the right home for longer-dated work — 45-DTE
iron condors, weekly strangles — where the vol anchors actually reach.

### The tau clock is a recorded input

At 09:45 on expiry day: calendar `tau` = 0.000713, trading-hours `tau` =
0.003815 — **5.3x in tau, 2.3x in `sigma*sqrt(tau)`**. Since `delta(0.16)`
and `stddev(1.0)` both scale with `sigma*sqrt(tau)`, the clock choice moves
*which strikes get sold* by better than a factor of two. If the two repos
pick clocks independently, no live-vs-backtest comparison means anything and
the discrepancy looks like a market finding rather than a units error.

`obl/timebase.py` therefore holds the convention, imports nothing from the
rest of the project, and — **as of 2026-09-09 — is the shared module itself,
not merely the extraction point for one.** `options-live-validator` imports it
from a pinned tag of this package rather than keeping its own copy; that repo's
plan section 3 makes an identical clock a hard requirement, and it had briefly
forked. Clocks carry a stable `id` (`calendar-365/v1`) that belongs in
`params.json` and the `run_id` hash.

Two consequences of becoming shared:

- **The protocol answers over timezone-aware instants, not dates.** Date
  resolution was survivable while this repo priced only multi-week tenors and
  fatal the moment 0DTE entered the program: on expiry day a date-resolution
  clock can only answer zero, pricing every 0DTE contract at intrinsic.
  Date-granular callers normalise with `at_midnight()` and get exactly the old
  `days/365` back, so no backtest result is restated.
- **Sessions are injected** (`SessionSource`), so the module stays pure-stdlib
  and can be depended on without dragging `pandas_market_calendars`, numpy or
  polars behind it. Each repo supplies its own calendar.

`TradingHoursClock` is the intraday-capable clock live-validator needs, and
`TradingDayClock` now raises `ShortDatedTauError` on a same-session expiry
rather than silently answering zero. What is still missing is the *weighting*:
intraday vol is U-shaped, so flat session time still misstates tau through the
session, and `VolWeightedClock` raises rather than approximating — that curve
must be **measured** from live-validator's recorded data.

`tests/test_timebase.py::test_the_headline_ratio_holds` pins the 5.3x/2.31x
figures above. Both repos size strikes with them, so if that test ever changes
value, every recorded residual in the program changes with it.

### Two further modelling constraints, folded into the design

- **Skew is an input, not a residual.** Flat-vol deltas are wrong by 3-5
  points on the wings — wider than the gap between adjacent strikes — so
  `DeltaStrike` reads deltas from the chain snapshot and never derives its
  own. Asserted by a test that feeds deliberately uniform deltas.
- **Vega is not additive across expirations.** Front months move more than
  backs, so summing raw vegas can show a flat book that is materially long or
  short vol. Resolved legs carry expiry identity so aggregation can weight by
  term structure at Phase 5. Cheap now, expensive to retrofit.

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
    chains.py           # raw -> processed synthetic chains (NumPy + Polars)
    engine.py           # instrument- and strategy-agnostic day walker
    positions.py        # leg/position state, assignment, expiry
    fills.py            # bid/ask + slippage
    capital.py          # margin / sizing
    metrics.py
    run.py              # run_id, manifest, CLI
```

**Hard rule from `CLAUDE.md`:** `black_scholes.py` is pure numpy — no I/O, no
config, no dataframe library. Everything imports it; it imports nothing of
ours. That is what let it be benchmarked and cross-validated against QuantLib
in isolation (§9, §12).

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
per symbol over 10 years — bad for both the writer and DuckDB's scan planner.

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

0. **QuantLib cross-validation** *(implemented)* — price and all five greeks
   against an independent reference implementation. Every other check below
   compares our code against *itself* and shares its formulae, so a subtly
   wrong `d1` could satisfy all of them at once. **Result: agreement to
   1e-13 or better across 720 cases**, including negative rates, confirming
   both the math and the greek conventions (vega per 1.00 vol, theta per
   year, rho per 1.00 rate).
1. **Put-call parity** — `C − P = S·e^{−qτ} − K·e^{−rτ}`, to ~1e-10
2. **Greeks vs. finite differences** across a moneyness/τ grid
3. **Known reference values** — Hull worked examples
4. **Monotonicity & bounds** — price ↑ in σ; calls ↓ in K; `max(S−K,0) ≤ C ≤ S`
5. **Surface no-arbitrage** — calendar + butterfly (§5)
6. **Against real market quotes** — back out IV from a real chain's mid, feed
   it to our pricer, confirm we reproduce the price. Alpha Vantage is
   premium-gated, so use **yfinance `Ticker.option_chain()`** (free, live
   bid/ask/IV) for this. Note that `implied_vol` returns `NaN` for deep
   ITM/OTM contracts where vol is not identifiable — those must be filtered,
   not treated as failures.
7. **Strategy-level golden tests** — a hand-computed iron condor P&L over a
   short window, asserted exactly. This is what catches engine bugs that
   pricing tests cannot.

---

## 10. Build order

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | env, deps, pytest/ruff | **done** — `pytest` clean |
| 1 | `black_scholes.py` + validation 0–4 | **done** — 748 tests, parity 1e-10, QuantLib to 1e-13 |
| 2 | `Instrument` + registry + ingestion → local `data/` | **done** — QQQ + AAPL resolve and ingest; 791 tests |
| 3 | selectors + strategy spec loader + registry | **done** — both YAML strategies resolve through one path; 844 tests |
| 4 | `surface.py` + `chains.py` (Polars, short window) | validation 5–6 pass |
| 5 | engine + positions + fills + capital + metrics | validation 7: hand-checked IC run |
| 6 | second strategy + second instrument, **zero engine changes** | **the real test of the abstraction** |
| 7 | sweeps (joblib), DuckDB result queries, notebooks | sensitivity surface |

**Phase 6 is the acceptance test for this whole redesign.** If adding
strategy #2 on instrument #2 requires editing the engine, the abstraction
was wrong and it is much cheaper to find that out at Phase 6 than Phase 7.

---

## 11. Decisions

**Resolved 2026-09-08:**

| # | Decision | Choice |
|---|---|---|
| 1 | Strategy definition | **YAML specs + Python ABC escape hatch**, compiling to one engine path |
| 2 | Single-stock accuracy | **Ship with `directional_only` tag** — assumed IV premium, labelled honestly |
| 3 | First targets | **short strangle + iron condor**, on **QQQ + AAPL** — both axes exercised early |
| 4 | Calendar spread | **Deferred** — two-expiry selector path built but not yet exercised by a strategy |
| 5 | Alpha Vantage premium | **No** — proceed synthetic, re-raise before Phase 4 |

Choice 3 matters most: one high-confidence ETF plus one `directional_only`
single stock from the start, so ETF-shaped assumptions cannot harden in the
engine before Phase 6 tests for them.

**Still open — gate Phase 5, not Phases 0–4:**

6. **European approximation** — proceeding European for Phase 1 (it is what
   `CLAUDE.md`'s "Black-Scholes" implies, and it is correct for OTM ETF
   strategies). §3 notes this is weaker for single stocks. An
   American/Bjerksund-Stensland pricer is an additive extension behind the
   same interface, not a rewrite — flag if you want it before Phase 5.
7. **Fill model** — vega-based vol-point half-spread (§8), or your own assumption?
8. **Capital model** — Reg-T margin with per-strategy rules (§8), or fixed notional?
9. **Backtest window** — 2015→now spans 2018 Volmageddon and 2020 COVID
   (good regime coverage, but the options market changed structurally).

## 12. Why not Spark — measured

The v1/v2 plans inherited PySpark, EMR Serverless and Athena from
`CLAUDE.md`. Benchmarked on this machine against the real pricing module:

| Operation | Measured |
|---|---|
| Full 10-year chain, 1 symbol (806,400 contracts, all 5 greeks) | **0.09 s** |
| Throughput | 8.8M contracts/sec, single core |
| Peak memory | 78 MB |
| 10 symbols | ~0.9 s, ~0.8 GB |
| Polars write 806k rows → Parquet | 0.08 s |
| DuckDB group-by over that Parquet | 0.01 s |
| *Spark job startup, before any work* | *10–30 s* |

Spark's startup overhead alone is 100–300× the actual compute. The rough
industry threshold is that single-node tools win below ~10 GB and Spark pays
off from ~100 GB upward — this workload is three orders of magnitude below
that. Worse, the backtest engine is a **sequential state machine** (tomorrow's
position depends on today's), which is the poorest possible fit for Spark's
execution model. The parallelism that genuinely exists is *across independent
sweep runs* — a job-queue problem, handled by `joblib`.

**Resulting stack:** NumPy (pricing) · Polars (chains/dataframes) · DuckDB
(result queries, replacing Athena) · joblib (sweep parallelism) · Parquet
(storage, S3 optional). This also removes the Python 3.11 constraint, EMR
cost, and the Athena layer entirely.

**Revisit if** the project moves to tick-level or intraday options data —
billions of rows is genuinely distributed territory.

## 13. New dependencies

**Runtime:** `numpy` · `scipy` (norm CDF/PDF, IV root-finding) · `polars` ·
`duckdb` · `pyarrow` · `pydantic` (param schemas) · `pyyaml` (strategy specs)
· `pandas_market_calendars` (trading days, expiries) · `yfinance` · `boto3`

**Dev only:** `pytest`, `pytest-cov`, `ruff`, `QuantLib` (independent pricing
oracle — never imported by the runtime path)
