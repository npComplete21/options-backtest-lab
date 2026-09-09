"""Instrument definitions: what a tradeable symbol *is*.

The instrument layer answers questions the strategy layer must never ask:
which contracts exist, on what strike grid, expiring when, priced off which
volatility source. Strategies address contracts through selectors and never
name a symbol; instruments describe a market and never name a strategy. See
``docs/IMPLEMENTATION_PLAN.md`` section 2.

Adding a symbol is a config change (``config/instruments.yaml``), not code.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class AssetClass(StrEnum):
    ETF = "etf"
    SINGLE_STOCK = "single_stock"
    INDEX = "index"


class ExerciseStyle(StrEnum):
    AMERICAN = "american"
    EUROPEAN = "european"


class Settlement(StrEnum):
    PHYSICAL = "physical"
    CASH = "cash"


class ExpirationType(StrEnum):
    """Expiration series. Weekday variants are separate because they became
    listable at different times, and a backtest must not trade one before it
    existed."""

    MONTHLY = "monthly"  # third Friday - the series that always existed
    WEEKLY_MON = "weekly_mon"
    WEEKLY_TUE = "weekly_tue"
    WEEKLY_WED = "weekly_wed"
    WEEKLY_THU = "weekly_thu"
    WEEKLY_FRI = "weekly_fri"

    @property
    def weekday(self) -> int:
        """Target weekday, Monday=0. Monthlies target Friday."""
        return {
            ExpirationType.MONTHLY: 4,
            ExpirationType.WEEKLY_MON: 0,
            ExpirationType.WEEKLY_TUE: 1,
            ExpirationType.WEEKLY_WED: 2,
            ExpirationType.WEEKLY_THU: 3,
            ExpirationType.WEEKLY_FRI: 4,
        }[self]


class ExpirationRule(BaseModel):
    """When one expiration series became listable for a symbol.

    ``verified`` is deliberately explicit and defaults to ``False``. Listing
    dates are genuinely hard to source, and an unverified date silently
    treated as fact would reintroduce the exact problem these rules exist to
    prevent - backtesting contracts that never traded. An unverified rule is
    usable but self-declares as an estimate, so it surfaces in run metadata
    rather than hiding in code.
    """

    model_config = {"frozen": True}

    type: ExpirationType
    available_from: dt.date
    verified: bool = False
    note: str = ""


class ContractSpec(BaseModel):
    """The listed-contract grid. Synthetic strikes must land on it, or the
    backtest prices contracts that could not have been traded."""

    model_config = {"frozen": True}

    multiplier: int = 100
    strike_increment: float = Field(gt=0)
    tick_size: float = Field(default=0.01, gt=0)
    exercise: ExerciseStyle = ExerciseStyle.AMERICAN
    settlement: Settlement = Settlement.PHYSICAL

    def snap_strike(self, strike: float) -> float:
        """Round to the nearest listed strike.

        Callers must record requested vs. realized (plan section 7): silent
        snapping is how a backtest stops describing a tradeable strategy.
        """
        return round(round(strike / self.strike_increment) * self.strike_increment, 6)


class VolModelSpec(BaseModel):
    """How this instrument's implied volatility is sourced.

    ``confidence`` propagates into ``params.json`` and the results index so
    that observed-IV and assumed-IV runs can never be silently compared in
    one table. See ``docs/IMPLEMENTATION_PLAN.md`` section 3.
    """

    model_config = {"frozen": True}

    kind: str  # "vol_index" | "single_stock" | "real_chain"
    vol_index: str | None = None
    term_structure_source: list[str] = Field(default_factory=list)
    confidence: str = "directional_only"

    @model_validator(mode="after")
    def _index_model_needs_an_index(self) -> VolModelSpec:
        if self.kind == "vol_index" and not self.vol_index:
            raise ValueError("vol_model.kind='vol_index' requires a vol_index symbol")
        return self


class Instrument(BaseModel):
    """A tradeable underlying and everything the engine needs to model it."""

    model_config = {"frozen": True}

    symbol: str
    asset_class: AssetClass
    price_symbol: str | None = None  # data-source symbol, if it differs
    contract_spec: ContractSpec
    vol_model: VolModelSpec
    expirations: list[ExpirationRule] = Field(min_length=1)
    earnings_sensitive: bool = False
    notes: str = ""

    @property
    def data_symbol(self) -> str:
        return self.price_symbol or self.symbol

    @model_validator(mode="after")
    def _single_stocks_are_earnings_sensitive(self) -> Instrument:
        if self.asset_class == AssetClass.SINGLE_STOCK and not self.earnings_sensitive:
            raise ValueError(
                f"{self.symbol}: single stocks have quarterly IV ramp and crush, which "
                "changes the risk profile four times a year. Set earnings_sensitive=true "
                "(and model it) rather than leaving it implicit."
            )
        return self

    def expirations_available_on(self, as_of: dt.date) -> list[ExpirationType]:
        """Expiration series listable on ``as_of``.

        The generator must filter through this. QQQ has expirations on all
        five weekdays today but had only monthlies and Friday weeklies in
        2015; emitting the modern grid across a 2015 window would backtest
        contracts that did not exist.
        """
        return [r.type for r in self.expirations if r.available_from <= as_of]

    def unverified_expiration_rules(self) -> list[ExpirationRule]:
        """Rules whose listing date is an estimate. Surfaced in run metadata."""
        return [r for r in self.expirations if not r.verified]
