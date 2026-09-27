"""Entry triggers and management rules, validated at load time.

These are the half of the DSL that says *when* rather than *what*. Two things
make them worth typing rather than passing through as dicts: Natenberg's first
rule is specified as a **load-time** rejection, and this vocabulary is shared
with ``options-live-validator``, which pins a tag of this package. A rule that
is an opaque dict is one the two repos agree on the spelling of and not the
meaning.
"""

from __future__ import annotations

import datetime as dt

import pytest
import yaml

from obl.strategy.rules import (
    SAFE_ACTIONS,
    UNSAFE_OVERRIDE,
    AbsPortfolioDelta,
    CalendarTrigger,
    NoOpenPosition,
    ProfitTarget,
    RuleError,
    TimeOfDayExit,
    TimeOfDayTrigger,
    UnderlyingTouch,
    build_entry,
    build_management,
)
from obl.strategy.spec import StrategySpec, StrategySpecError

ZERO_DTE = """
name: short_strangle_0dte
legs:
  - {id: short_put,  right: P, qty: -1, expiry: {dte: {target: 0, tolerance: 0}},
                                        strike: {delta: "{{short_put_delta}}"}}
  - {id: short_call, right: C, qty: -1, expiry: {same_as: short_put},
                                        strike: {delta: "{{short_call_delta}}"}}
params:
  short_put_delta:  {type: float, default: 0.10, min: 0.01, max: 0.50}
  short_call_delta: {type: float, default: 0.10, min: 0.01, max: 0.50}
entry:
  trigger: {time_of_day: "09:45", tz: America/New_York}
  filters: [{no_open_position: true}]
management:
  - {profit_target: {pct_of_credit: 0.50}, action: close}
  - {underlying_touch: {leg: short_put},   action: close}
  - {underlying_touch: {leg: short_call},  action: close}
  - {time_of_day: "15:45",                 action: force_flat}
"""


def spec(src: str = ZERO_DTE) -> StrategySpec:
    return StrategySpec(yaml.safe_load(src))


class TestTheZeroDteBaselineLoads:
    """Verbatim from live-validator plan section 4. If this stops parsing, the
    tournament cannot be expressed."""

    def test_it_binds(self):
        bound = spec().bind()
        assert bound.name == "short_strangle_0dte"
        assert len(bound.management) == 4

    def test_entry_is_an_intraday_trigger_in_market_time(self):
        """A naive "09:45" is ambiguous, and 09:45 UTC is four hours before the
        open."""
        trigger = spec().bind().entry.trigger
        assert isinstance(trigger, TimeOfDayTrigger)
        assert trigger.at == dt.time(9, 45)
        assert str(trigger.tz) == "America/New_York"

    def test_timezone_defaults_to_market_time_when_omitted(self):
        entry = build_entry({"trigger": {"time_of_day": "09:45"}})
        assert str(entry.trigger.tz) == "America/New_York"

    def test_the_pre_close_flatten_is_present(self):
        (flat,) = [r for r in spec().bind().management if r.action == "force_flat"]
        assert isinstance(flat.condition, TimeOfDayExit)
        assert flat.condition.at == dt.time(15, 45)

    def test_stops_are_price_levels_not_delta_thresholds(self):
        """As tau approaches zero delta becomes a step function, so a per-leg
        delta stop fires erratically in the last hour - exactly when it matters.
        See live-validator plan section 3."""
        touches = [r for r in spec().bind().management if isinstance(r.condition, UnderlyingTouch)]
        assert {t.condition.leg for t in touches} == {"short_put", "short_call"}


class TestNatenbergRuleIsStructural:
    """ "Never adjust by adding to a losing structure" is specified as a
    *config-load* rejection. A rule caught only at runtime is one the engine
    discovers mid-session with a position on."""

    @pytest.mark.parametrize("action", SAFE_ACTIONS)
    def test_the_safe_actions_are_accepted(self, action):
        rules = build_management([{"profit_target": {"pct_of_credit": 0.5}, "action": action}])
        assert rules[0].action == action

    @pytest.mark.parametrize("action", ["roll", "add", "double_down", "sell_more"])
    def test_anything_that_could_add_risk_is_refused(self, action):
        with pytest.raises(RuleError, match="not permitted"):
            build_management([{"profit_target": {"pct_of_credit": 0.5}, "action": action}])

    def test_the_refusal_names_the_override_rather_than_just_failing(self):
        with pytest.raises(RuleError, match=UNSAFE_OVERRIDE):
            build_management([{"profit_target": {"pct_of_credit": 0.5}, "action": "roll"}])

    def test_the_override_works_but_must_be_explicit(self):
        rules = build_management(
            [{"profit_target": {"pct_of_credit": 0.5}, "action": "roll"}], allow_unsafe=True
        )
        assert rules[0].action == "roll"

    def test_the_override_is_hard_to_type_by_accident(self):
        """Per-strategy whitelisting requires an explicit loud flag, so the
        flag is named to be unmissable in review."""
        assert UNSAFE_OVERRIDE == "unsafe_allow_adding_to_losing_position"

    def test_it_reaches_through_a_whole_strategy_file(self):
        bad = ZERO_DTE.replace(
            "pct_of_credit: 0.50}, action: close", "pct_of_credit: 0.50}, action: roll"
        )
        with pytest.raises(StrategySpecError, match="not permitted"):
            spec(bad)


class TestMistakesFailAtLoadNotAtRuntime:
    def test_a_stop_pointing_at_a_nonexistent_leg(self):
        """A typo the engine would otherwise discover when the stop should have
        fired, which is the worst possible moment to learn about it."""
        with pytest.raises(StrategySpecError, match="does not declare"):
            spec(ZERO_DTE.replace("{leg: short_put}", "{leg: shortt_put}"))

    def test_an_impossible_time(self):
        with pytest.raises(StrategySpecError, match="must be a time"):
            spec(ZERO_DTE.replace('"15:45"', '"25:99"'))

    def test_an_unknown_timezone(self):
        with pytest.raises(StrategySpecError, match="unknown timezone"):
            spec(ZERO_DTE.replace("America/New_York", "Mars/Olympus"))

    def test_a_typo_in_a_condition_name(self):
        with pytest.raises(StrategySpecError, match="unknown management condition"):
            spec(ZERO_DTE.replace("profit_target:", "profit_targt:"))

    def test_a_rule_with_no_action(self):
        with pytest.raises(RuleError, match="declares no action"):
            build_management([{"profit_target": {"pct_of_credit": 0.5}}])

    def test_a_rule_naming_two_conditions(self):
        with pytest.raises(RuleError, match="exactly one"):
            build_management(
                [{"profit_target": {"pct_of_credit": 0.5}, "stop_loss": 2.0, "action": "close"}]
            )

    @pytest.mark.parametrize("pct", [0, -0.5, 1.5])
    def test_a_profit_target_outside_zero_to_one(self, pct):
        with pytest.raises(RuleError, match="pct_of_credit"):
            build_management([{"profit_target": {"pct_of_credit": pct}, "action": "close"}])

    def test_a_non_positive_delta_threshold(self):
        with pytest.raises(RuleError, match="must be > 0"):
            build_management(
                [{"abs_portfolio_delta": {"threshold": -1}, "action": "hedge_underlying"}]
            )


class TestBothEntryDialectsParse:
    """The repo already used a flat list; 0DTE wanted trigger/filters. Both
    normalise here so neither dialect leaks past the loader."""

    def test_the_flat_list_form(self):
        entry = build_entry(
            [{"calendar": {"every": "week", "on": "friday"}}, {"no_open_position": True}]
        )
        assert isinstance(entry.trigger, CalendarTrigger)
        assert entry.filters == (NoOpenPosition(required=True),)

    def test_the_explicit_form(self):
        entry = build_entry(
            {"trigger": {"time_of_day": "09:45"}, "filters": [{"no_open_position": True}]}
        )
        assert isinstance(entry.trigger, TimeOfDayTrigger)
        assert len(entry.filters) == 1

    def test_an_empty_entry_block_is_allowed(self):
        assert build_entry(None).trigger is None

    def test_two_triggers_are_refused(self):
        with pytest.raises(RuleError, match="more than one trigger"):
            build_entry([{"time_of_day": "09:45"}, {"calendar": {"every": "week"}}])

    def test_an_unknown_entry_rule(self):
        with pytest.raises(RuleError, match="unknown entry rule"):
            build_entry([{"when_i_feel_like_it": True}])


class TestExistingStrategiesStillLoad:
    """The multi-week library predates all of this and must be untouched by it."""

    @pytest.mark.parametrize("name", ["short_strangle", "iron_condor"])
    def test_the_library_binds(self, name):
        from obl.strategy import registry

        bound = registry.get(name).bind()
        assert isinstance(bound.entry.trigger, CalendarTrigger)
        assert all(r.action in SAFE_ACTIONS for r in bound.management)

    def test_their_conditions_are_the_daily_ones(self):
        from obl.strategy import registry

        kinds = {r.kind for r in registry.get("short_strangle").bind().management}
        assert kinds == {"ProfitTarget", "StopLoss", "DTEExit"}


def test_rules_are_hashable_so_a_bound_strategy_stays_comparable():
    """Frozen dataclasses throughout: a run's configuration is an identity, and
    identities that can mutate are not identities."""
    assert ProfitTarget(0.5) == ProfitTarget(0.5)
    assert AbsPortfolioDelta(0.2) == AbsPortfolioDelta(0.2)
    assert hash(UnderlyingTouch("short_put")) == hash(UnderlyingTouch("short_put"))
