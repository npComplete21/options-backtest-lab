"""Strategy specs: YAML in, resolved leg templates out.

A strategy is a leg template plus a parameter schema. Both live in YAML, so
adding a strategy is a config change - the whole point of the framework. A
Python escape hatch remains for logic the DSL cannot express, and it produces
the same :class:`LegSpec` list the DSL does, so the engine has one execution
path rather than two.

The ``params`` block generates a Pydantic model at load time. One declaration
then yields validation with bounds, defaults, CLI coercion, the JSON written
to ``params.json``, and the canonical hash that identifies the run. Invalid
parameter combinations fail before any compute starts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, create_model
from pydantic_core import PydanticUndefined

from obl.strategy.rules import (
    UNSAFE_OVERRIDE,
    EntrySpec,
    ManagementRule,
    RuleError,
    build_entry,
    build_management,
)
from obl.strategy.selectors import (
    ATMStrike,
    DeltaStrike,
    DTEExpiry,
    ExpirySelector,
    LegSpec,
    MoneynessStrike,
    NthExpiry,
    OffsetStrike,
    PremiumStrike,
    SameExpiryAs,
    StdDevStrike,
    StrikeSelector,
)

_PLACEHOLDER = re.compile(r"^\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}$")
_EMBEDDED = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

_EXPIRY_SELECTORS: dict[str, type] = {
    "dte": DTEExpiry,
    "nth": NthExpiry,
    "same_as": SameExpiryAs,
}
_STRIKE_SELECTORS: dict[str, type] = {
    "delta": DeltaStrike,
    "moneyness": MoneynessStrike,
    "stddev": StdDevStrike,
    "atm": ATMStrike,
    "premium": PremiumStrike,
    "offset": OffsetStrike,
}
_PARAM_TYPES: dict[str, type] = {"int": int, "float": float, "bool": bool, "str": str}


class StrategySpecError(ValueError):
    """A strategy file is malformed. Raised at load time, before any compute."""


def _substitute(node: Any, params: dict[str, Any]) -> Any:
    """Replace ``{{param}}`` placeholders throughout a nested structure.

    A placeholder that is the entire string yields the parameter's native type
    (so ``"{{wing_width}}"`` becomes a float, not ``"5.0"``); embedded
    placeholders interpolate as text, which is what makes ``"-{{width}}"``
    work for a downside wing.
    """
    if isinstance(node, dict):
        return {k: _substitute(v, params) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, params) for v in node]
    if not isinstance(node, str):
        return node

    whole = _PLACEHOLDER.match(node)
    if whole:
        key = whole.group(1)
        if key not in params:
            raise StrategySpecError(f"unknown parameter {{{{{key}}}}}")
        return params[key]

    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in params:
            raise StrategySpecError(f"unknown parameter {{{{{key}}}}}")
        return str(params[key])

    out = _EMBEDDED.sub(repl, node)
    if out != node:  # "-{{width}}" -> "-5.0"; coerce back to a number if possible
        try:
            return float(out) if "." in out or "e" in out.lower() else int(out)
        except ValueError:
            return out
    return out


def _build_selector(block: Any, table: dict[str, type], kind: str) -> Any:
    if not isinstance(block, dict) or len(block) != 1:
        raise StrategySpecError(
            f"{kind} selector must be a single-key mapping naming the selector, "
            f"got {block!r}. Known: {sorted(table)}"
        )
    ((name, args),) = block.items()
    try:
        cls = table[name]
    except KeyError:
        raise StrategySpecError(
            f"unknown {kind} selector {name!r}; known: {sorted(table)}"
        ) from None
    if isinstance(args, dict):
        return cls(**args)
    if args is None:
        return cls()
    return cls(args)


@dataclass(frozen=True)
class BoundStrategy:
    """A strategy with concrete parameters and fully constructed selectors."""

    name: str
    params: dict[str, Any]
    legs: list[LegSpec]
    entry: EntrySpec
    management: tuple[ManagementRule, ...]

    def param_hash(self, length: int = 8) -> str:
        """Stable hash of name plus canonicalized params, for the ``run_id``.

        Sorting keys makes it order-independent, so the same configuration
        reached two different ways produces one identity.
        """
        payload = json.dumps(
            {"strategy": self.name, "params": self.params}, sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:length]


class StrategySpec:
    """A loaded strategy template, not yet bound to parameter values."""

    def __init__(self, raw: dict[str, Any], source: Path | None = None):
        self.source = source
        try:
            self.name: str = raw["name"]
            self._legs: list[dict[str, Any]] = raw["legs"]
        except KeyError as exc:
            raise StrategySpecError(f"strategy file missing required key: {exc}") from None
        if not self._legs:
            raise StrategySpecError(f"{self.name}: strategy declares no legs")

        self._entry = raw.get("entry") or []
        self._management = raw.get("management") or []
        self._allow_unsafe = bool(raw.get(UNSAFE_OVERRIDE, False))
        self.params_model = self._build_params_model(raw.get("params") or {})

        # Shape-check entry and management now, with placeholders still in
        # place, so a malformed rule fails on load rather than on the first
        # bind. Parameter-dependent values are re-validated in bind().
        self._validate_rules(self._placeholder_params())

    def _build_params_model(self, block: dict[str, Any]) -> type[BaseModel]:
        fields: dict[str, Any] = {}
        for key, meta in block.items():
            if not isinstance(meta, dict) or "type" not in meta:
                raise StrategySpecError(f"{self.name}: param {key!r} needs a 'type'")
            try:
                py_type = _PARAM_TYPES[meta["type"]]
            except KeyError:
                raise StrategySpecError(
                    f"{self.name}: param {key!r} has unknown type {meta['type']!r}; "
                    f"known: {sorted(_PARAM_TYPES)}"
                ) from None
            constraints: dict[str, Any] = {}
            if "min" in meta:
                constraints["ge"] = meta["min"]
            if "max" in meta:
                constraints["le"] = meta["max"]
            default = meta.get("default", ...)
            fields[key] = (py_type, Field(default, **constraints))
        return create_model(f"{self.name}_params", **fields)

    def _placeholder_params(self) -> dict[str, Any]:
        """Stand-in values so rules can be shape-checked before binding.

        A rule whose value is a parameter cannot be range-checked until bind
        time, but its *structure* can be checked immediately - and structure is
        where the dangerous mistakes live: an unknown action, a stop pointing at
        a leg that does not exist.
        """
        out: dict[str, Any] = {}
        for key, field in self.params_model.model_fields.items():
            out[key] = field.default if field.default is not PydanticUndefined else 1
        return out

    def _validate_rules(
        self, params: dict[str, Any]
    ) -> tuple[EntrySpec, tuple[ManagementRule, ...]]:
        leg_ids = frozenset(
            str(leg["id"]) for leg in self._legs if isinstance(leg, dict) and "id" in leg
        )
        try:
            entry = build_entry(_substitute(self._entry, params))
            management = build_management(
                _substitute(self._management, params),
                leg_ids=leg_ids,
                allow_unsafe=self._allow_unsafe,
            )
        except RuleError as exc:
            raise StrategySpecError(f"{self.name}: {exc}") from None
        return entry, management

    @property
    def param_names(self) -> list[str]:
        return sorted(self.params_model.model_fields)

    def bind(self, **overrides: Any) -> BoundStrategy:
        """Validate parameters and construct the concrete leg specs."""
        params = self.params_model(**overrides).model_dump()
        legs = [self._build_leg(raw, params) for raw in self._legs]
        entry, management = self._validate_rules(params)
        return BoundStrategy(
            name=self.name, params=params, legs=legs, entry=entry, management=management
        )

    def _build_leg(self, raw: dict[str, Any], params: dict[str, Any]) -> LegSpec:
        raw = _substitute(raw, params)
        missing = {"id", "right", "qty", "expiry", "strike"} - set(raw)
        if missing:
            raise StrategySpecError(
                f"{self.name}: leg {raw.get('id', '?')!r} missing {sorted(missing)}"
            )
        right = str(raw["right"]).upper()
        if right not in ("C", "P"):
            raise StrategySpecError(f"{self.name}: leg {raw['id']!r} right must be C or P")
        if raw["qty"] == 0:
            raise StrategySpecError(f"{self.name}: leg {raw['id']!r} has qty 0")
        expiry: ExpirySelector = _build_selector(raw["expiry"], _EXPIRY_SELECTORS, "expiry")
        strike: StrikeSelector = _build_selector(raw["strike"], _STRIKE_SELECTORS, "strike")
        return LegSpec(id=raw["id"], right=right, qty=int(raw["qty"]), expiry=expiry, strike=strike)

    @classmethod
    def from_yaml(cls, path: Path | str) -> StrategySpec:
        path = Path(path)
        return cls(yaml.safe_load(path.read_text()), source=path)
