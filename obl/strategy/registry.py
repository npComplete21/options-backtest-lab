"""Strategy discovery. Strategies are files, so registration is a directory listing."""

from __future__ import annotations

import functools
from pathlib import Path

from obl.strategy.spec import StrategySpec

LIBRARY = Path(__file__).parent / "library"


class UnknownStrategyError(KeyError):
    pass


@functools.lru_cache(maxsize=1)
def _index() -> dict[str, Path]:
    return {p.stem: p for p in sorted(LIBRARY.glob("*.yaml"))}


def get(name: str) -> StrategySpec:
    try:
        return StrategySpec.from_yaml(_index()[name])
    except KeyError:
        raise UnknownStrategyError(
            f"unknown strategy {name!r}. Known: {sorted(_index())}. "
            f"Add a YAML file to {LIBRARY} - no code change is needed."
        ) from None


def all_strategies() -> list[str]:
    return sorted(_index())
