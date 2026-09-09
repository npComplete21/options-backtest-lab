"""Symbol -> :class:`Instrument` resolution, driven by YAML config."""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

from src.instruments.base import Instrument

_CONFIG = Path(__file__).parent / "config" / "instruments.yaml"


class UnknownInstrumentError(KeyError):
    """Raised for a symbol with no registry entry.

    Deliberately fatal rather than defaulted: guessing a strike grid, exercise
    style and vol model for an unknown symbol would produce a backtest that
    runs cleanly and describes nothing real.
    """


@functools.lru_cache(maxsize=1)
def _load(path: str | None = None) -> dict[str, Instrument]:
    raw = yaml.safe_load(Path(path or _CONFIG).read_text())
    return {
        symbol: Instrument(symbol=symbol, **body)
        for symbol, body in (raw.get("instruments") or {}).items()
    }


def get(symbol: str) -> Instrument:
    """Resolve a symbol, case-insensitively."""
    registry = _load()
    try:
        return registry[symbol.upper()]
    except KeyError:
        raise UnknownInstrumentError(
            f"{symbol!r} is not in the instrument registry. Known: "
            f"{sorted(registry)}. Add it to src/instruments/config/instruments.yaml "
            "- it needs a strike grid, exercise style, vol model and expiration "
            "availability dates before it can be backtested."
        ) from None


def all_symbols() -> list[str]:
    return sorted(_load())
