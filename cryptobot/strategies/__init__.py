"""Strategy contract.

A strategy is a module with one pure function:

    def signal(bars: pd.DataFrame, params: dict) -> Signal | None

`bars` holds every completed bar up to and including the one that just closed (the last row),
for one symbol, indexed by bar start time in the exchange's timezone. Return a Signal to enter
at the market now, or None. No I/O, no broker calls, no randomness that is not seeded by params.
Optional module attributes: DEFAULT_PARAMS (dict) and PARAM_GRID (dict of lists) for the
learning loop, and

    def due(symbol: str, start: datetime, tf: int, params: dict) -> bool

a cheap, pure pre-check from the bar's start time alone: False means `signal` would certainly
return None on this bar, so the engine can skip building the bar window (backtest speed).
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

BUILTIN = {
    "orb": "cryptobot.strategies.orb",
    "intraday_momentum": "cryptobot.strategies.intraday_momentum",
    "vwap_revert": "cryptobot.strategies.vwap_revert",
    "random_entry": "cryptobot.strategies.random_entry",
}


@dataclass(frozen=True)
class Signal:
    side: str  # "long" | "short"
    entry: float  # reference entry price (the order is a market order)
    stop: float  # protective stop; required
    target: float | None = None  # take-profit; None = hold until the end-of-day flatten
    reason: str = ""

    def __post_init__(self) -> None:
        if self.side not in ("long", "short"):
            raise ValueError(f"side must be 'long' or 'short', got {self.side!r}")
        if not all(x > 0 for x in (self.entry, self.stop)):
            raise ValueError("entry and stop must be positive prices")
        if self.side == "long" and self.stop >= self.entry:
            raise ValueError("long stop must be below entry")
        if self.side == "short" and self.stop <= self.entry:
            raise ValueError("short stop must be above entry")
        if self.target is not None:
            if self.side == "long" and self.target <= self.entry:
                raise ValueError("long target must be above entry")
            if self.side == "short" and self.target >= self.entry:
                raise ValueError("short target must be below entry")

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop)


def load_strategy(name_or_path: str) -> ModuleType:
    """Load a built-in strategy by name, or any .py file that defines `signal`."""
    if name_or_path in BUILTIN:
        module = importlib.import_module(BUILTIN[name_or_path])
    elif name_or_path.endswith(".py"):
        path = Path(name_or_path)
        if not path.exists():
            raise FileNotFoundError(f"strategy file not found: {path}")
        spec = importlib.util.spec_from_file_location(f"cryptobot_strategy_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load strategy from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        raise ValueError(f"unknown strategy {name_or_path!r}; built-ins: {sorted(BUILTIN)}")
    if not callable(getattr(module, "signal", None)):
        raise ValueError(f"strategy {name_or_path!r} has no signal(bars, params) function")
    return module


def due_check(module: ModuleType):
    """The strategy's optional `due` pre-check, or None."""
    fn = getattr(module, "due", None)
    return fn if callable(fn) else None


def default_params(module: ModuleType) -> dict:
    return dict(getattr(module, "DEFAULT_PARAMS", {}) or {})


def param_grid(module: ModuleType) -> dict[str, list]:
    return dict(getattr(module, "PARAM_GRID", {}) or {})


def strategy_name(module: ModuleType) -> str:
    for name, path in BUILTIN.items():
        if getattr(module, "__name__", None) == path:
            return name
    return str(getattr(module, "NAME", getattr(module, "__name__", "custom")))
