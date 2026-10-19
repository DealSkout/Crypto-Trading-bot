"""Load and validate config.yaml. Secrets never live here: they come from .env via os.environ."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cryptobot.sessions import is_pair, normalize

VALID_MODES = ("paper", "live")
VALID_BROKERS = ("alpaca", "paper")
VALID_PROVIDERS = ("alpaca", "csv", "synthetic")
VALID_TIME_IN_FORCE = ("gtc", "ioc")
VALID_ENTRY_TYPES = ("market", "limit")
USD_QUOTES = ("USD", "USDT", "USDC")
VALID_COACH_PROVIDERS = ("none", "claude", "ollama")
VALID_DECIDER_ROLES = ("last_resort", "final_say", "primary")
TIMEFRAME_RE = re.compile(r"^(\d+)m$")

# Keys under `risk:` are owned by the human. Nothing automated may change them.
RISK_KEYS = frozenset(
    {
        "risk_per_trade_pct",
        "max_daily_loss_pct",
        "max_positions",
        "flatten_minutes_before_close",
        "max_stop_pct",
        "max_total_exposure_pct",
        "sizing",
        "target_daily_vol_pct",
        "kelly_fraction",
        "kelly_min_trades",
        "kelly_lookback_trades",
        "kelly_floor_pct",
        "pause_on_macro",
        "pause_on_stale_data",
        "stale_bars",
    }
)
VALID_SIZING = ("risk", "vol_target", "kelly", "min")


class ConfigError(ValueError):
    pass


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.5
    max_daily_loss_pct: float = 2.0
    max_positions: int = 3
    flatten_minutes_before_close: int = 10
    max_stop_pct: float = 3.0  # no stop may sit further than this from the entry
    max_total_exposure_pct: float = 100.0  # all open positions together, % of equity
    # position sizing, always capped by risk_per_trade_pct:
    #   risk        risk_per_trade_pct of equity between entry and stop
    #   vol_target  also no bigger than a position whose typical daily move is
    #               target_daily_vol_pct of equity
    #   kelly       risk kelly_fraction x the Kelly fraction of recent trades (<= the cap)
    #   min         the smallest of all three
    sizing: str = "min"
    target_daily_vol_pct: float = 1.0
    kelly_fraction: float = 0.25  # never above 0.25
    kelly_min_trades: int = 30  # below this many closed trades Kelly is not used
    kelly_lookback_trades: int = 100
    kelly_floor_pct: float = 0.1  # with no measured edge, keep trading this small (% risk)
    # automatic pauses of NEW entries (open trades keep their stops)
    pause_on_macro: bool = True  # inside a macro-calendar high-risk window
    pause_on_stale_data: bool = True  # prices (or research.required_sources) gone stale
    stale_bars: int = 3  # no new bar for this many bar lengths = stale


@dataclass
class CostConfig:
    commission: float = 0.0  # flat, per order, in USD (Alpaca crypto: 0)
    fee_pct: float = 0.25  # Alpaca crypto taker fee per side at the lowest volume tier
    spread_bps: float = 5.0
    slippage_bps: float = 5.0
    # extra adverse slippage on software-stop exits in simulated fills: live stops are polled
    # (every poll_seconds) and closed at the market, so they fill beyond the stop price
    stop_slippage_bps: float = 10.0


@dataclass
class OrderConfig:
    time_in_force: str = "gtc"  # gtc | ioc (Alpaca crypto accepts these, never "day")
    entry_type: str = "market"  # market | limit (a marketable limit, see limit_offset_bps)
    limit_offset_bps: float = 10.0  # limit price this far through the last price
    fill_timeout_seconds: float = 30.0
    # a resting stop_limit sell at Alpaca behind every software stop, so a crashed process
    # still has protection. It sits backup_stop_buffer_pct beyond the software stop.
    backup_stop: bool = True
    backup_stop_buffer_pct: float = 1.0


@dataclass
class DataConfig:
    provider: str = "alpaca"
    lookback_days: int = 365
    cache_dir: str = "cache"
    csv_dir: str = "data"


@dataclass
class LearningConfig:
    # legacy `refit` and the backtest's train/test split
    refit_every_days: int = 5
    train_fraction: float = 0.7
    # the weekly learning loop (cryptobot/learning.py) and its promotion gate
    strategies: list[str] = field(default_factory=lambda: ["orb", "vwap_revert"])
    max_challengers: int = 20
    holdout_fraction: float = 0.3  # last 30% of days: the out-of-sample holdout
    min_oos_trades: int = 50  # holdout trades a challenger needs
    min_sharpe_gain: float = 0.25  # holdout Sharpe must beat the champion's by this much
    dsr_min: float = 0.95  # deflated Sharpe probability the gain is not luck
    max_pbo: float = 0.5  # probability of backtest overfitting allowed
    cv_blocks: int = 8  # combinatorially purged cross-validation blocks
    wf_folds: int = 4  # walk-forward folds
    embargo_days: int = 1  # days purged between train and test
    min_profitable_regimes: int = 2
    min_regime_trades: int = 5
    trial_memory_days: int = 182  # learning runs in this window count as looks at the holdout
    shadow_days: int = 14  # a winner runs this long in shadow before it can be champion
    shadow_max_days: int = 28  # ...and is rejected if it still has too few trades by then
    min_shadow_trades: int = 5
    workers: int = 0  # parallel simulations; 0 = CPU count - 1


@dataclass
class CoachConfig:
    enabled: bool = True  # one switch: false turns the whole coach off
    # the daily trade review (cryptobot/trade_coach.py): one capped Claude call per day
    review_model: str = "claude-haiku-5-5"
    review_max_tokens: int = 3000
    min_trades_for_params: int = 30
    dir: str = "coach"
    # the strategy research coach (cryptobot/coach.py), run with `python -m cryptobot coach`
    provider: str = "none"
    model: str = "claude-opus-5-5"
    ollama_model: str = "llama3.1:8b"
    lookback_days: int = 20
    auto_apply_params: bool = False  # ignored: the coach never applies anything (kept so old
    # config files still load)
    max_daily_cost_usd: float = 0.5


@dataclass
class DeciderConfig:
    """An LLM making buy/sell decisions inside the trading loop. Off unless provider is set."""

    provider: str = "none"
    role: str = "last_resort"
    model: str = "claude-opus-5-5"
    ollama_model: str = "llama3.1:8b"
    every_minutes: int = 30
    max_stop_pct: float = 2.0
    min_confidence: float = 0.6
    max_daily_cost_usd: float = 2.0
    effort: str = "medium"
    timeout_seconds: float = 30.0


@dataclass
class GuardianConfig:
    """The watchdog that can stop the bot completely. Rules always on; AI reviewer optional."""

    max_drawdown_pct: float = 10.0
    max_loss_30d_pct: float = 6.0
    max_consecutive_losing_days: int = 5
    max_halts_30d: int = 3
    max_errors_per_day: int = 10
    min_trades_for_judgement: int = 10
    expected_range_r: float = 1.0
    provider: str = "claude"  # none | claude | ollama; it can only stop, never resume
    model: str = "claude-haiku-5-5"
    ollama_model: str = "llama3.1:8b"
    min_confidence: float = 0.7
    max_daily_cost_usd: float = 0.5
    timeout_seconds: float = 120.0
    stop_file: str = "cryptobot.STOPPED"


@dataclass
class ResearchConfig:
    """The market research feed (cryptobot/research). Data only: it never trades."""

    enabled: bool = True
    exchanges: list[str] = field(default_factory=lambda: ["okx", "bybit", "binance"])
    rss_feeds: list[str] = field(
        default_factory=lambda: [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
            "https://www.theblock.co/rss.xml",
        ]
    )
    cryptopanic: bool = False  # needs CRYPTOPANIC_TOKEN in .env
    cryptopanic_url: str = "https://cryptopanic.com/api/developer/v2/posts/"
    calendar_file: str = "macro_calendar.yaml"
    calendar_warn_days: int = 45
    refresh_minutes: dict[str, float] = field(
        default_factory=lambda: {"fear_greed": 60, "funding": 30, "open_interest": 30, "news": 20}
    )
    stale_after_minutes: dict[str, float] = field(
        default_factory=lambda: {
            "fear_greed": 2880,
            "funding": 720,
            "open_interest": 360,
            "news": 720,
        }
    )
    # sources whose staleness pauses new entries (prices are always required)
    required_sources: list[str] = field(default_factory=list)
    backfill_days: int = 400
    digest: bool = True  # the one LLM call a day; without a key the digest uses numbers only
    digest_model: str = "claude-haiku-5-5"
    digest_max_tokens: int = 1200
    max_daily_cost_usd: float = 0.05


@dataclass
class RegimeConfig:
    """Rule thresholds for cryptobot/regime.py (one label per market per UTC day)."""

    vol_short_days: int = 5
    vol_long_days: int = 20
    high_vol_ratio: float = 1.6  # 5-day vol this many times the 20-day vol = high_vol
    er_days: int = 14
    trending_efficiency: float = 0.35  # efficiency ratio at or above this = trending
    ma_days: int = 20
    risk_off_breadth: float = 0.34  # at most this share of markets above their 20-day average
    risk_off_market_ret_pct: float = -10.0  # ...and the average 20-day return at or below this
    risk_off_fear_greed: float = 20.0  # or extreme fear with half the markets below average


VALID_EXCHANGES = ("okx", "bybit", "binance")
RESEARCH_SOURCES = ("fear_greed", "funding", "open_interest", "news")


@dataclass
class Config:
    mode: str = "paper"
    broker: str = "alpaca"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD", "SOL/USD"])
    timeframe: str = "5m"
    strategy: str = "orb"
    params: dict[str, Any] = field(default_factory=dict)
    starting_cash: float = 1000.0
    allow_short: bool = False  # always false: crypto is long-only, no margin
    max_entries_per_day: int = 1
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    orders: OrderConfig = field(default_factory=OrderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    coach: CoachConfig = field(default_factory=CoachConfig)
    decider: DeciderConfig = field(default_factory=DeciderConfig)
    guardian: GuardianConfig = field(default_factory=GuardianConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    poll_seconds: int = 30
    journal_path: str = "cryptobot.sqlite"
    reports_dir: str = "reports"
    store_dir: str = "store"  # Parquet export of decisions/trades, refreshed after each UTC day
    telegram: bool = True
    account_currency: str = "USD"  # Alpaca accounts are in USD
    path: str = "config.yaml"

    @property
    def timeframe_minutes(self) -> int:
        return timeframe_minutes(self.timeframe)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("path", None)
        return d


def timeframe_minutes(tf: str) -> int:
    m = TIMEFRAME_RE.match(str(tf).strip())
    if not m or int(m.group(1)) <= 0:
        raise ConfigError(f"timeframe must look like '5m', got {tf!r}")
    return int(m.group(1))


def _build(section_cls, raw: dict | None, name: str):
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"'{name}' must be a mapping")
    allowed = {f for f in section_cls.__dataclass_fields__}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"unknown keys under '{name}': {sorted(unknown)}")
    return section_cls(**raw)


def from_dict(raw: dict[str, Any], path: str = "config.yaml") -> Config:
    raw = dict(raw or {})
    cfg = Config(
        mode=str(raw.get("mode", "paper")).lower(),
        broker=str(raw.get("broker", "alpaca")).lower(),
        symbols=[normalize(str(s)) for s in (raw.get("symbols") or [])],
        timeframe=str(raw.get("timeframe", "5m")),
        strategy=str(raw.get("strategy", "orb")),
        params=dict(raw.get("params") or {}),
        starting_cash=float(raw.get("starting_cash", 1000.0)),
        allow_short=bool(raw.get("allow_short", False)),
        max_entries_per_day=int(raw.get("max_entries_per_day", 1)),
        risk=_build(RiskConfig, raw.get("risk"), "risk"),
        costs=_build(CostConfig, raw.get("costs"), "costs"),
        orders=_build(OrderConfig, raw.get("orders"), "orders"),
        data=_build(DataConfig, raw.get("data"), "data"),
        learning=_build(LearningConfig, raw.get("learning"), "learning"),
        coach=_build(CoachConfig, raw.get("coach"), "coach"),
        decider=_build(DeciderConfig, raw.get("decider"), "decider"),
        guardian=_build(GuardianConfig, raw.get("guardian"), "guardian"),
        research=_build(ResearchConfig, raw.get("research"), "research"),
        regime=_build(RegimeConfig, raw.get("regime"), "regime"),
        poll_seconds=int(raw.get("poll_seconds", 30)),
        journal_path=str(raw.get("journal_path", "cryptobot.sqlite")),
        reports_dir=str(raw.get("reports_dir", "reports")),
        store_dir=str(raw.get("store_dir", "store")),
        telegram=bool(raw.get("telegram", True)),
        path=path,
    )
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.mode not in VALID_MODES:
        raise ConfigError(f"mode must be one of {VALID_MODES}, got {cfg.mode!r}")
    if cfg.broker not in VALID_BROKERS:
        raise ConfigError(f"broker must be one of {VALID_BROKERS}, got {cfg.broker!r}")
    if not cfg.symbols:
        raise ConfigError("symbols must list at least one crypto pair, e.g. BTC/USD")
    for s in cfg.symbols:
        if not is_pair(s) or s.split("/")[1] not in USD_QUOTES:
            raise ConfigError(f"{s!r} is not a USD crypto pair like BTC/USD or ETH/USD")
    if cfg.allow_short:
        raise ConfigError("allow_short must be false: crypto is long-only (no shorting, no margin)")
    o = cfg.orders
    if o.time_in_force not in VALID_TIME_IN_FORCE:
        raise ConfigError(f"orders.time_in_force must be one of {VALID_TIME_IN_FORCE}")
    if o.entry_type not in VALID_ENTRY_TYPES:
        raise ConfigError(f"orders.entry_type must be one of {VALID_ENTRY_TYPES}")
    if not (0 <= o.limit_offset_bps <= 200) or o.fill_timeout_seconds <= 0:
        raise ConfigError("orders: 0 <= limit_offset_bps <= 200 and fill_timeout_seconds > 0")
    if not (0.1 <= o.backup_stop_buffer_pct <= 10):
        raise ConfigError("orders.backup_stop_buffer_pct must be between 0.1 and 10")
    timeframe_minutes(cfg.timeframe)
    if cfg.data.provider not in VALID_PROVIDERS:
        raise ConfigError(f"data.provider must be one of {VALID_PROVIDERS}")
    if cfg.coach.provider not in VALID_COACH_PROVIDERS:
        raise ConfigError(f"coach.provider must be one of {VALID_COACH_PROVIDERS}")
    if not 256 <= cfg.coach.review_max_tokens <= 16000:
        raise ConfigError("coach.review_max_tokens must be between 256 and 16000")
    if cfg.coach.min_trades_for_params < 1:
        raise ConfigError("coach.min_trades_for_params must be at least 1")
    d = cfg.decider
    if d.provider not in VALID_COACH_PROVIDERS:
        raise ConfigError(f"decider.provider must be one of {VALID_COACH_PROVIDERS}")
    if d.role not in VALID_DECIDER_ROLES:
        raise ConfigError(f"decider.role must be one of {VALID_DECIDER_ROLES}")
    if d.every_minutes < 1 or not (0 < d.max_stop_pct <= 10) or not (0 <= d.min_confidence <= 1):
        raise ConfigError(
            "decider: every_minutes >= 1, 0 < max_stop_pct <= 10, 0 <= min_confidence <= 1"
        )
    if d.effort not in ("low", "medium", "high"):
        raise ConfigError("decider.effort must be low, medium or high")
    r = cfg.risk
    if not (0 < r.risk_per_trade_pct <= 5):
        raise ConfigError("risk.risk_per_trade_pct must be between 0 and 5 (percent)")
    if not (0 < r.max_daily_loss_pct <= 20):
        raise ConfigError("risk.max_daily_loss_pct must be between 0 and 20 (percent)")
    if r.max_positions < 1:
        raise ConfigError("risk.max_positions must be >= 1")
    if r.flatten_minutes_before_close < 1:
        raise ConfigError("risk.flatten_minutes_before_close must be >= 1")
    if not (0 < r.max_stop_pct <= 20):
        raise ConfigError("risk.max_stop_pct must be between 0 and 20 (percent)")
    if not (0 < r.max_total_exposure_pct <= 100):
        raise ConfigError("risk.max_total_exposure_pct must be between 0 and 100 (no leverage)")
    if r.sizing not in VALID_SIZING:
        raise ConfigError(f"risk.sizing must be one of {VALID_SIZING}")
    if not (0 < r.kelly_fraction <= 0.25):
        raise ConfigError("risk.kelly_fraction must be above 0 and at most 0.25")
    if not (0 < r.target_daily_vol_pct <= 10) or not (0 <= r.kelly_floor_pct <= 1):
        raise ConfigError("risk: 0 < target_daily_vol_pct <= 10 and 0 <= kelly_floor_pct <= 1")
    if r.kelly_min_trades < 10 or r.kelly_lookback_trades < r.kelly_min_trades:
        raise ConfigError("risk: kelly_min_trades >= 10 and kelly_lookback_trades >= it")
    if r.stale_bars < 2:
        raise ConfigError("risk.stale_bars must be at least 2")
    g = cfg.guardian
    if g.provider not in VALID_COACH_PROVIDERS:
        raise ConfigError(f"guardian.provider must be one of {VALID_COACH_PROVIDERS}")
    if not (0 < g.max_drawdown_pct <= 100) or not (0 < g.max_loss_30d_pct <= 100):
        raise ConfigError("guardian drawdown/loss limits must be percentages between 0 and 100")
    if g.max_consecutive_losing_days < 1 or g.max_halts_30d < 1 or g.max_errors_per_day < 1:
        raise ConfigError("guardian day/halt/error limits must be >= 1")
    if not (0 <= g.min_confidence <= 1):
        raise ConfigError("guardian.min_confidence must be between 0 and 1")
    rg = cfg.regime
    if min(rg.vol_short_days, rg.er_days, rg.ma_days) < 2 or rg.vol_long_days <= rg.vol_short_days:
        raise ConfigError("regime: windows must be >= 2 days and vol_long_days > vol_short_days")
    rs = cfg.research
    bad = [x for x in rs.exchanges if x not in VALID_EXCHANGES]
    if bad:
        raise ConfigError(f"research.exchanges: unknown {bad}; use {VALID_EXCHANGES}")
    bad = [x for x in rs.required_sources if x not in RESEARCH_SOURCES]
    if bad:
        raise ConfigError(f"research.required_sources: unknown {bad}; use {RESEARCH_SOURCES}")
    if not (0 <= rs.max_daily_cost_usd <= 5):
        raise ConfigError("research.max_daily_cost_usd must be between 0 and 5")
    c = cfg.costs
    if min(c.commission, c.fee_pct, c.spread_bps, c.slippage_bps, c.stop_slippage_bps) < 0:
        raise ConfigError("costs must not be negative")
    lc = cfg.learning
    if not (0.3 <= lc.train_fraction <= 0.9):
        raise ConfigError("learning.train_fraction must be between 0.3 and 0.9")
    if not (0.1 <= lc.holdout_fraction <= 0.5):
        raise ConfigError("learning.holdout_fraction must be between 0.1 and 0.5")
    if not (0.5 <= lc.dsr_min < 1) or not (0 < lc.max_pbo <= 1):
        raise ConfigError("learning: 0.5 <= dsr_min < 1 and 0 < max_pbo <= 1")
    if lc.min_oos_trades < 1 or lc.min_profitable_regimes < 1 or lc.shadow_days < 1:
        raise ConfigError("learning: min_oos_trades, min_profitable_regimes, shadow_days >= 1")
    if lc.cv_blocks < 4 or lc.cv_blocks % 2 or lc.wf_folds < 2:
        raise ConfigError("learning: cv_blocks must be even and >= 4; wf_folds >= 2")
    if cfg.starting_cash <= 0:
        raise ConfigError("starting_cash must be positive")
    if cfg.mode == "live" and cfg.broker == "paper":
        raise ConfigError("mode: live needs broker: alpaca, not the 'paper' simulator")


def load_config(path: str | os.PathLike = "config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with p.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml must be a mapping at the top level")
    return from_dict(raw, path=str(p))


def live_confirmed() -> bool:
    """Rule 1: live trading needs `mode: live` AND the env var CRYPTOBOT_LIVE_CONFIRM=yes."""
    return os.environ.get("CRYPTOBOT_LIVE_CONFIRM", "").strip().lower() == "yes"


def write_params(path: str | os.PathLike, params: dict[str, Any]) -> None:
    """Rewrite the `params:` line of config.yaml in place, keeping every other line and comment.

    Used by the learning loop when a parameter change passes the gate. Only `params:` is ever
    touched; `risk:` is never written by code.
    """
    p = Path(path)
    text = p.read_text() if p.exists() else ""
    line = "params: " + yaml.safe_dump(params, default_flow_style=True, sort_keys=True).strip()
    lines = text.splitlines()
    for i, existing in enumerate(lines):
        if re.match(r"^params\s*:", existing):
            end = i + 1  # also drop an indented block-style mapping under `params:`
            while end < len(lines) and (lines[end].strip() == "" or lines[end][0] in " \t"):
                end += 1
            while end > i + 1 and lines[end - 1].strip() == "":
                end -= 1
            lines[i:end] = [line]
            break
    else:
        lines.append(line)
    p.write_text("\n".join(lines) + "\n")
