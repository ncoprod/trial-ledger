"""A small vectorised backtester for single-asset rule-based strategies.

It is deliberately simple: the point of the project is counting trials, not the engine.
Conventions: the signal computed from information up to the close of day ``t`` sets the
position held over day ``t+1``; costs are ``fee_bps`` per unit of turnover.
"""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .hashing import hash_array, hash_file, sha256_hex
from .stats import annualise_sharpe, sharpe_moments

# family -> {param: (type, min, max, default)}
FAMILIES: dict[str, dict[str, tuple[type, float, float, Any]]] = {
    "momentum": {"lookback": (int, 2, 365, 20)},
    "ma_crossover": {"fast": (int, 2, 200, 10), "slow": (int, 3, 365, 50)},
    "mean_reversion": {"lookback": (int, 3, 365, 20), "entry_z": (float, 0.1, 4.0, 1.0)},
    "breakout": {"lookback": (int, 3, 365, 20)},
    "rsi": {"period": (int, 2, 100, 14), "lower": (float, 1, 50, 30), "upper": (float, 50, 99, 70)},
}
MODIFIERS: dict[str, tuple[type, Any]] = {"long_only": (bool, False), "vol_target": (bool, False)}
FAMILY_DOCS = {
    "momentum": "sign of the trailing `lookback`-day return",
    "ma_crossover": "long when SMA(fast) > SMA(slow) of price, short otherwise",
    "mean_reversion": "short when the price z-score vs its `lookback` SMA exceeds +entry_z, "
    "long below -entry_z, flat in between",
    "breakout": "long on a new `lookback`-day high, short on a new low, otherwise hold",
    "rsi": "long when RSI(period) < lower, short when > upper, flat otherwise",
}
VOL_WINDOW = 20
VOL_TARGET = 0.02  # fixed per-period risk target, never estimated from the evaluation window
MAX_LEVERAGE = 2.0


@dataclass
class Market:
    """A single asset's per-period simple returns (``returns[t]`` = close_t / close_{t-1} - 1)."""

    name: str
    returns: np.ndarray
    dates: list[str] | None = None
    periods_per_year: float = 365.0
    fee_bps: float = 5.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.returns = np.asarray(self.returns, dtype=float)
        if self.dates is not None and len(self.dates) != len(self.returns):
            raise ValueError("dates and returns must have the same length")

    @property
    def prices(self) -> np.ndarray:
        return np.cumprod(1.0 + self.returns)

    @property
    def dataset_hash(self) -> str:
        return hash_array(self.returns)

    def slice(self, start: int, end: int | None = None) -> Market:
        d = self.dates[start:end] if self.dates is not None else None
        return Market(
            self.name, self.returns[start:end], d, self.periods_per_year, self.fee_bps, self.meta
        )

    @classmethod
    def from_csv(cls, path: str | Path, name: str | None = None, **kw: Any) -> Market:
        """CSV with a ``date`` column and either ``close`` or ``return``."""
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        dates = [r["date"] for r in rows]
        if "return" in rows[0]:
            rets = np.array([float(r["return"]) for r in rows])
        else:
            close = np.array([float(r["close"]) for r in rows])
            rets = np.empty_like(close)
            rets[0] = 0.0
            rets[1:] = close[1:] / close[:-1] - 1.0
        m = cls(name or Path(path).stem, rets, dates, **kw)
        m.meta.setdefault("file_sha256", hash_file(path))
        return m


# ----------------------------------------------------------------------------- specs


def normalise_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate a strategy spec and fill defaults. Raises ``ValueError`` with a helpful message."""
    if not isinstance(spec, dict):
        raise ValueError("strategy must be an object")
    family = spec.get("family")
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose one of {sorted(FAMILIES)}")
    out: dict[str, Any] = {"family": family}
    params = spec.get("params", {}) or {}
    flat = {**params, **{k: v for k, v in spec.items() if k not in {"family", "params"}}}
    allowed = set(FAMILIES[family]) | set(MODIFIERS)
    unknown = set(flat) - allowed
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {sorted(unknown)} for {family}; allowed: {sorted(allowed)}"
        )
    for p, (typ, lo, hi, default) in FAMILIES[family].items():
        v = flat.get(p, default)
        try:
            v = typ(round(float(v))) if typ is int else typ(float(v))
        except (TypeError, ValueError):
            raise ValueError(f"{p} must be a number") from None
        if not (lo <= v <= hi):
            raise ValueError(f"{p}={v} out of range [{lo}, {hi}]")
        out[p] = v
    for m, (_typ, default) in MODIFIERS.items():
        v = flat.get(m, default)
        if isinstance(v, str):
            v = v.strip().lower() in {"1", "true", "yes"}
        out[m] = bool(v)
    if family == "ma_crossover" and out["fast"] >= out["slow"]:
        raise ValueError("ma_crossover requires fast < slow")
    if family == "rsi" and out["lower"] >= out["upper"]:
        raise ValueError("rsi requires lower < upper")
    return out


def expand_grid(family: str, grid: dict[str, list[Any]], **fixed: Any) -> list[dict[str, Any]]:
    keys = sorted(grid)
    values = [v if isinstance(v, list) else [v] for v in (grid[k] for k in keys)]
    return [
        {"family": family, **fixed, **dict(zip(keys, combo, strict=True))}
        for combo in itertools.product(*values)
    ]


# ----------------------------------------------------------------------------- signals


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if w <= len(x):
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[w - 1 :] = (c[w:] - c[:-w]) / w
    return out


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if w <= len(x):
        out[w - 1 :] = sliding_window_view(x, w).std(axis=1, ddof=1)
    return out


def _ffill_zero_start(sig: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs (hold the last position); leading NaNs become 0."""
    idx = np.where(np.isnan(sig), 0, np.arange(sig.size))
    np.maximum.accumulate(idx, out=idx)
    out = sig[idx]
    return np.nan_to_num(out, nan=0.0)


def signal(spec: dict[str, Any], market: Market) -> np.ndarray:
    """Target position in [-1, 1] (before vol targeting) decided at each close."""
    s = normalise_spec(spec)
    r = market.returns
    p = market.prices
    n = r.size
    f = s["family"]
    if f == "momentum":
        lb = s["lookback"]
        trail = _rolling_mean(r, lb)
        sig = np.sign(trail)
    elif f == "ma_crossover":
        sig = np.sign(_rolling_mean(p, s["fast"]) - _rolling_mean(p, s["slow"]))
    elif f == "mean_reversion":
        lb = s["lookback"]
        z = (p - _rolling_mean(p, lb)) / _rolling_std(p, lb)
        sig = np.where(z > s["entry_z"], -1.0, np.where(z < -s["entry_z"], 1.0, 0.0))
        sig[np.isnan(z)] = 0.0
    elif f == "breakout":
        lb = s["lookback"]
        sig = np.full(n, np.nan)
        if lb < n:
            win = sliding_window_view(p[:-1], lb)  # window ending at t-1, aligned to t = lb..n-1
            hi, lo = win.max(axis=1), win.min(axis=1)
            cur = p[lb:]
            sig[lb:] = np.where(cur > hi, 1.0, np.where(cur < lo, -1.0, np.nan))
        sig = _ffill_zero_start(sig)
    elif f == "rsi":
        per = s["period"]
        gain = _rolling_mean(np.maximum(r, 0.0), per)
        loss = _rolling_mean(np.maximum(-r, 0.0), per)
        with np.errstate(divide="ignore", invalid="ignore"):
            rsi = 100.0 - 100.0 / (1.0 + gain / loss)
        rsi = np.where(loss == 0, 100.0, rsi)
        sig = np.where(rsi < s["lower"], 1.0, np.where(rsi > s["upper"], -1.0, 0.0))
        sig[np.isnan(gain)] = 0.0
    else:  # pragma: no cover - guarded by normalise_spec
        raise ValueError(f)
    sig = np.nan_to_num(sig, nan=0.0)
    if s["long_only"]:
        sig = np.clip(sig, 0.0, 1.0)
    if s["vol_target"]:
        vol = _rolling_std(r, VOL_WINDOW)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.clip(VOL_TARGET / vol, 0.0, MAX_LEVERAGE)
        sig = sig * np.nan_to_num(scale, nan=0.0)
    return sig


# ----------------------------------------------------------------------------- run


@dataclass
class BacktestResult:
    spec: dict[str, Any]
    returns: np.ndarray  # strategy returns for periods 1..T-1 of the evaluated window
    sharpe: float  # per period
    sharpe_ann: float
    ann_return: float
    ann_vol: float
    max_drawdown: float
    turnover: float  # average absolute position change per period
    exposure: float  # fraction of periods with a non-zero position
    n_obs: int
    skew: float
    kurtosis: float

    def summary(self) -> dict[str, Any]:
        return {
            "sharpe_annualized": round(self.sharpe_ann, 3),
            "annual_return": round(self.ann_return, 4),
            "annual_volatility": round(self.ann_vol, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "avg_daily_turnover": round(self.turnover, 4),
            "time_in_market": round(self.exposure, 3),
            "n_obs": self.n_obs,
        }


def strategy_returns(spec: dict[str, Any], market: Market) -> np.ndarray:
    pos = signal(spec, market)
    held = pos[:-1]  # decided at close t, held over t+1
    gross = held * market.returns[1:]
    turn = np.abs(np.diff(np.concatenate([[0.0], held])))
    return gross - market.fee_bps * 1e-4 * turn


def run_backtest(spec: dict[str, Any], market: Market, eval_from: int = 0) -> BacktestResult:
    """Backtest ``spec`` on ``market``; statistics use strategy returns from ``eval_from`` on.

    ``eval_from`` lets callers warm indicators up on earlier data (e.g. out-of-sample tests).
    """
    s = normalise_spec(spec)
    pos = signal(s, market)
    rets = strategy_returns(s, market)[max(eval_from - 1, 0) :]
    held = pos[:-1]
    turn = np.abs(np.diff(np.concatenate([[0.0], held])))[max(eval_from - 1, 0) :]
    held = held[max(eval_from - 1, 0) :]
    m = sharpe_moments(rets)
    ppy = market.periods_per_year
    equity = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    mdd = float(np.max(1.0 - equity / peak)) if equity.size else 0.0
    return BacktestResult(
        spec=s,
        returns=rets,
        sharpe=m.sharpe,
        sharpe_ann=annualise_sharpe(m.sharpe, ppy),
        ann_return=float(rets.mean() * ppy),
        ann_vol=float(rets.std(ddof=1) * math.sqrt(ppy)),
        max_drawdown=mdd,
        turnover=float(turn.mean()),
        exposure=float(np.mean(held != 0.0)),
        n_obs=m.n_obs,
        skew=m.skew,
        kurtosis=m.kurtosis,
    )


ENGINE_CODE_HASH = sha256_hex(Path(__file__).read_bytes())
