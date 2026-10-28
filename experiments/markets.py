"""Markets for the experiment: two zero-alpha null markets and real BTCUSDT.

Null markets have zero alpha *by construction*: for any strategy whose position at ``t`` uses
only information up to ``t``, the expected return at ``t+1`` is exactly zero (before costs).

* ``signflip``: magnitudes |r| from a stationary bootstrap (Politis & Romano 1994, mean block
  20 days) of real BTCUSDT daily returns, multiplied by i.i.d. random signs. This keeps fat
  tails and volatility clustering (which live in |r|) but makes returns a symmetric martingale
  difference sequence: E[r_{t+1} | past] = 0.
* ``garch``: zero-mean GARCH(1,1) with standardised Student-t innovations, parameters fitted
  by maximum likelihood to the same BTCUSDT returns.

The real market is BTCUSDT daily (Binance spot) with 2018-01-01..2023-12-31 shown to the
agent and 2024-01-01 onwards held out for out-of-sample evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import optimize, special

from trial_ledger.backtest import Market

DATA = Path(__file__).parent / "data" / "btcusdt_1d.csv"
IS_START, IS_END, OOS_START = "2018-01-01", "2023-12-31", "2024-01-01"
N_NULL = 2190  # same length as the real in-sample period (6 years of daily data)
PERIODS_PER_YEAR = 365.0
FEE_BPS = 5.0
MARKET_KINDS = ("signflip", "garch", "btc")


@lru_cache(maxsize=1)
def btc_full() -> Market:
    m = Market.from_csv(DATA, name="btcusdt", periods_per_year=PERIODS_PER_YEAR, fee_bps=FEE_BPS)
    i0 = m.dates.index(IS_START)
    return m.slice(i0)


def btc_split() -> tuple[Market, int]:
    """Full BTC series from IS_START and the index of the first out-of-sample day."""
    full = btc_full()
    return full, full.dates.index(OOS_START)


def btc_in_sample_returns() -> np.ndarray:
    full, oos = btc_split()
    return full.returns[1:oos]  # drop the first (undefined) return


def _dates(n: int) -> list[str]:
    d0 = date.fromisoformat(IS_START)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]


def stationary_bootstrap_indices(n_src: int, n: int, mean_block: float, rng) -> np.ndarray:
    idx = np.empty(n, dtype=int)
    p = 1.0 / mean_block
    idx[0] = rng.integers(n_src)
    jumps = rng.random(n) < p
    starts = rng.integers(n_src, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % n_src
    return idx


def signflip_returns(seed: int, n: int = N_NULL, mean_block: float = 20.0) -> np.ndarray:
    rng = np.random.default_rng([seed, 1])
    src = np.abs(btc_in_sample_returns())
    mag = src[stationary_bootstrap_indices(src.size, n, mean_block, rng)]
    return mag * rng.choice([-1.0, 1.0], size=n)


@dataclass(frozen=True)
class GarchParams:
    omega: float
    alpha: float
    beta: float
    nu: float


MAX_PERSISTENCE = 0.995


def _garch_nll(theta: np.ndarray, r: np.ndarray) -> float:
    """Negative log-likelihood with variance targeting: omega = var(r) * (1 - alpha - beta)."""
    alpha, beta, nu = theta
    if alpha < 0 or beta < 0 or alpha + beta >= MAX_PERSISTENCE or nu <= 2.05 or nu > 100:
        return 1e10
    omega = r.var() * (1.0 - alpha - beta)
    n = r.size
    h = np.empty(n)
    h[0] = r.var()
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
    # standardised Student-t log-density of r_t / sqrt(h_t)
    z2 = r**2 / (h * (nu - 2.0))
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    ll = c - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(z2)
    return float(-ll.sum())


@lru_cache(maxsize=1)
def fit_garch() -> GarchParams:
    r = btc_in_sample_returns()
    r = r - r.mean()
    v = r.var()
    x0 = np.array([0.08, 0.88, 4.0])
    res = optimize.minimize(
        _garch_nll,
        x0,
        args=(r,),
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8},
    )
    alpha, beta, nu = map(float, res.x)
    return GarchParams(float(v * (1.0 - alpha - beta)), alpha, beta, nu)


def garch_returns(seed: int, n: int = N_NULL, burn: int = 500) -> np.ndarray:
    g = fit_garch()
    rng = np.random.default_rng([seed, 2])
    z = rng.standard_t(g.nu, size=n + burn) * math.sqrt((g.nu - 2) / g.nu)
    h = g.omega / (1 - g.alpha - g.beta)
    out = np.empty(n + burn)
    for t in range(n + burn):
        out[t] = math.sqrt(h) * z[t]
        h = g.omega + g.alpha * out[t] ** 2 + g.beta * h
    r = out[burn:]
    return np.clip(r, -0.95, None)  # simple returns cannot fall below -100%


def make_market(kind: str, seed: int) -> tuple[Market, Market | None, int | None]:
    """Return (in-sample market shown to the agent, full market for OOS, OOS start index)."""
    if kind == "btc":
        full, oos = btc_split()
        return full.slice(0, oos), full, oos
    if kind == "signflip":
        r = signflip_returns(seed)
    elif kind == "garch":
        r = garch_returns(seed)
    else:
        raise ValueError(kind)
    r = np.concatenate([[0.0], r])  # first row is the base price, like a CSV of closes
    m = Market(f"{kind}-{seed}", r, _dates(r.size), PERIODS_PER_YEAR, FEE_BPS)
    return m, None, None
