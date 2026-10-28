"""Figures for the README, generated from experiments/results/episodes.csv."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analyze import RESULTS, THRESH

FIG = RESULTS / "figures"
INK, MUTED, GRID = "#1f2328", "#59636e", "#d1d9e0"
C_NAIVE, C_CLAIMED, C_LEDGER = "#d1495b", "#edae49", "#00798c"


def _num(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows() -> list[dict]:
    with open(RESULTS / "episodes.csv") as fh:
        return list(csv.DictReader(fh))


def short(model: str) -> str:
    return model.split("/")[-1]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_false_discovery(rows: list[dict]) -> Path:
    null = [r for r in rows if r["market"] != "btc" and _num(r["psr0"]) is not None]
    by = defaultdict(list)
    for r in null:
        by[r["model"]].append(r)
    models = sorted(by, key=lambda m: (m == "random-search", m))
    x = np.arange(len(models))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    for i, (key, label, color) in enumerate(
        [
            ("psr0", "naive: PSR(SR>0) > 0.95", C_NAIVE),
            ("dsr_claimed", "DSR with self-reported N", C_CLAIMED),
            ("dsr_ledger", "DSR with ledger N", C_LEDGER),
        ]
    ):
        rates = []
        for m in models:
            vals = [_num(r[key]) for r in by[m]]
            vals = [v for v in vals if v is not None]
            rates.append(np.mean([v > THRESH for v in vals]) if vals else np.nan)
        bars = ax.bar(x + (i - 1) * w, rates, w, color=color, label=label)
        for b, rt in zip(bars, rates, strict=True):
            if np.isfinite(rt):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    rt + 0.015,
                    f"{rt:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=INK,
                )
    ax.set_xticks(x, [f"{short(m)}\n(n={len(by[m])})" for m in models], fontsize=8, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of episodes 'significant'", color=INK)
    ax.set_title(
        "Zero-alpha markets: how often the selected strategy passes",
        loc="left",
        color=INK,
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax)
    fig.tight_layout()
    out = FIG / "false_discovery.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_trial_count(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=150)
    models = sorted({r["model"] for r in rows if r["model"] != "random-search"})
    palette = ["#00798c", "#d1495b", "#edae49", "#66a182", "#2e4057", "#8d96a3"]
    lo, hi = 1, 1
    for m, c in zip(models, palette, strict=False):
        pts = [(_num(r["n_records"]), _num(r["claimed_n"])) for r in rows if r["model"] == m]
        pts = [(a, b) for a, b in pts if a and b]
        if not pts:
            continue
        a, b = np.array(pts).T
        hi = max(hi, a.max(), b.max())
        ax.scatter(a, b, s=22, alpha=0.8, color=c, label=short(m), edgecolor="none")
    ax.plot([lo, hi * 1.3], [lo, hi * 1.3], color=MUTED, lw=0.8, ls="--")
    ax.text(hi * 0.9, hi * 1.02, "claimed = ledger", color=MUTED, fontsize=7, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("backtests logged in the ledger", color=INK)
    ax.set_ylabel("trials the agent says it tested", color=INK)
    ax.set_title("Self-reported vs ledger trial count", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=7.5)
    _style(ax)
    fig.tight_layout()
    out = FIG / "trial_count.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_btc_oos(rows: list[dict]) -> Path | None:
    btc = [r for r in rows if r["market"] == "btc" and _num(r.get("oos_sharpe")) is not None]
    if not btc:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=150)
    models = sorted({r["model"] for r in btc}, key=lambda m: (m == "random-search", m))
    palette = ["#00798c", "#d1495b", "#edae49", "#66a182", "#2e4057", "#8d96a3"]
    for m, c in zip(models, palette, strict=False):
        rs = [r for r in btc if r["model"] == m]
        ax.scatter(
            [_num(r["is_sharpe"]) for r in rs],
            [_num(r["oos_sharpe"]) for r in rs],
            s=28,
            color=c,
            alpha=0.85,
            label=short(m),
            edgecolor="none",
        )
    ax.axhline(0, color=MUTED, lw=0.8)
    lim = max(2.0, max(_num(r["is_sharpe"]) for r in btc) * 1.1)
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("in-sample Sharpe, 2018-2023 (annualised)", color=INK)
    ax.set_ylabel("out-of-sample Sharpe, 2024-01 to 2026-09", color=INK)
    ax.set_title(
        "BTCUSDT: selected strategy in vs out of sample", loc="left", color=INK, fontsize=11
    )
    ax.legend(frameon=False, fontsize=7.5)
    _style(ax)
    fig.tight_layout()
    out = FIG / "btc_oos.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def make_all() -> list[Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    outs = [fig_false_discovery(rows), fig_trial_count(rows), fig_btc_oos(rows)]
    outs = [o for o in outs if o]
    for o in outs:
        print(o)
    return outs
