"""Turn a ledger into a multiple-testing report: N, best Sharpe, PSR, DSR and PBO."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from .ledger import Ledger
from .stats import (
    annualise_sharpe,
    deflated_sharpe_ratio,
    effective_number_of_trials,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


@dataclass
class Report:
    n_records: int
    n_trials: int  # distinct scored trials (re-runs of an identical return series counted once)
    n_unscored: int  # logged backtest runs without a Sharpe ratio (e.g. from a shell hook)
    best_seq: int | None
    best_strategy: dict[str, Any] | None
    best_sharpe: float | None  # per period
    best_sharpe_ann: float | None
    n_obs: int | None
    skew: float | None
    kurtosis: float | None
    moments_known: bool | None  # False: no return series logged, Gaussian moments assumed
    var_trials_sharpe: float | None  # per period, across distinct trials
    psr0: float | None
    sr0_ledger: float | None  # expected max Sharpe under the null given ledger N (per period)
    dsr_ledger: float | None
    claimed_n: int | None
    dsr_claimed: float | None
    n_effective: float | None
    dsr_effective: float | None
    pbo: float | None
    pbo_n_trials: int | None
    pbo_partitions: int | None
    pbo_note: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_scored(r: dict[str, Any]) -> bool:
    x = r.get("sharpe")
    return isinstance(x, (int, float)) and math.isfinite(x)


def _distinct(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for r in records:
        if not _is_scored(r):
            continue
        key = r.get("returns_hash") or f"{r.get('strategy_hash')}|{r.get('dataset_hash')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def pbo_from_ledger(
    ledger: Ledger, trials: list[dict[str, Any]], n_partitions: int = 16
) -> tuple[float | None, int | None, str | None]:
    series = []
    missing = 0
    for r in trials:
        arr = ledger.load_returns(r["returns_hash"]) if r.get("returns_hash") else None
        if arr is not None:
            series.append(arr)
        else:
            missing += 1
    if missing:
        return (
            None,
            None,
            f"missing returns for {missing} trials; PBO requires the full trial universe",
        )
    if len(series) < 2:
        return None, None, "fewer than 2 trials with stored return series"
    lengths = {len(s) for s in series}
    if len(lengths) != 1:
        # keep the most common length (trials evaluated on the same period)
        common = max(lengths, key=lambda n: sum(len(s) == n for s in series))
        series = [s for s in series if len(s) == common]
        note = f"PBO over the {len(series)} trials sharing the most common length {common}"
    else:
        note = None
    if len(series) < 2:
        return None, None, "fewer than 2 trials share a common period"
    m = np.column_stack(series)
    s = n_partitions
    while s > 2 and m.shape[0] // s < 4:
        s -= 2
    if s != n_partitions:
        note = f"{note + '; ' if note else ''}series too short, S reduced to {s}"
    from .pbo import pbo  # local import: optional Rust backend

    return pbo(m, s).pbo, m.shape[1], note


def _used_partitions(n_partitions: int, rep_t: int) -> int:
    s = n_partitions
    while s > 2 and rep_t // s < 4:
        s -= 2
    return s


def build_report(
    ledger: Ledger,
    records: list[dict[str, Any]] | None = None,
    claimed_n: int | None = None,
    n_partitions: int = 16,
    with_pbo: bool = True,
) -> Report:
    recs = ledger.records() if records is None else records
    trials = _distinct(recs)
    n_unscored = sum(1 for r in recs if not _is_scored(r))
    n_scored = len(trials)
    if n_scored == 0:
        blank = {f.name: None for f in fields(Report)}
        return Report(
            **{
                **blank,
                "n_records": len(recs),
                "n_trials": 0,
                "n_unscored": n_unscored,
                "claimed_n": claimed_n,
            }
        )
    # Every logged run counts towards N; only scored runs inform the variance of Sharpes.
    n = n_scored + n_unscored
    best = max(trials, key=lambda r: r["sharpe"])
    srs = np.array([r["sharpe"] for r in trials], dtype=float)
    var_sr = float(srs.var(ddof=1)) if n_scored > 1 else 0.0
    sr, n_obs = float(best["sharpe"]), int(best["n_obs"] or 0)
    skew = float(best["skew"]) if best.get("skew") is not None else 0.0
    kurt = float(best["kurtosis"]) if best.get("kurtosis") is not None else 3.0
    ppy = float(best.get("periods_per_year") or 252.0)
    psr0 = probabilistic_sharpe_ratio(sr, n_obs, 0.0, skew, kurt) if n_obs > 1 else None
    dsr = deflated_sharpe_ratio(sr, n_obs, n, var_sr, skew, kurt) if n_obs > 1 else None
    dsr_claimed = None
    if claimed_n is not None and n_obs > 1:
        dsr_claimed = deflated_sharpe_ratio(sr, n_obs, max(claimed_n, 1), var_sr, skew, kurt)

    n_eff = dsr_eff = None
    mats = [ledger.load_returns(r["returns_hash"]) for r in trials if r.get("returns_hash")]
    mats = [a for a in mats if a is not None]
    if len(mats) >= 2 and len({len(a) for a in mats}) == 1:
        n_eff = effective_number_of_trials(np.column_stack(mats))
        if n_obs > 1:
            dsr_eff = deflated_sharpe_ratio(sr, n_obs, max(n_eff, 1.0), var_sr, skew, kurt)

    pbo_val = pbo_n = None
    pbo_note = None
    if with_pbo:
        pbo_val, pbo_n, pbo_note = pbo_from_ledger(ledger, trials, n_partitions)

    return Report(
        n_records=len(recs),
        n_trials=n_scored,
        n_unscored=n_unscored,
        best_seq=best.get("seq"),
        best_strategy=best.get("strategy"),
        best_sharpe=sr,
        best_sharpe_ann=annualise_sharpe(sr, ppy),
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurt,
        moments_known=best.get("skew") is not None and best.get("kurtosis") is not None,
        var_trials_sharpe=var_sr,
        psr0=psr0,
        sr0_ledger=expected_max_sharpe(n, var_sr),
        dsr_ledger=dsr,
        claimed_n=claimed_n,
        dsr_claimed=dsr_claimed,
        n_effective=n_eff,
        dsr_effective=dsr_eff,
        pbo=pbo_val,
        pbo_n_trials=pbo_n,
        pbo_partitions=_used_partitions(n_partitions, rep_t=n_obs) if pbo_val is not None else None,
        pbo_note=pbo_note,
    )


def _f(x: float | None, fmt: str = ".3f") -> str:
    return "n/a" if x is None else format(x, fmt)


def format_report(rep: Report, threshold: float = 0.95) -> str:
    lines = [
        f"records in ledger           {rep.n_records}",
        f"distinct scored trials      {rep.n_trials}",
        f"unscored backtest runs      {rep.n_unscored}",
    ]
    if rep.n_trials == 0:
        return "\n".join([*lines, "no trials with a Sharpe ratio yet"])
    verdict_psr = "pass" if (rep.psr0 or 0) > threshold else "fail"
    verdict_dsr = "pass" if (rep.dsr_ledger or 0) > threshold else "fail"
    lines += [
        f"best trial                  #{rep.best_seq} {rep.best_strategy}",
        f"best Sharpe (annualised)    {_f(rep.best_sharpe_ann, '.2f')}"
        f"  (per period {_f(rep.best_sharpe, '.4f')}, T={rep.n_obs}, "
        + (
            f"skew={_f(rep.skew, '.2f')}, kurt={_f(rep.kurtosis, '.2f')})"
            if rep.moments_known
            else "skew/kurtosis unknown: Gaussian assumed)"
        ),
        f"PSR(SR>0), ignores N        {_f(rep.psr0)}  [{verdict_psr} at {threshold}]",
        f"E[max SR | N, null]         {_f(rep.sr0_ledger, '.4f')} per period",
        f"{f'DSR with ledger N={rep.n_trials + (rep.n_unscored or 0)}':<28}"
        f"{_f(rep.dsr_ledger)}  [{verdict_dsr}]",
    ]
    if rep.claimed_n is not None:
        lines.append(f"{f'DSR with claimed N={rep.claimed_n}':<28}{_f(rep.dsr_claimed)}")
    if rep.n_effective is not None:
        label = f"DSR with effective N={rep.n_effective:.1f}"
        lines.append(f"{label:<28}{_f(rep.dsr_effective)}")
    if rep.pbo is not None:
        lines.append(
            f"PBO (CSCV, S={rep.pbo_partitions}, {rep.pbo_n_trials} trials)  {_f(rep.pbo)}"
        )
    elif rep.pbo_note:
        lines.append(f"PBO                         n/a ({rep.pbo_note})")
    if rep.pbo_note and rep.pbo is not None:
        lines.append(f"  note: {rep.pbo_note}")
    return "\n".join(lines)
