"""Multiple-testing statistics for backtests.

References
----------
* Bailey, D. H. and López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier".
  Journal of Risk 15(2). -- Probabilistic Sharpe Ratio (PSR).
* Bailey, D. H. and López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality". Journal of Portfolio Management
  40(5). -- Deflated Sharpe Ratio (DSR) and the expected maximum Sharpe ratio.
* Bailey, D. H., Borwein, J., López de Prado, M. and Zhu, Q. J. (2017). "The Probability of
  Backtest Overfitting". Journal of Computational Finance 20(4). -- PBO via CSCV.
* López de Prado, M. (2018). "Advances in Financial Machine Learning", ch. 7 -- purged and
  embargoed K-fold cross-validation.

Conventions
-----------
All Sharpe ratios passed to PSR/DSR are *per-period* (not annualised) and estimated on the
same sampling frequency as ``n_obs``. ``kurtosis`` is the *raw* (non-excess) fourth
standardised moment, so a Gaussian has kurtosis 3.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------------------
# Moments and Sharpe ratio
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SharpeMoments:
    sharpe: float  # per period, ddof=1 standard deviation
    skew: float
    kurtosis: float  # raw (Gaussian = 3)
    n_obs: int


def sharpe_ratio(returns: np.ndarray, ddof: int = 1) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=ddof)
    if sd == 0.0 or not np.isfinite(sd):
        return 0.0
    return float(r.mean() / sd)


def sharpe_moments(returns: np.ndarray) -> SharpeMoments:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 3 or r.std() == 0.0:
        return SharpeMoments(sharpe_ratio(r) if n >= 2 else float("nan"), 0.0, 3.0, n)
    return SharpeMoments(
        sharpe=sharpe_ratio(r),
        skew=float(sps.skew(r, bias=True)),
        kurtosis=float(sps.kurtosis(r, fisher=False, bias=True)),
        n_obs=n,
    )


def annualise_sharpe(sr: float, periods_per_year: float) -> float:
    return float(sr * math.sqrt(periods_per_year))


# --------------------------------------------------------------------------------------
# PSR / DSR
# --------------------------------------------------------------------------------------


def sharpe_std_error(sr: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """Standard error of the per-period Sharpe estimator under non-normal returns.

    sigma(SR) = sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (n - 1))   (Mertens 2002; BLdP 2012)
    """
    var = (1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2) / (n_obs - 1)
    return math.sqrt(max(var, 1e-300))


def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: int,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """PSR(SR*) = Phi((SR - SR*) * sqrt(n-1) / sqrt(1 - g3*SR + (g4-1)/4*SR^2)).

    Probability that the true Sharpe ratio exceeds ``sr_benchmark`` given an observed ``sr``
    estimated on ``n_obs`` observations with the given skewness and raw kurtosis.
    """
    if n_obs < 2 or not np.isfinite(sr):
        return float("nan")
    z = (sr - sr_benchmark) / sharpe_std_error(sr, n_obs, skew, kurtosis)
    return float(sps.norm.cdf(z))


def expected_max_sharpe(
    n_trials: float, var_trials_sr: float, mean_trials_sr: float = 0.0
) -> float:
    """Expected maximum of ``n_trials`` Sharpe ratios under the null (BLdP 2014, eq. 8 / prop. 1).

    E[max SR] ~= E[SR] + sqrt(V[SR]) * ((1-g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e)))

    ``var_trials_sr`` is the cross-sectional variance of the per-period Sharpe ratios of the
    trials. With a single trial there is no selection, so the threshold is ``mean_trials_sr``.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return float(mean_trials_sr)
    sd = math.sqrt(max(var_trials_sr, 0.0))
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(mean_trials_sr + sd * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    n_trials: float,
    var_trials_sr: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR = PSR(SR0) with SR0 the expected maximum Sharpe ratio across ``n_trials`` trials.

    Following BLdP (2014) the null mean of the trials' Sharpe ratios is zero.
    """
    sr0 = expected_max_sharpe(n_trials, var_trials_sr)
    return probabilistic_sharpe_ratio(sr, n_obs, sr0, skew, kurtosis)


def effective_number_of_trials(returns_matrix: np.ndarray) -> float:
    """Rough effective number of independent trials from the mean pairwise correlation.

    N_eff = rho + (1 - rho) * N  (the 'average correlation' heuristic discussed in
    López de Prado (2019), "A Data Science Solution to the Multiple-Testing Crisis in
    Financial Research"). Correlated trials (e.g. neighbouring grid points) count less.
    """
    m = np.asarray(returns_matrix, dtype=float)
    n = m.shape[1]
    if n < 2:
        return float(n)
    sd = m.std(axis=0)
    keep = sd > 0
    m = m[:, keep]
    n_eff_cols = m.shape[1]
    if n_eff_cols < 2:
        return 1.0
    c = np.corrcoef(m, rowvar=False)
    rho = (c.sum() - n_eff_cols) / (n_eff_cols * (n_eff_cols - 1))
    rho = float(np.clip(rho, 0.0, 1.0))
    return rho + (1.0 - rho) * n


# --------------------------------------------------------------------------------------
# PBO via combinatorially symmetric cross-validation (CSCV)
# --------------------------------------------------------------------------------------


@dataclass
class PBOResult:
    pbo: float
    logits: np.ndarray
    is_best_perf: np.ndarray  # in-sample performance of the selected trial, per split
    oos_perf_of_best: np.ndarray  # out-of-sample performance of that trial, per split
    n_splits: int
    n_partitions: int
    n_trials: int
    meta: dict = field(default_factory=dict)

    @property
    def prob_oos_loss(self) -> float:
        return float(np.mean(self.oos_perf_of_best < 0.0))


def _sharpe_from_sums(s: np.ndarray, ss: np.ndarray, n: int) -> np.ndarray:
    """Per-column Sharpe (ddof=1) from sums and sums of squares over ``n`` rows."""
    mean = s / n
    var = (ss - n * mean**2) / (n - 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, mean / sd, 0.0)
    return out


def _relative_rank_logit(test_perf: np.ndarray, best: int) -> float:
    """logit(w) with w = rank of the IS-best trial among OOS performances / (N + 1).

    Ranks are 1..N in ascending order of OOS performance; ties get the average rank.
    """
    n = test_perf.size
    v = test_perf[best]
    rank = np.sum(test_perf < v) + 0.5 * (np.sum(test_perf == v) - 1) + 1.0
    w = rank / (n + 1.0)
    return float(math.log(w / (1.0 - w)))


def cscv_splits(n_partitions: int) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    """All C(S, S/2) (train, test) partitions of ``range(n_partitions)``."""
    if n_partitions % 2 != 0 or n_partitions < 2:
        raise ValueError("n_partitions must be an even integer >= 2")
    parts = tuple(range(n_partitions))
    for train in combinations(parts, n_partitions // 2):
        test = tuple(p for p in parts if p not in train)
        yield train, test


def _partition_bounds(n_rows: int, n_partitions: int) -> np.ndarray:
    """Row boundaries of ``n_partitions`` contiguous, equal-sized blocks.

    All blocks have length ``n_rows // n_partitions``; the ``n_rows % n_partitions`` leading
    (oldest) rows are dropped, as in pypbo, so the most recent data is always used.
    """
    size = n_rows // n_partitions
    if size < 2:
        raise ValueError("each partition needs at least 2 rows")
    return n_rows % n_partitions + np.arange(n_partitions + 1) * size


def pbo_cscv(
    returns_matrix: np.ndarray,
    n_partitions: int = 16,
    metric: Callable[[np.ndarray], np.ndarray] | None = None,
) -> PBOResult:
    """Probability of Backtest Overfitting via CSCV (Bailey, Borwein, López de Prado, Zhu 2017).

    Reference implementation in NumPy.

    Parameters
    ----------
    returns_matrix : (T, N) array of per-period returns of N trials over the same T periods.
    n_partitions   : S, the (even) number of contiguous blocks; C(S, S/2) splits are evaluated.
    metric         : maps a (t, N) block of returns to N performance values. Defaults to the
                     Sharpe ratio (ddof=1), computed from per-block sufficient statistics,
                     which is what the Rust core implements.
    """
    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("returns_matrix must be (T, N) with N >= 2")
    t, n = m.shape
    bounds = _partition_bounds(t, n_partitions)
    size = int(bounds[1] - bounds[0])
    logits, is_best, oos_best = [], [], []

    if metric is None:
        # sufficient statistics per block: (S, N)
        blocks = m[bounds[0] : bounds[-1]].reshape(n_partitions, size, n)
        bsum = blocks.sum(axis=1)
        bsq = (blocks**2).sum(axis=1)
        half = (n_partitions // 2) * size
        for train, test in cscv_splits(n_partitions):
            tr = _sharpe_from_sums(bsum[list(train)].sum(0), bsq[list(train)].sum(0), half)
            te = _sharpe_from_sums(bsum[list(test)].sum(0), bsq[list(test)].sum(0), half)
            best = int(np.argmax(tr))
            logits.append(_relative_rank_logit(te, best))
            is_best.append(tr[best])
            oos_best.append(te[best])
    else:
        blocks = [m[bounds[i] : bounds[i + 1]] for i in range(n_partitions)]
        for train, test in cscv_splits(n_partitions):
            tr = np.asarray(metric(np.vstack([blocks[i] for i in train])), dtype=float)
            te = np.asarray(metric(np.vstack([blocks[i] for i in test])), dtype=float)
            best = int(np.argmax(tr))
            logits.append(_relative_rank_logit(te, best))
            is_best.append(tr[best])
            oos_best.append(te[best])

    logits_a = np.asarray(logits)
    return PBOResult(
        pbo=float(np.mean(logits_a <= 0.0)),
        logits=logits_a,
        is_best_perf=np.asarray(is_best),
        oos_perf_of_best=np.asarray(oos_best),
        n_splits=len(logits),
        n_partitions=n_partitions,
        n_trials=n,
    )


# --------------------------------------------------------------------------------------
# Purged / embargoed K-fold
# --------------------------------------------------------------------------------------


def purged_kfold(
    n_obs: int,
    n_splits: int = 5,
    label_horizon: int = 0,
    embargo: int | float = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Purged and embargoed K-fold split for time-ordered samples (López de Prado 2018, ch. 7).

    Sample ``i`` is assumed to have a label that depends on observations ``[i, i + label_horizon]``.
    For each contiguous test fold ``[a, b)``:

    * purging removes training samples whose label window overlaps the test fold, i.e. samples
      ``i`` in ``[a - label_horizon, a)``;
    * the embargo additionally removes ``embargo`` samples right after the test fold,
      ``[b, b + embargo)`` (``embargo`` may be given as a fraction of ``n_obs``).
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if isinstance(embargo, float) and embargo < 1:
        emb = math.ceil(embargo * n_obs)
    else:
        emb = int(embargo)
    idx = np.arange(n_obs)
    for test in np.array_split(idx, n_splits):
        a, b = int(test[0]), int(test[-1]) + 1
        lo = max(0, a - label_horizon)
        hi = min(n_obs, b + emb)
        mask = np.ones(n_obs, dtype=bool)
        mask[lo:hi] = False
        yield idx[mask], test
