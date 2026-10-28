import math
from itertools import combinations

import numpy as np
import pytest
from scipy import stats as sps

from trial_ledger import stats as S

# ---------------------------------------------------------------------------------------
# Known values from Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", section 6:
# annualised SR = 2.5 over 5 years of daily data (T = 1250), N = 100 trials whose annualised
# Sharpe ratios have variance 0.5, returns with skewness -3 and kurtosis 10.
# The paper reports E[max SR] = 0.1132 (non-annualised) and DSR = 0.9004.
# ---------------------------------------------------------------------------------------


def test_expected_max_sharpe_paper_example():
    sr0 = S.expected_max_sharpe(100, 0.5 / 250)
    assert sr0 == pytest.approx(0.1132, abs=5e-5)


def test_deflated_sharpe_paper_example():
    dsr = S.deflated_sharpe_ratio(
        sr=2.5 / math.sqrt(250),
        n_obs=1250,
        n_trials=100,
        var_trials_sr=0.5 / 250,
        skew=-3.0,
        kurtosis=10.0,
    )
    assert dsr == pytest.approx(0.9004, abs=5e-4)


def test_expected_max_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for n in (10, 100, 1000):
        sims = rng.standard_normal((4000, n)).max(axis=1).mean()
        assert S.expected_max_sharpe(n, 1.0) == pytest.approx(sims, rel=0.03)


def test_expected_max_single_trial_has_no_selection():
    assert S.expected_max_sharpe(1, 0.3) == 0.0


def test_psr_basic_properties():
    assert S.probabilistic_sharpe_ratio(0.0, 500) == pytest.approx(0.5)
    a = S.probabilistic_sharpe_ratio(0.05, 500)
    b = S.probabilistic_sharpe_ratio(0.10, 500)
    assert 0.5 < a < b < 1
    # negative skew and fat tails widen the standard error -> lower PSR
    assert S.probabilistic_sharpe_ratio(0.1, 500, skew=-2, kurtosis=12) < b


def test_psr_closed_form():
    sr, n, g3, g4, sr_star = 0.08, 750, -0.4, 6.0, 0.02
    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    assert S.probabilistic_sharpe_ratio(sr, n, sr_star, g3, g4) == pytest.approx(sps.norm.cdf(z))


def test_psr_is_calibrated_under_the_null():
    """With i.i.d. Gaussian returns and true SR = 0, PSR(0) > 0.95 about 5% of the time."""
    rng = np.random.default_rng(1)
    hits = 0
    n_sims = 4000
    for _ in range(n_sims):
        r = rng.standard_normal(500)
        m = S.sharpe_moments(r)
        hits += S.probabilistic_sharpe_ratio(m.sharpe, m.n_obs, 0.0, m.skew, m.kurtosis) > 0.95
    assert hits / n_sims == pytest.approx(0.05, abs=0.012)


def test_dsr_controls_selection_under_the_null():
    """Best-of-N under the null: PSR is fooled, DSR with the true N is not."""
    rng = np.random.default_rng(2)
    n_trials, t = 50, 500
    psr_hits = dsr_hits = 0
    reps = 300
    for _ in range(reps):
        m = rng.standard_normal((t, n_trials))
        srs = m.mean(0) / m.std(0, ddof=1)
        k = int(np.argmax(srs))
        mm = S.sharpe_moments(m[:, k])
        psr_hits += S.probabilistic_sharpe_ratio(mm.sharpe, t, 0, mm.skew, mm.kurtosis) > 0.95
        dsr_hits += (
            S.deflated_sharpe_ratio(mm.sharpe, t, n_trials, srs.var(ddof=1), mm.skew, mm.kurtosis)
            > 0.95
        )
    assert psr_hits / reps > 0.6
    assert dsr_hits / reps < 0.1


def test_sharpe_moments_gaussian():
    r = np.random.default_rng(3).normal(0.01, 0.02, 200_000)
    m = S.sharpe_moments(r)
    assert m.sharpe == pytest.approx(0.5, abs=0.01)
    assert m.skew == pytest.approx(0, abs=0.02)
    assert m.kurtosis == pytest.approx(3, abs=0.05)


def test_effective_number_of_trials():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((1000, 1))
    assert S.effective_number_of_trials(np.repeat(x, 10, axis=1)) == pytest.approx(1.0)
    ind = rng.standard_normal((5000, 10))
    assert S.effective_number_of_trials(ind) == pytest.approx(10, abs=0.3)


# ---------------------------------------------------------------------------------------
# PBO / CSCV
# ---------------------------------------------------------------------------------------


def _sharpe_cols(x):
    return x.mean(0) / x.std(0, ddof=1)


def test_cscv_number_of_splits():
    assert sum(1 for _ in S.cscv_splits(16)) == math.comb(16, 8)
    with pytest.raises(ValueError):
        list(S.cscv_splits(7))


def test_pbo_fast_path_equals_generic_metric():
    rng = np.random.default_rng(5)
    m = rng.standard_normal((1003, 25)) * 0.01
    fast = S.pbo_cscv(m, 10)
    slow = S.pbo_cscv(m, 10, metric=_sharpe_cols)
    assert fast.pbo == slow.pbo
    np.testing.assert_allclose(fast.logits, slow.logits)
    np.testing.assert_allclose(fast.oos_perf_of_best, slow.oos_perf_of_best, rtol=1e-9)


def test_pbo_by_hand_tiny_example():
    """S = 2: two splits, each half is IS once. Check against a manual computation."""
    m = np.array(
        [
            [0.01, 0.00, -0.02],
            [0.03, 0.01, 0.00],
            [0.02, -0.01, 0.01],
            [-0.01, 0.02, 0.03],
            [0.00, 0.01, 0.02],
            [-0.02, 0.03, 0.04],
        ]
    )
    res = S.pbo_cscv(m, 2)
    a, b = m[:3], m[3:]
    expected = []
    for is_, oos in ((a, b), (b, a)):
        best = int(np.argmax(_sharpe_cols(is_)))
        oos_rank = sps.rankdata(_sharpe_cols(oos))[best]
        w = oos_rank / 4
        expected.append(math.log(w / (1 - w)))
    np.testing.assert_allclose(res.logits, expected)
    assert res.pbo == np.mean(np.array(expected) <= 0)


def test_pbo_noise_is_high_and_skill_is_low():
    rng = np.random.default_rng(6)
    noise = rng.standard_normal((1600, 50)) * 0.01
    assert S.pbo_cscv(noise, 16).pbo > 0.25
    skilled = noise.copy()
    skilled[:, 7] += 0.004  # one strategy with a real edge (per-period SR ~ 0.4)
    assert S.pbo_cscv(skilled, 16).pbo < 0.05


def test_pbo_drops_leading_rows():
    rng = np.random.default_rng(7)
    m = rng.standard_normal((105, 4))
    a = S.pbo_cscv(m, 4)
    b = S.pbo_cscv(m[1:], 4)  # 105 % 4 == 1 -> the first row is ignored
    np.testing.assert_allclose(a.logits, b.logits)


def test_pbo_splits_cover_every_combination_once():
    seen = {tuple(sorted(tr)) for tr, _ in S.cscv_splits(6)}
    assert seen == set(combinations(range(6), 3))


# ---------------------------------------------------------------------------------------
# Purged / embargoed K-fold
# ---------------------------------------------------------------------------------------


def test_purged_kfold_no_leakage():
    n, h, emb = 100, 3, 2
    folds = list(S.purged_kfold(n, 5, label_horizon=h, embargo=emb))
    assert len(folds) == 5
    covered = np.concatenate([te for _, te in folds])
    np.testing.assert_array_equal(np.sort(covered), np.arange(n))
    for train, test in folds:
        a, b = test[0], test[-1] + 1
        assert not set(train) & set(test)
        # no training label window [i, i+h] overlaps the test fold
        assert not any(a <= i + h and i < b for i in train)
        # embargo after the test fold
        assert not any(b <= i < b + emb for i in train)


def test_purged_kfold_fractional_embargo():
    train, test = next(S.purged_kfold(1000, 4, embargo=0.01))
    assert test[-1] == 249
    assert train.min() == 260
