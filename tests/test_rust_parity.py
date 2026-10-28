"""Parity of the Rust CSCV core with the NumPy reference (skipped if not built)."""

import numpy as np
import pytest

from trial_ledger.stats import pbo_cscv

core = pytest.importorskip("trial_ledger_core")

from trial_ledger.pbo import pbo  # noqa: E402


@pytest.mark.parametrize(
    ("t", "n", "s", "seed"),
    [(1000, 2, 2, 0), (1003, 25, 10, 1), (2000, 100, 16, 2), (517, 7, 8, 3), (400, 300, 6, 4)],
)
def test_rust_matches_python(t, n, s, seed):
    rng = np.random.default_rng(seed)
    m = rng.standard_t(4, (t, n)) * 0.01
    m[:, 0] += 0.001 * seed  # sometimes a genuine edge
    py = pbo_cscv(m, s)
    rs = pbo(m, s, backend="rust")
    assert rs.pbo == py.pbo
    assert rs.n_splits == py.n_splits
    np.testing.assert_allclose(rs.logits, py.logits, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rs.is_best_perf, py.is_best_perf, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(rs.oos_perf_of_best, py.oos_perf_of_best, rtol=1e-10, atol=1e-14)


def test_rust_handles_non_contiguous_and_ties():
    m = np.random.default_rng(5).standard_normal((300, 20))
    m[:, 3] = m[:, 4]  # exact duplicate trial -> ties
    view = np.asfortranarray(m)
    np.testing.assert_allclose(pbo(view, 6, "rust").logits, pbo_cscv(m, 6).logits, atol=1e-12)


def test_rust_errors():
    with pytest.raises(ValueError):
        core.pbo_cscv_sharpe(np.zeros((100, 3)), 5)
    with pytest.raises(ValueError):
        core.pbo_cscv_sharpe(np.zeros((100, 1)), 4)


def test_auto_backend_uses_rust():
    m = np.random.default_rng(6).standard_normal((200, 5))
    assert pbo(m, 4).meta.get("backend") == "rust"
