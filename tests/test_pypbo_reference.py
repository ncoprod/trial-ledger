"""Agreement with pypbo (esvhd/pypbo @ 4d723f0) on fixed inputs.

The expected values were produced by ``benchmarks/pypbo_reference.py``; pypbo itself is not a
test dependency.
"""

import math

import numpy as np
import pytest

from trial_ledger import stats as S


def fixed_matrix() -> np.ndarray:
    rng = np.random.default_rng(42)
    m = rng.standard_t(5, (483, 12)) * 0.01
    m[:, 3] += 0.0015
    return m


PYPBO = {
    4: {
        "pbo": 0.3333333333333333,
        "prob_oos_loss": 0.5,
        "logits_first5": [-1.7047480922384253, 0.8109302162163288, 1.7047480922384253,
                          0.4700036292457357, -0.8109302162163288],
    },
    8: {
        "pbo": 0.3142857142857143,
        "prob_oos_loss": 0.37142857142857144,
        "logits_first5": [-1.7047480922384253, 2.4849066497880012, -1.7047480922384253,
                          2.4849066497880012, -1.7047480922384253],
    },
}  # fmt: skip


@pytest.mark.parametrize("s", [4, 8])
def test_pbo_matches_pypbo(s):
    res = S.pbo_cscv(fixed_matrix(), s)
    assert res.pbo == pytest.approx(PYPBO[s]["pbo"], abs=1e-12)
    assert res.prob_oos_loss == pytest.approx(PYPBO[s]["prob_oos_loss"], abs=1e-12)
    np.testing.assert_allclose(res.logits[:5], PYPBO[s]["logits_first5"], atol=1e-12)


def test_psr_dsr_match_pypbo():
    assert S.probabilistic_sharpe_ratio(0.08, 750, 0.02, -0.4, 6.0) == pytest.approx(
        0.946321146368199, abs=1e-12
    )
    # pypbo's dsr() takes the standard deviation of trial Sharpes, ours the variance
    assert S.deflated_sharpe_ratio(0.1581, 1250, 100, 0.04472**2, -3, 10) == pytest.approx(
        0.9003484067097614, abs=1e-10
    )
    assert S.expected_max_sharpe(100, 1.0) == pytest.approx(2.5306028932016846, abs=1e-12)
    assert math.isclose(S.expected_max_sharpe(100, 0.04472**2), 0.04472 * 2.5306028932016846)
