"""The N comparison tests decisions separately from continuous-score differences."""

import pytest

pytest.importorskip("openai")
from experiments.paired_tests import paired
from trial_ledger.stats import deflated_sharpe_ratio


def test_zero_discordance_does_not_imply_equal_scores():
    result = paired([{"dsr_claimed": 0.9, "dsr_ledger": 0.5}] * 12)
    assert result["claimed_only_pass"] == result["ledger_only_pass"] == 0
    assert result["mcnemar_exact_two_sided_p"] == 1.0
    assert result["median_claimed_minus_ledger"] == pytest.approx(0.4)
    assert result["wilcoxon_two_sided_p"] < 0.01


def test_severe_undercount_can_change_the_095_decision():
    # Controlled sensitivity case; these are not reported as observed agent outcomes.
    claimed = deflated_sharpe_ratio(0.15, 1000, 2, 0.01)
    logged = deflated_sharpe_ratio(0.15, 1000, 500, 0.01)
    assert claimed > 0.95 > logged
    result = paired([{"dsr_claimed": claimed, "dsr_ledger": logged}] * 8)
    assert result["claimed_only_pass"] == 8
    assert result["ledger_only_pass"] == 0
    assert result["mcnemar_exact_two_sided_p"] == pytest.approx(2 / 2**8)
