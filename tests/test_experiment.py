"""Sanity checks for the experiment's markets and harness (no API calls)."""

import numpy as np
import pytest

pytest.importorskip("openai")

from experiments.agent import blind
from experiments.markets import (
    N_NULL,
    btc_split,
    garch_returns,
    make_market,
    signflip_returns,
)
from trial_ledger.hashing import hash_array


def test_garch_frozen_calibration_reproduces_original_market():
    # Original market fingerprint from the committed GARCH seed-0 ledger.
    import json
    from pathlib import Path

    path = Path("experiments/results/runs/random-search/garch-s0-r0/ledger/ledger.jsonl")
    original = json.loads(path.read_text().splitlines()[0])
    market, _, _ = make_market("garch", 0)
    assert hash_array(market.returns) == original["dataset_hash"]


def test_null_markets_are_deterministic_and_sized():
    np.testing.assert_array_equal(signflip_returns(3), signflip_returns(3))
    assert not np.array_equal(signflip_returns(3), signflip_returns(4))
    np.testing.assert_array_equal(garch_returns(3), garch_returns(3))
    for kind in ("signflip", "garch"):
        m, full, oos = make_market(kind, 0)
        assert m.returns.size == N_NULL + 1 and full is None and oos is None


def test_signflip_has_symmetric_signs_and_real_magnitudes():
    r = signflip_returns(0, n=200_000)
    assert abs(np.mean(r > 0) - 0.5) < 0.01
    # sign is independent of the previous return (no exploitable autocorrelation)
    assert abs(np.corrcoef(np.sign(r[1:]), r[:-1])[0, 1]) < 0.01
    # volatility clustering survives in |r|
    a = np.abs(r)
    assert np.corrcoef(a[1:], a[:-1])[0, 1] > 0.05


def test_btc_split_dates():
    full, oos = btc_split()
    assert full.dates[0] == "2018-01-01" and full.dates[oos] == "2024-01-01"
    assert oos == 2191


def test_blind_hides_trial_ids():
    res = {"trial": 7, "sharpe_annualized": 1.0, "top": [{"trial": 3, "x": 1}]}
    assert blind(res) == {"sharpe_annualized": 1.0, "top": [{"x": 1}]}
