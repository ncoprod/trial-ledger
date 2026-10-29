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


def test_csv_preserves_acceptance_boundary(tmp_path, monkeypatch):
    import csv

    from experiments import analyze

    monkeypatch.setattr(analyze, "RESULTS", tmp_path)
    row = {"psr0": 0.950000001, "dsr_claimed": 0.950000002, "dsr_ledger": 0.949999999}
    analyze.write_tables([row], [{"episodes": 1}])
    with (tmp_path / "episodes.csv").open() as stream:
        restored = next(csv.DictReader(stream))
    for key, value in row.items():
        assert float(restored[key]) == value
        assert (float(restored[key]) > analyze.THRESH) == (value > analyze.THRESH)


@pytest.mark.parametrize("status", ["ok", "invalid_submission"])
def test_invalid_submission_preserves_counts_without_substitute_pick(tmp_path, monkeypatch, status):
    import json

    from experiments.analyze import analyze_episode
    from trial_ledger.backtest import run_backtest
    from trial_ledger.ledger import Ledger

    market, _, _ = make_market("signflip", 0)
    result = run_backtest({"family": "momentum"}, market)
    ledger = Ledger(tmp_path / "ledger")
    ledger.append(strategy=result.spec, returns=result.returns)
    ep = {
        "model": "test",
        "market": "signflip",
        "seed": 0,
        "rep": 0,
        "status": status,
        "tool_calls": 1,
        "ledger_head": " ".join(map(str, ledger.head())),
        "submission": {"strategy": {"family": "invalid"}, "n_strategies_tested": 7},
    }
    (tmp_path / "episode.json").write_text(json.dumps(ep))
    row = analyze_episode(tmp_path)
    assert row["n_ledger"] == 1 and row["claimed_n"] == 7
    assert not row["submitted_valid"] and "psr0" not in row
    assert row["status"] == "invalid_submission"
    from experiments import paired_tests

    monkeypatch.setattr(paired_tests, "RESULTS", tmp_path)
    paired_tests.main([row])
    result = json.loads((tmp_path / "paired_tests.json").read_text())["headlines"]["nim"]
    assert result["attempts"] == 1 and result["episodes"] == 0


def test_shared_return_migration_preserves_anchor_and_bytes(tmp_path, monkeypatch):
    import json

    from experiments import store_returns
    from trial_ledger.backtest import run_backtest
    from trial_ledger.ledger import Ledger

    monkeypatch.setattr(store_returns, "RESULTS", tmp_path)
    run = tmp_path / "runs/test/signflip-s0-r0"
    ledger = Ledger(run / "ledger")
    market, _, _ = make_market("signflip", 0)
    result = run_backtest({"family": "momentum"}, market)
    record = ledger.append(strategy=result.spec, returns=result.returns)
    head, original = ledger.head(), ledger.path.read_bytes()
    (run / "episode.json").write_text(
        json.dumps(
            {
                "market": "signflip",
                "seed": 0,
                "ledger_head": " ".join(map(str, head)),
            }
        )
    )
    store_returns.main()
    migrated = Ledger(run / "ledger")
    assert migrated.verify(expect_head=" ".join(map(str, head)), check_returns=True).ok
    assert migrated.path.read_bytes() == original
    assert hash_array(migrated.load_returns(record["returns_hash"])) == record["returns_hash"]
    assert not list((run / "ledger/returns").glob("*.npz"))
    assert len(list((tmp_path / "returns").glob("*.npz"))) == 1
