import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openai")

from experiments.round3 import metrics, null_strata, transfer
from trial_ledger.backtest import Market, run_backtest
from trial_ledger.hashing import hash_json
from trial_ledger.ledger import Ledger


def test_strata_reconstruct_each_headline_without_losing_failures():
    base = Path("experiments/results/causal-v2")
    rows = json.loads((base / "episodes.json").read_text())["rows"]
    expected = json.loads((base / "paired_tests.json").read_text())["headlines"]
    strata = null_strata(rows)
    for cohort, headline in expected.items():
        subset = [
            s
            for s in strata
            if s["model"] != "random-search" and ("-cli/" in s["model"]) == (cohort == "cli")
        ]
        assert sum(s["attempts"] for s in subset) == headline["attempts"]
        assert sum(s["positive_claims"] for s in subset) == headline["claims_available"]
        assert sum(s["underreported"] for s in subset) == headline["underreported"]
        for field in ("psr0", "dsr_claimed", "dsr_ledger"):
            assert sum(s[field]["n"] for s in subset) == headline[field]["n"]
            assert sum(s[field]["passes"] for s in subset) == headline[field]["passes"]


def test_transfer_keeps_repeated_and_failed_selections_and_boundary_holdings(tmp_path):
    dates = np.arange("2023-09-01", "2024-03-01", dtype="datetime64[D]").astype(str).tolist()
    market = Market("synthetic", np.random.default_rng(2).normal(0, 0.01, len(dates)), dates)
    spec = {"family": "momentum", "lookback": 20, "vol_target": True}
    rows = [
        {
            "model": "subject",
            "market": "btc",
            "seed": i,
            "rep": 0,
            "status": "ok",
            "selected_strategy": spec,
            "submitted_valid": True,
        }
        for i in range(2)
    ]
    rows.append({**rows[0], "seed": 2, "status": "api_error", "submitted_valid": False})
    ledger = Ledger(tmp_path / "ledger")
    result = transfer(rows, market, ledger)
    assert len(result["selections"]) == 3
    assert result["unique_rules"] == 1 and result["evaluations"] == 2
    assert result["per_model"][0]["valid"] == 2
    assert result["per_model"][0]["unique_rules"] == 1
    key = next(k for k in result["candidates"] if k != "buy_and_hold")
    expected = run_backtest(spec, market, eval_from=dates.index("2024-01-01"))
    assert result["candidates"][key]["later"] == metrics(expected.returns)
    benchmark = market.returns[1:].copy()
    benchmark[0] -= market.fee_bps * 1e-4
    np.testing.assert_array_equal(
        ledger.load_returns(result["candidates"]["buy_and_hold"]["full_returns_hash"]), benchmark
    )
    assert len(ledger.records()) == 2
    assert ledger.verify(check_returns=True).ok
    assert transfer(rows, market, ledger) == result
    assert len(ledger.records()) == 2
    assert all(r["period"]["time_axis_hash"] == hash_json(dates[1:]) for r in ledger.records())
    market.returns[-1] += 0.01
    with pytest.raises(ValueError, match="replay differs"):
        transfer(rows, market, ledger)
