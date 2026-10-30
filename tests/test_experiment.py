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


def test_startup_api_failure_has_a_verifiable_empty_ledger(tmp_path):
    from experiments.agent import run_episode
    from experiments.analyze import analyze_episode
    from trial_ledger.ledger import Ledger

    episode = run_episode(
        "tests/no-startup-response", "signflip", 0, offline=True, results_dir=tmp_path
    )
    directory = tmp_path / "runs/tests__no-startup-response/signflip-s0-r0"
    assert episode["status"] == "api_error: CacheMiss"
    assert episode["ledger_records"] == 0
    assert (
        Ledger(directory / "ledger")
        .verify(expect_head=episode["ledger_head"], check_returns=True)
        .ok
    )
    row = analyze_episode(directory)
    assert row["status"] == episode["status"] and row["n_ledger"] == 0
    assert "psr0" not in row and not row["submitted_valid"]


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


@pytest.mark.parametrize(
    ("status", "family", "valid"),
    [
        ("ok", "invalid", False),
        ("invalid_submission", "invalid", False),
        ("timeout", "momentum", True),
        ("cli_error", "momentum", True),
    ],
)
def test_failed_submission_preserves_counts_without_substitute_pick(
    tmp_path, monkeypatch, status, family, valid
):
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
        "submission": {"strategy": {"family": family}, "n_strategies_tested": 7},
    }
    (tmp_path / "episode.json").write_text(json.dumps(ep))
    row = analyze_episode(tmp_path)
    assert row["n_ledger"] == 1 and row["claimed_n"] == 7
    assert row["submitted_valid"] == valid and "psr0" not in row
    assert row["status"] == ("invalid_submission" if status == "ok" else status)
    from experiments import paired_tests

    monkeypatch.setattr(paired_tests, "RESULTS", tmp_path)
    paired_tests.main([row])
    result = json.loads((tmp_path / "paired_tests.json").read_text())["headlines"]["nim"]
    assert result["attempts"] == 1 and result["episodes"] == 0
    assert result["claims_available"] == 1 and result["underreported"] == 0


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


@pytest.mark.parametrize("corruption", ["score", "return_hash", "tail"])
def test_analysis_refuses_corrupted_episode(tmp_path, corruption):
    import json

    from experiments.analyze import analyze_episode
    from trial_ledger.backtest import run_backtest
    from trial_ledger.ledger import Ledger

    market, _, _ = make_market("garch", 10)
    ledger = Ledger(tmp_path / "ledger")
    for spec in [{"family": "momentum", "lookback": 20}, {"family": "momentum", "lookback": 30}]:
        result = run_backtest(spec, market)
        ledger.append(strategy=result.spec, returns=result.returns)
    ep = dict(
        model="test",
        market="garch",
        seed=10,
        rep=0,
        status="ok",
        tool_calls=2,
        ledger_head=" ".join(map(str, ledger.head())),
        submission={"strategy": {"family": "momentum"}, "n_strategies_tested": 2},
    )
    (tmp_path / "episode.json").write_text(json.dumps(ep))
    records = ledger.records()
    if corruption == "score":
        records[0]["sharpe"] = 10
    elif corruption == "return_hash":
        records[0]["returns_hash"] = "0" * 64
    else:
        records.pop()
    ledger.path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="verification failed"):
        analyze_episode(tmp_path)


def test_reproduce_does_not_publish_after_failed_replay(monkeypatch):
    from experiments import run

    monkeypatch.setattr(run, "cmd_replay_check", lambda args: 1)

    def unexpected(*args):
        pytest.fail("publication attempted after failed replay")

    monkeypatch.setattr(run, "cmd_analyze", unexpected)
    monkeypatch.setattr(run, "cmd_figures", unexpected)
    assert run.cmd_reproduce(None) == 1


def test_summaries_and_paired_tests_refuse_failed_integrity(tmp_path, monkeypatch):
    from experiments import analyze, paired_tests

    monkeypatch.setattr(analyze, "RESULTS", tmp_path)
    monkeypatch.setattr(paired_tests, "RESULTS", tmp_path)
    for row in [{"ledger_ok": False}, {"replay_mismatches": 1}]:
        for call in [analyze.summarise, paired_tests.main]:
            with pytest.raises(ValueError, match="unverified episode"):
                call([row])
        with pytest.raises(ValueError, match="unverified episode"):
            analyze.write_tables([row], [])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("tamper_transcript", [False, True])
def test_failed_provider_attempt_audits_exact_cached_prefix(
    tmp_path, monkeypatch, capsys, tamper_transcript
):
    import json

    from experiments import agent, llm, run
    from trial_ledger.ledger import Ledger
    from trial_ledger.service import BacktestService

    results = tmp_path / "results"
    monkeypatch.setattr(agent, "RESULTS", results)
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "_CACHES", {})
    model = "test-model"
    client = llm.ChatClient(model)
    market, _, _ = make_market("signflip", 0)
    description = BacktestService(market, Ledger(tmp_path / "unused")).describe()
    messages = [
        {"role": "system", "content": agent.system_prompt(description)},
        {"role": "user", "content": agent.USER_TASK},
    ]
    response = {
        "message": {
            "content": None,
            "tool_calls": [
                {
                    "id": "fixture-call",
                    "type": "function",
                    "function": {
                        "name": "run_backtest",
                        "arguments": json.dumps({"strategy": {"family": "momentum"}}),
                    },
                }
            ],
        }
    }
    client.cache.put(client.request_key(messages, agent.TOOLS, "signflip-s0-r0"), response)
    cached_chat = llm.ChatClient.chat

    def outage_after_cached_response(self, messages, tools, salt):
        if self.cache.get(self.request_key(messages, tools, salt)) is None:
            raise RuntimeError("simulated provider outage")
        return cached_chat(self, messages, tools, salt)

    monkeypatch.setattr(llm.ChatClient, "chat", outage_after_cached_response)
    episode = agent.run_episode(model, "signflip", 0)
    assert episode["status"] == "api_error: RuntimeError" and episode["ledger_records"] == 1
    monkeypatch.setattr(llm.ChatClient, "chat", cached_chat)
    transcript = results / "runs/test-model/signflip-s0-r0/transcript.json"
    if tamper_transcript:
        messages = json.loads(transcript.read_text())
        messages[-1]["content"] = "modified feedback"
        transcript.write_text(json.dumps(messages))
    assert run.cmd_replay_check(None) == int(tamper_transcript)
    assert "failed cache-prefix audits" in capsys.readouterr().out
