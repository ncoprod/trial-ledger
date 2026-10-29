import itertools
import json
import threading

import numpy as np
import pytest

from trial_ledger.hashing import GENESIS_HASH, canonical_json, hash_array
from trial_ledger.ledger import Ledger


@pytest.fixture
def ledger(tmp_path):
    lg = Ledger(tmp_path / "tl")
    rng = np.random.default_rng(0)
    for i in range(5):
        lg.append(
            strategy={"family": "momentum", "lookback": 10 + i},
            returns=rng.standard_normal(300) * 0.01,
            periods_per_year=365,
            agent="test",
        )
    return lg


def _rewrite(lg: Ledger, lines: list[str]) -> None:
    lg.path.write_text("".join(lines))


def test_clean_ledger_verifies(ledger):
    res = ledger.verify(check_returns=True)
    assert res.ok, res.issues
    assert res.n_records == 5
    assert ledger.head() == (5, res.head_hash)


def test_chain_links(ledger):
    recs = ledger.records()
    assert recs[0]["prev_hash"] == GENESIS_HASH
    for a, b in itertools.pairwise(recs):
        assert b["prev_hash"] == a["hash"]
    assert [r["seq"] for r in recs] == [1, 2, 3, 4, 5]


def test_record_contents(ledger):
    rec = ledger.records()[0]
    arr = ledger.load_returns(rec["returns_hash"])
    assert arr is not None and hash_array(arr) == rec["returns_hash"]
    assert rec["n_obs"] == 300
    assert rec["sharpe_ann"] == pytest.approx(rec["sharpe"] * np.sqrt(365))
    assert rec["kurtosis"] > 0


def test_detects_modified_value(ledger):
    lines = ledger.path.read_text().splitlines(keepends=True)
    rec = json.loads(lines[2])
    rec["sharpe"] = 0.5  # make a trial look better
    lines[2] = canonical_json(rec) + "\n"
    _rewrite(ledger, lines)
    kinds = {i.kind for i in ledger.verify().issues}
    assert "hash_mismatch" in kinds


def test_detects_deleted_record(ledger):
    lines = ledger.path.read_text().splitlines(keepends=True)
    del lines[1]
    _rewrite(ledger, lines)
    res = ledger.verify()
    kinds = {i.kind for i in res.issues}
    assert {"broken_link", "seq_gap"} <= kinds
    assert res.issues[0].line == 2


def test_detects_reordering(ledger):
    lines = ledger.path.read_text().splitlines(keepends=True)
    lines[1], lines[2] = lines[2], lines[1]
    _rewrite(ledger, lines)
    assert not ledger.verify().ok


def test_detects_tail_truncation_with_anchor(ledger):
    seq, head = ledger.head()
    lines = ledger.path.read_text().splitlines(keepends=True)
    _rewrite(ledger, lines[:-2])
    # the chain itself is still internally consistent ...
    res = ledger.verify(expect_head=f"{seq} {head}")
    assert [i.kind for i in res.issues] == ["head_mismatch"]
    # ... and the HEAD file written at append time also catches it
    assert [i.kind for i in ledger.verify().issues] == ["head_mismatch"]


def test_rehashed_forgery_is_caught_by_external_anchor(ledger):
    """An attacker who rewrites a value and rebuilds the whole chain + HEAD still fails
    against a head hash anchored elsewhere before the edit."""
    from trial_ledger.hashing import chain_hash

    anchored = " ".join(map(str, ledger.head()))
    recs = ledger.records()
    recs[0]["sharpe"] = 1.0
    prev = GENESIS_HASH
    out = []
    for r in recs:
        r.pop("hash")
        r["prev_hash"] = prev
        r["hash"] = chain_hash(prev, r)
        prev = r["hash"]
        out.append(canonical_json(r) + "\n")
    _rewrite(ledger, out)
    ledger.head_path.write_text(f"{len(recs)} {prev}\n")
    assert ledger.verify().ok  # internally consistent forgery
    assert not ledger.verify(expect_head=anchored).ok


def test_detects_tampered_returns_file(ledger):
    rec = ledger.records()[0]
    p = ledger.returns_dir / f"{rec['returns_hash']}.npz"
    with np.load(p) as stored:
        arr = stored["returns"]
    arr[0] += 1.0
    np.savez_compressed(p, returns=arr)
    kinds = {i.kind for i in ledger.verify(check_returns=True).issues}
    assert kinds == {"returns_mismatch"}


def test_shared_compressed_store_and_legacy_npy(tmp_path):
    arr = np.random.default_rng(4).normal(size=200)
    shared = tmp_path / "objects"
    first = Ledger(tmp_path / "a", returns_directory=shared)
    rec = first.append(returns=arr)
    second = Ledger(tmp_path / "b", returns_directory=shared)
    second.append(returns=arr)
    assert len(list(shared.glob("*.npz"))) == 1
    np.testing.assert_array_equal(Ledger(first.dir).load_returns(rec["returns_hash"]), arr)
    assert first.verify(check_returns=True).ok
    # Existing user stores remain readable.
    (shared / f"{rec['returns_hash']}.npz").unlink()
    np.save(shared / f"{rec['returns_hash']}.npy", arr)
    assert second.verify(check_returns=True).ok


def test_concurrent_appends_keep_chain_valid(tmp_path):
    lg = Ledger(tmp_path / "tl")

    def worker(k):
        for j in range(20):
            Ledger(tmp_path / "tl").append(strategy={"w": k, "j": j}, sharpe=0.1, n_obs=100)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    res = lg.verify()
    assert res.ok, res.issues[:3]
    assert res.n_records == 80


def test_empty_ledger(tmp_path):
    lg = Ledger(tmp_path / "none")
    assert lg.head() == (0, GENESIS_HASH)
    assert not lg.verify().ok  # missing file is reported
    lg.append(strategy={}, sharpe=None)
    assert lg.verify().ok


def test_nan_values_hash_stably(tmp_path):
    lg = Ledger(tmp_path / "tl")
    lg.append(strategy={"x": 1}, sharpe=float("nan"), extra={"inf": float("inf")})
    assert lg.verify().ok
