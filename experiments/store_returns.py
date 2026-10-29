"""Restore missing return objects without changing any ledger record or anchor."""

from __future__ import annotations

import json

from trial_ledger.backtest import run_backtest
from trial_ledger.hashing import hash_array
from trial_ledger.ledger import Ledger

from .agent import RESULTS
from .markets import make_market


def main() -> None:
    from .provenance import provenance

    counts = {"ledgers": 0, "trials": 0, "objects_restored": 0, "local_duplicates_removed": 0}
    for path in sorted((RESULTS / "runs").glob("*/*/episode.json")):
        ep = json.loads(path.read_text())
        ledger = Ledger(path.parent / "ledger", returns_directory=RESULTS / "returns")
        local = Ledger(path.parent / "ledger", returns_directory=path.parent / "ledger/returns")
        if not ledger.verify(expect_head=ep["ledger_head"]).ok:
            raise ValueError(f"ledger verification failed: {path}")
        market, _, _ = make_market(ep["market"], ep["seed"])
        seen = set()
        for rec in ledger.records():
            h = rec["returns_hash"]
            if h is None or h in seen:
                continue
            seen.add(h)
            if ledger.load_returns(h) is None:
                returns = local.load_returns(h)
                if returns is None:
                    returns = run_backtest(rec["strategy"], market).returns
                if hash_array(returns) != h:
                    raise ValueError(f"returns differ: {path}, trial {rec['seq']}")
                assert ledger.store_returns(returns) == h
                counts["objects_restored"] += 1
            counts["trials"] += 1
        ledger._ensure()
        assert ledger.verify(expect_head=ep["ledger_head"], check_returns=True).ok
        for duplicate in local.returns_dir.glob("*"):
            if duplicate.suffix in {".npy", ".npz"} and duplicate.stem in seen:
                assert ledger.load_returns(duplicate.stem) is not None
                duplicate.unlink()
                counts["local_duplicates_removed"] += 1
        counts["ledgers"] += 1
        print(path.parent.name, counts, flush=True)
    counts["unique_compressed_objects"] = len(list((RESULTS / "returns").glob("*.npz")))
    counts["compressed_bytes"] = sum(p.stat().st_size for p in (RESULTS / "returns").glob("*.npz"))
    counts["provenance"] = provenance(
        {
            "hash_check": "every distinct stored trial",
            "storage": "np.savez_compressed float64, SHA-256",
        }
    )
    (RESULTS / "returns_integrity.json").write_text(json.dumps(counts, indent=2) + "\n")


if __name__ == "__main__":
    main()
