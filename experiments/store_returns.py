"""Restore missing return objects without changing any ledger record or anchor."""

from __future__ import annotations

import json

from trial_ledger.backtest import run_backtest
from trial_ledger.hashing import hash_array
from trial_ledger.ledger import Ledger

from .agent import RESULTS
from .markets import make_market


def main() -> None:
    counts = {"ledgers": 0, "trials": 0, "objects_restored": 0}
    for path in sorted((RESULTS / "runs").glob("*/*/episode.json")):
        ep = json.loads(path.read_text())
        ledger = Ledger(path.parent / "ledger", returns_directory=RESULTS / "returns")
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
                result = run_backtest(rec["strategy"], market)
                if hash_array(result.returns) != h:
                    raise ValueError(f"returns differ: {path}, trial {rec['seq']}")
                assert ledger.store_returns(result.returns) == h
                counts["objects_restored"] += 1
            counts["trials"] += 1
        ledger._ensure()
        assert ledger.verify(expect_head=ep["ledger_head"], check_returns=True).ok
        counts["ledgers"] += 1
        print(path.parent.name, counts, flush=True)
    counts["unique_compressed_objects"] = len(list((RESULTS / "returns").glob("*.npz")))
    counts["compressed_bytes"] = sum(p.stat().st_size for p in (RESULTS / "returns").glob("*.npz"))
    (RESULTS / "returns_integrity.json").write_text(json.dumps(counts, indent=2) + "\n")


if __name__ == "__main__":
    main()
