"""Experiment driver.

uv run python -m experiments.run agents --models openai/gpt-oss-20b --seeds 0-9
uv run python -m experiments.run baseline --seeds 0-9
uv run python -m experiments.run analyze          # tables from the committed ledgers
uv run python -m experiments.run figures
uv run python -m experiments.run reproduce        # replay every episode from the cache,
                                                  # then analyze + figures (no API calls)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext

from .markets import MARKET_KINDS

DEFAULT_MODELS = [
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3-ultra-550b-a55b",
]
BTC_REPS = 5


def _seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = map(int, part.split("-"))
            out.extend(range(a, b + 1))
        elif part:
            out.append(int(part))
    return out


def _jobs(models, markets, seeds, btc_reps):
    """Model by model (in the order given), interleaved by seed within a model."""
    for model in models:
        for s in sorted(set(seeds) | set(range(btc_reps))):
            for kind in markets:
                if (kind == "btc" and s < btc_reps) or (kind != "btc" and s in seeds):
                    yield model, kind, s


def cmd_agents(args) -> int:
    from .agent import RESULTS, run_episode
    from .llm import nim_slot, slug

    if not args.offline and args.workers != 1:
        raise ValueError("online batches must use one worker per shared NIM slot")

    jobs = list(_jobs(args.models, args.markets, _seeds(args.seeds), args.btc_reps))
    if args.resume:

        def complete(job):
            m, k, s = job
            p = RESULTS / "runs" / slug(m) / f"{k}-s{s}-r0" / "episode.json"
            return p.exists() and json.loads(p.read_text())["status"] == "ok"

        jobs = [j for j in jobs if not complete(j)]
    print(f"{len(jobs)} episodes, {args.workers} workers, offline={args.offline}", flush=True)
    t0 = time.time()
    failures = 0
    with nullcontext() if args.offline else nim_slot(), ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_episode, m, k, s, 0, args.offline): (m, k, s) for m, k, s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            m, k, s = futs[f]
            try:
                r = f.result()
                sub = r["submission"] or {}
                print(
                    f"[{i}/{len(jobs)} {time.time() - t0:6.0f}s] {m} {k} s{s}: "
                    f"{r['status']} calls={r['tool_calls']} ledger={r['ledger_records']} "
                    f"claimed={sub.get('n_strategies_tested')} "
                    f"sr={sub.get('annualized_sharpe')} api={r['api_calls']}",
                    flush=True,
                )
                if r["status"].startswith("api_error"):
                    failures += 1
            except Exception as exc:
                failures += 1
                print(
                    f"[{i}/{len(jobs)}] {m} {k} s{s}: FAILED {type(exc).__name__}: {exc}",
                    flush=True,
                )
    return 1 if failures else 0


def cmd_baseline(args) -> int:
    from .agent import run_random_search

    for kind in args.markets:
        for s in range(args.btc_reps) if kind == "btc" else _seeds(args.seeds):
            r = run_random_search(kind, s)
            print(kind, s, r["ledger_records"], r["submission"]["annualized_sharpe"])
    return 0


def cmd_analyze(args) -> int:
    from .analyze import collect, summarise, write_tables
    from .report_md import main as write_results_md

    rows = collect(args.models)
    summary = summarise(rows)
    write_tables(rows, summary)
    from .paired_tests import main as paired_tests

    paired_tests(rows)
    write_results_md()
    return 0


def cmd_figures(args) -> int:
    from .figures import make_all

    make_all()
    return 0


def _trial_fingerprint(run_dir) -> list[tuple[str, str]]:
    from trial_ledger.ledger import Ledger

    return [(r["strategy_hash"], r["returns_hash"]) for r in Ledger(run_dir / "ledger").records()]


def cmd_replay_check(args) -> int:
    """Replay every committed episode from the response cache (no API calls) into a temporary
    directory and check that the agent makes exactly the same trials and submission."""
    import tempfile
    from pathlib import Path

    from .agent import RESULTS, run_episode, run_random_search

    eps = [json.loads(p.read_text()) for p in sorted((RESULTS / "runs").glob("*/*/episode.json"))]
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for e in eps:
            if "-cli/" in e["model"]:
                # CLI inference is not reproducible offline. Audit its anchored ledger
                # and rerun each recorded evaluation; never impersonate an API replay.
                from trial_ledger.backtest import run_backtest
                from trial_ledger.hashing import hash_array
                from trial_ledger.ledger import Ledger

                from .markets import make_market

                path = RESULTS / "runs" / e["model"].replace("/", "__") / e["run_id"]
                ledger = Ledger(path / "ledger")
                market, _, _ = make_market(e["market"], e["seed"])
                same = ledger.verify(expect_head=e["ledger_head"], check_returns=True).ok
                same &= all(
                    hash_array(run_backtest(r["strategy"], market).returns) == r["returns_hash"]
                    for r in ledger.records()
                )
                bad += not same
                if not same:
                    print(f"CLI AUDIT MISMATCH {e['run_id']}", flush=True)
                continue
            if e["model"] == "random-search":
                new = run_random_search(e["market"], e["seed"], e["rep"], results_dir=tmp)
                sub = "random-search"
            else:
                new = run_episode(
                    e["model"], e["market"], e["seed"], e["rep"], offline=True, results_dir=tmp
                )
                sub = e["model"].replace("/", "__")
            rid = e["run_id"]
            same = (
                _trial_fingerprint(RESULTS / "runs" / sub / rid)
                == _trial_fingerprint(tmp / "runs" / sub / rid)
                and new["submission"] == e["submission"]
                and new["status"] == e["status"]
            )
            bad += not same
            if not same:
                print(f"MISMATCH {e['model']} {rid}", flush=True)
    cli = sum("-cli/" in e["model"] for e in eps)
    print(
        f"checked {len(eps)} episodes: {len(eps) - cli} cache/baseline replays, "
        f"{cli} CLI ledger audits; {bad} mismatches"
    )
    return 1 if bad else 0


def cmd_reproduce(args) -> int:
    """Rebuild every table and figure from the committed ledgers (re-running each logged
    trial and checking its returns hash), after replaying all episodes from the cache."""
    rc = cmd_replay_check(args)
    cmd_analyze(argparse.Namespace(models=None))
    cmd_figures(args)
    from .provenance import write_manifest

    write_manifest()
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="experiments.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("agents")
    a.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    a.add_argument("--markets", nargs="+", default=list(MARKET_KINDS), choices=MARKET_KINDS)
    a.add_argument("--seeds", default="0-9")
    a.add_argument("--btc-reps", type=int, default=BTC_REPS)
    a.add_argument("--workers", type=int, default=1)
    a.add_argument("--resume", action="store_true", help="skip completed episodes")
    a.add_argument("--offline", action="store_true", help="replay from cache only")
    a.set_defaults(func=cmd_agents)
    b = sub.add_parser("baseline")
    b.add_argument("--markets", nargs="+", default=list(MARKET_KINDS), choices=MARKET_KINDS)
    b.add_argument("--seeds", default="0-9")
    b.add_argument("--btc-reps", type=int, default=BTC_REPS)
    b.set_defaults(func=cmd_baseline)
    c = sub.add_parser("analyze")
    c.add_argument("--models", nargs="*", default=None)
    c.set_defaults(func=cmd_analyze)
    sub.add_parser("figures").set_defaults(func=cmd_figures)
    sub.add_parser("replay-check").set_defaults(func=cmd_replay_check)
    sub.add_parser("reproduce").set_defaults(func=cmd_reproduce)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
