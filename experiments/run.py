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
    """Interleaved by seed so that every model/market gets results early."""
    for s in sorted(set(seeds) | set(range(btc_reps))):
        for model in models:
            for kind in markets:
                if (kind == "btc" and s < btc_reps) or (kind != "btc" and s in seeds):
                    yield model, kind, s


def cmd_agents(args) -> int:
    from .agent import run_episode

    jobs = list(_jobs(args.models, args.markets, _seeds(args.seeds), args.btc_reps))
    print(f"{len(jobs)} episodes, {args.workers} workers, offline={args.offline}", flush=True)
    t0 = time.time()
    failures = 0
    with ThreadPoolExecutor(args.workers) as ex:
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

    rows = collect(args.models)
    summary = summarise(rows)
    write_tables(rows, summary)
    print(json.dumps(summary, indent=1))
    return 0


def cmd_figures(args) -> int:
    from .figures import make_all

    make_all()
    return 0


def cmd_reproduce(args) -> int:
    """Replay all cached episodes (no API calls), then rebuild tables and figures."""
    from pathlib import Path

    from .agent import RESULTS

    episodes = []
    for ep in sorted((RESULTS / "runs").glob("*/*/episode.json")):
        episodes.append(json.loads(ep.read_text()))
    from .agent import run_episode, run_random_search

    for e in episodes:
        if e["model"] == "random-search":
            run_random_search(e["market"], e["seed"], e["rep"])
        else:
            run_episode(e["model"], e["market"], e["seed"], e["rep"], offline=True)
    print(f"replayed {len(episodes)} episodes from {Path('experiments/cache')}")
    cmd_analyze(argparse.Namespace(models=None))
    cmd_figures(args)
    return 0


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
    a.add_argument("--workers", type=int, default=2)
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
    sub.add_parser("reproduce").set_defaults(func=cmd_reproduce)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
