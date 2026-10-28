"""Command-line interface: ``trial-ledger {verify,report,head,log,exec}``."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .hooks import append_trials, code_hash_for_command, command_fields, parse_trial_attempts
from .ledger import Ledger
from .report import build_report, format_report


def _ledger(args: argparse.Namespace) -> Ledger:
    return Ledger(args.dir) if args.dir else Ledger()


def cmd_verify(args: argparse.Namespace) -> int:
    lg = _ledger(args)
    res = lg.verify(expect_head=args.expect_head, check_returns=args.check_returns)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": res.ok,
                    "n_records": res.n_records,
                    "head": res.head_hash,
                    "issues": [i.__dict__ for i in res.issues],
                },
                indent=2,
            )
        )
    else:
        status = "OK" if res.ok else "TAMPERED"
        print(f"{status}: {res.n_records} records, head {res.head_hash}")
        for i in res.issues:
            where = f"line {i.line}" if i.line else "ledger"
            print(f"  [{i.kind}] {where}: {i.message}")
    return 0 if res.ok else 1


def cmd_report(args: argparse.Namespace) -> int:
    lg = _ledger(args)
    recs = lg.records()
    if args.session:
        recs = [r for r in recs if r.get("session_id") == args.session]
    if args.agent:
        recs = [r for r in recs if r.get("agent") == args.agent]
    rep = build_report(lg, recs, claimed_n=args.claimed_n, n_partitions=args.partitions)
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2, default=float))
    else:
        print(format_report(rep, threshold=args.threshold))
    return 0


def cmd_head(args: argparse.Namespace) -> int:
    seq, h = _ledger(args).head()
    print(f"{seq} {h}")
    return 0


def _load_series(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False)
    return np.loadtxt(p, delimiter="," if p.suffix == ".csv" else None, ndmin=1)


def cmd_log(args: argparse.Namespace) -> int:
    lg = _ledger(args)
    strategy = json.loads(args.strategy) if args.strategy else {}
    returns = _load_series(args.returns) if args.returns else None
    rec = lg.append(
        strategy=strategy,
        returns=returns,
        sharpe=args.sharpe,
        n_obs=args.n_obs,
        periods_per_year=args.periods_per_year,
        source="cli",
        agent=args.agent,
    )
    print(f"{rec['seq']} {rec['hash']}")
    return 0


def cmd_exec(args: argparse.Namespace) -> int:
    """Run a command, echo its output, and log the trials it reports (or one unscored run)."""
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("usage: trial-ledger exec -- <command ...>", file=sys.stderr)
        return 2
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    lg = _ledger(args)
    command = " ".join(cmd)
    code_hash = code_hash_for_command(command)
    attempts = parse_trial_attempts(proc.stdout)
    append_trials(
        lg,
        attempts,
        source="exec",
        base_extra={**command_fields(command), "returncode": proc.returncode},
        agent=args.agent,
        code_hash=code_hash,
    )
    if not attempts:
        lg.append(
            strategy={},
            source="exec",
            agent=args.agent,
            code_hash=code_hash,
            extra={**command_fields(command), "returncode": proc.returncode, "unscored": True},
        )
    print(f"trial-ledger: logged {max(len(attempts), 1)} run(s)", file=sys.stderr)
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="trial-ledger", description=__doc__)
    ap.add_argument(
        "--dir",
        default=None,
        help="ledger directory (default: $TRIAL_LEDGER_DIR or ./.trial-ledger)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="check the hash chain for tampering")
    v.add_argument("--expect-head", default=None, help='externally anchored "<seq> <hash>"')
    v.add_argument("--check-returns", action="store_true", help="also re-hash stored returns")
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_verify)

    r = sub.add_parser("report", help="N trials, best Sharpe, PSR, DSR, PBO")
    r.add_argument("--claimed-n", type=int, default=None, help="self-reported number of trials")
    r.add_argument("--session", default=None)
    r.add_argument("--agent", default=None)
    r.add_argument("--partitions", type=int, default=16, help="CSCV partitions S (even)")
    r.add_argument("--threshold", type=float, default=0.95)
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_report)

    h = sub.add_parser("head", help='print "<seq> <hash>" of the last record (to anchor it)')
    h.set_defaults(func=cmd_head)

    lg = sub.add_parser("log", help="append one trial manually")
    lg.add_argument("--strategy", default=None, help="JSON object describing the trial")
    lg.add_argument("--returns", default=None, help=".npy/.csv/.txt file of per-period returns")
    lg.add_argument("--sharpe", type=float, default=None, help="per-period Sharpe ratio")
    lg.add_argument("--n-obs", type=int, default=None)
    lg.add_argument("--periods-per-year", type=float, default=252.0)
    lg.add_argument("--agent", default=None)
    lg.set_defaults(func=cmd_log)

    e = sub.add_parser("exec", help="run a backtest command and log it")
    e.add_argument("--agent", default=None)
    e.add_argument("command", nargs=argparse.REMAINDER)
    e.set_defaults(func=cmd_exec)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
