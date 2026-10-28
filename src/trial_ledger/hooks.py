"""Agent hooks: log backtests launched through a shell tool, outside the MCP server.

``trial-ledger-hook`` reads a hook event as JSON on stdin and appends to the ledger.
It understands Claude Code ``PostToolUse`` events for the ``Bash`` tool and the similar
payloads of other agents (Codex-style ``shell``/``exec_command`` tools whose command is an
argv list). It always exits 0 so it can never block the agent.

Two levels of capture:

1. **Scored**: any line printed by the script of the form ``TRIAL_LEDGER {json}`` (see
   :func:`emit`) becomes one trial with its Sharpe ratio. A grid search printing 200 such
   lines is logged as 200 trials.
2. **Unscored**: otherwise, if the command looks like a backtest (regex, configurable with
   ``TRIAL_LEDGER_HOOK_PATTERN``), one run is logged without a Sharpe ratio. It still counts
   towards N in the report.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import hash_file, hash_json, sha256_hex
from .ledger import Ledger

TRIAL_PREFIX = "TRIAL_LEDGER "
DEFAULT_PATTERN = (
    r"(backtest|back_test|strateg|sweep|grid.?search|optimi[sz]e|walk.?forward|"
    r"vectorbt|backtrader|zipline|nautilus|bt\.run|alpha|signal)"
)


def emit(sharpe: float, n_obs: int, strategy: dict[str, Any] | None = None, **extra: Any) -> str:
    """Print a machine-readable trial line from inside a backtest script.

    ``sharpe`` is per period. Returns the printed line.
    """
    payload = {"sharpe": sharpe, "n_obs": n_obs, "strategy": strategy or {}, **extra}
    line = TRIAL_PREFIX + json.dumps(payload, default=float)
    print(line, flush=True)
    return line


@dataclass
class TrialAttempt:
    """One ``TRIAL_LEDGER`` line: its JSON payload, or the reason it could not be parsed."""

    payload: dict[str, Any] | None
    error: str | None = None


def parse_trial_attempts(text: str) -> list[TrialAttempt]:
    """Every line starting with ``TRIAL_LEDGER `` is an attempt, parseable or not."""
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line.startswith(TRIAL_PREFIX):
            continue
        try:
            obj = json.loads(line[len(TRIAL_PREFIX) :])
        except json.JSONDecodeError:
            out.append(TrialAttempt(None, "JSONDecodeError"))
            continue
        if isinstance(obj, dict):
            out.append(TrialAttempt(obj))
        else:
            out.append(TrialAttempt(None, "TypeError"))
    return out


def parse_trial_lines(text: str) -> list[dict[str, Any]]:
    """Well-formed JSON payloads only (use :func:`parse_trial_attempts` for auditing)."""
    return [a.payload for a in parse_trial_attempts(text) if a.payload is not None]


def coerce_trial(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate one trial payload. Raises ``ValueError``/``TypeError`` if it is malformed."""
    t = dict(payload)
    sharpe = t.pop("sharpe", None)
    if sharpe is not None:
        if isinstance(sharpe, bool):
            raise TypeError("sharpe must be a number")
        sharpe = float(sharpe)
        if not math.isfinite(sharpe):
            raise ValueError("sharpe must be finite")
    n_obs = t.pop("n_obs", None)
    if n_obs is not None:
        if isinstance(n_obs, bool):
            raise TypeError("n_obs must be an integer")
        f = float(n_obs)
        if not f.is_integer() or f < 1:
            raise ValueError("n_obs must be a positive integer")
        n_obs = int(f)
    strategy = t.pop("strategy", None)
    if strategy is None:
        strategy = {}
    if not isinstance(strategy, dict):
        raise TypeError("strategy must be an object")
    ppy = float(t.pop("periods_per_year", 252.0))
    if not math.isfinite(ppy) or ppy <= 0:
        raise ValueError("periods_per_year must be positive")
    dataset_hash = t.pop("dataset_hash", None)
    if dataset_hash is not None and not isinstance(dataset_hash, str):
        raise TypeError("dataset_hash must be a string")
    return {
        "sharpe": sharpe,
        "n_obs": n_obs,
        "strategy": strategy,
        "periods_per_year": ppy,
        "dataset_hash": dataset_hash,
        "extra": t,
    }


def append_trials(
    ledger: Ledger,
    attempts: list[TrialAttempt],
    *,
    source: str,
    base_extra: dict[str, Any],
    agent: str | None = None,
    session_id: str | None = None,
    code_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Append one record per attempt. Fails closed: an attempt that cannot be validated is
    still logged, as an unscored record with ``extra.malformed = true``, so that one bad line
    never hides the trials around it (an audit log must not undercount)."""
    written = []
    for i, a in enumerate(attempts):
        try:
            if a.payload is None:
                raise ValueError(a.error or "unparseable")
            c = coerce_trial(a.payload)
            rec = ledger.append(
                strategy=c["strategy"],
                sharpe=c["sharpe"],
                n_obs=c["n_obs"],
                periods_per_year=c["periods_per_year"],
                source=source,
                agent=agent,
                session_id=session_id,
                code_hash=code_hash,
                dataset_hash=c["dataset_hash"],
                extra={**c["extra"], **base_extra},
            )
        except Exception as exc:
            error = a.error if a.payload is None else type(exc).__name__
            rec = ledger.append(
                strategy={},
                source=source,
                agent=agent,
                session_id=session_id,
                code_hash=code_hash,
                extra={
                    **base_extra,
                    "unscored": True,
                    "malformed": True,
                    "error": error,
                    "line_index": i,
                },
            )
        written.append(rec)
    return written


def looks_like_backtest(command: str, pattern: str | None = None) -> bool:
    pat = pattern or os.environ.get("TRIAL_LEDGER_HOOK_PATTERN") or DEFAULT_PATTERN
    runs_code = re.search(r"\b(python3?|uv run|poetry run|pytest|Rscript|julia|node)\b", command)
    return bool(runs_code and re.search(pat, command, flags=re.IGNORECASE))


def _command_from_event(event: dict[str, Any]) -> str | None:
    tool_input = event.get("tool_input") or event.get("input") or {}
    cmd = tool_input.get("command") if isinstance(tool_input, dict) else None
    if cmd is None:
        cmd = event.get("command")
    if isinstance(cmd, list):
        # Codex-style argv, often ["bash", "-lc", "<script>"]
        if len(cmd) >= 3 and cmd[1] in {"-c", "-lc"}:
            return str(cmd[2])
        return shlex.join(str(c) for c in cmd)
    return cmd if isinstance(cmd, str) else None


def _output_from_event(event: dict[str, Any]) -> str:
    resp = event.get("tool_response") or event.get("output") or event.get("result") or {}
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        parts = [resp.get(k) for k in ("stdout", "output", "stderr", "aggregated_output")]
        return "\n".join(p for p in parts if isinstance(p, str))
    return ""


def code_hash_for_command(command: str, cwd: str | None = None) -> str:
    """Hash of the command plus the content of any existing files it references."""
    parts = [command]
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for tok in tokens:
        p = Path(cwd or ".") / tok
        if p.suffix in {".py", ".R", ".jl", ".js", ".ts", ".ipynb", ".toml", ".yaml", ".yml"}:
            if p.is_file():
                parts.append(f"{tok}:{hash_file(p)}")
    return sha256_hex("\n".join(parts))


def log_shell_event(
    event: dict[str, Any], ledger: Ledger | None = None, agent: str | None = None
) -> list[dict[str, Any]]:
    """Append the trials found in one shell tool event; returns the records written."""
    tool = str(event.get("tool_name") or event.get("tool") or "")
    if tool and tool not in {"Bash", "shell", "exec_command", "local_shell", "container.exec"}:
        return []
    command = _command_from_event(event)
    if not command:
        return []
    output = _output_from_event(event)
    cwd = event.get("cwd")
    if ledger is None:
        use_cwd = cwd and not os.environ.get("TRIAL_LEDGER_DIR")
        ledger = Ledger(Path(cwd) / ".trial-ledger" if use_cwd else None)
    lg = ledger
    session = event.get("session_id") or event.get("thread_id") or event.get("turn_id")
    code_hash = code_hash_for_command(command, cwd)
    attempts = parse_trial_attempts(output)
    written = append_trials(
        lg,
        attempts,
        source="shell-hook",
        base_extra={"command": command[:2000]},
        agent=agent,
        session_id=session,
        code_hash=code_hash,
    )
    if not attempts and looks_like_backtest(command):
        written.append(
            lg.append(
                strategy={"command_hash": hash_json(command)},
                source="shell-hook",
                agent=agent,
                session_id=session,
                code_hash=code_hash,
                extra={"command": command[:2000], "unscored": True},
            )
        )
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Log backtests from agent shell-tool hook events.")
    ap.add_argument("--agent", default=None, help="agent name stored in each record")
    ap.add_argument("--dir", default=None, help="ledger directory (default: <cwd>/.trial-ledger)")
    args = ap.parse_args(argv)
    try:
        event = json.load(sys.stdin)
        ledger = Ledger(args.dir) if args.dir else None
        written = log_shell_event(event, ledger, agent=args.agent)
        if written:
            print(f"trial-ledger: logged {len(written)} trial(s)", file=sys.stderr)
    except Exception as exc:  # never block the agent
        print(f"trial-ledger hook error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
