"""A few headless Claude Code runs against the real MCP server (``claude -p``).

Shows the capture path end to end with a production coding agent: Claude Code talks to
``trial-ledger-mcp`` over stdio, the server writes the ledger, and the analysis treats the
episode exactly like the harness episodes. Only the MCP backtest tools are allowed (no shell,
no file access), ``list_trials``/``ledger_report`` are not exposed and trial ids are hidden.

    uv run python -m experiments.claude_code --seeds 100-102 --market signflip
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .agent import RESULTS, TOOL_BUDGET, USER_TASK
from .markets import make_market

REPO = Path(__file__).resolve().parent.parent
MODEL = "haiku"
ALLOWED = [
    "mcp__trial-ledger__describe_dataset",
    "mcp__trial-ledger__run_backtest",
    "mcp__trial-ledger__run_sweep",
]
PROMPT = (
    "You are a quantitative researcher. Use the trial-ledger MCP tools (describe_dataset, "
    "run_backtest, run_sweep) to research strategies on the configured daily crypto dataset. "
    f"{USER_TASK.replace('call submit_final_answer with', 'reply with')} "
    f"Use at most {TOOL_BUDGET} backtest tool calls. End your reply with one line of JSON: "
    '{"strategy": {...}, "annualized_sharpe": <number>, "n_strategies_tested": <integer>}'
)


def _write_csv(market, path: Path) -> None:
    with open(path, "w") as fh:
        fh.write("date,return\n")
        for d, r in zip(market.dates, market.returns, strict=True):
            fh.write(f"{d},{r!r}\n")


def _last_json(text: str) -> dict | None:
    for m in reversed(list(re.finditer(r"\{.*\}", text or ""))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "strategy" in obj:
            return obj
    return None


def run(kind: str, seed: int, model: str = MODEL, budget_usd: float = 0.5) -> dict:
    market, _, _ = make_market(kind, seed)
    run_id = f"{kind}-s{seed}-r0"
    out = RESULTS / "runs" / f"claude-code__{model}" / run_id
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    _write_csv(market, out / "data.csv")
    server = {
        "command": "uv",
        "args": ["run", "--directory", str(REPO), "trial-ledger-mcp", "--data",
                 str(out / "data.csv"), "--ledger", str(out / "ledger"), "--agent",
                 f"claude-code/{model}", "--session", run_id, "--hide-trial-ids"],
    }  # fmt: skip
    cfg = out / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"trial-ledger": server}}, indent=1))
    cmd = [
        "claude", "-p", PROMPT, "--model", model, "--output-format", "json",
        "--mcp-config", str(cfg), "--strict-mcp-config", "--tools", "",
        "--allowedTools", *ALLOWED, "--max-budget-usd", str(budget_usd),
        "--no-session-persistence", "--setting-sources", "",
    ]  # fmt: skip
    proc = subprocess.run(cmd, cwd=out, capture_output=True, text=True, timeout=1800)
    try:
        res = json.loads(proc.stdout)
    except json.JSONDecodeError:
        res = {"result": proc.stdout[-4000:], "is_error": True, "stderr": proc.stderr[-2000:]}
    sub = _last_json(res.get("result", ""))
    from trial_ledger.ledger import Ledger

    ledger = Ledger(out / "ledger")
    record = {
        "model": f"claude-code/{model}",
        "market": kind,
        "seed": seed,
        "rep": 0,
        "run_id": run_id,
        "status": "ok" if sub else ("error" if res.get("is_error") else "no_submission"),
        "turns": res.get("num_turns"),
        "tool_calls": None,  # not observable from the ledger; see num_turns
        "submission": sub,
        "ledger_records": len(ledger.records()),
        "ledger_head": " ".join(map(str, ledger.head())),
        "api_calls": res.get("num_turns"),
        "cost_usd": res.get("total_cost_usd"),
        "final_text": (res.get("result") or "")[-3000:],
    }
    (out / "episode.json").write_text(json.dumps(record, indent=1))
    (cfg).unlink()  # contains local absolute paths only; not needed to analyze the run
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="signflip")
    ap.add_argument("--seeds", default="100-102")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args(argv)
    a, _, b = args.seeds.partition("-")
    for s in range(int(a), int(b or a) + 1):
        r = run(args.market, s, args.model)
        print(
            json.dumps(
                {k: r[k] for k in ("run_id", "status", "ledger_records", "submission", "cost_usd")}
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
