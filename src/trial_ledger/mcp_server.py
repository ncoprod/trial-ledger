"""MCP server exposing a ledgered backtester to coding agents.

Run ``trial-ledger-mcp --data prices.csv`` (stdio transport). Every ``run_backtest`` /
``run_sweep`` evaluation is appended to the ledger before its result is returned.

Configuration (flags or environment variables):

* ``--data`` / ``TRIAL_LEDGER_DATA``: CSV with ``date`` and ``close`` (or ``return``) columns.
* ``--ledger`` / ``TRIAL_LEDGER_DIR``: ledger directory (default ``./.trial-ledger``).
* ``--periods-per-year`` (default 365, crypto daily), ``--fee-bps`` (default 5).
* ``--holdout-from DATE``: rows from DATE on are hidden from the agent (out-of-sample).
"""

from __future__ import annotations

import argparse
import os
import uuid
from typing import Any

from .backtest import FAMILY_DOCS, Market
from .ledger import Ledger
from .service import MAX_SWEEP, BacktestService


def build_service(args: argparse.Namespace) -> BacktestService:
    if not args.data:
        raise SystemExit("trial-ledger-mcp: --data or TRIAL_LEDGER_DATA is required")
    market = Market.from_csv(
        args.data, periods_per_year=args.periods_per_year, fee_bps=args.fee_bps
    )
    if args.holdout_from and market.dates:
        cut = next(
            (i for i, d in enumerate(market.dates) if d >= args.holdout_from), len(market.dates)
        )
        market = market.slice(0, cut)
    return BacktestService(
        market,
        Ledger(args.ledger),
        agent=args.agent,
        session_id=args.session or uuid.uuid4().hex[:12],
        source="mcp",
        expose_trial_ids=not args.hide_trial_ids,
        tool_budget=args.tool_budget,
    )


def _server_class():
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP

        return FastMCP


def create_server(service: BacktestService, research_only: bool = False):
    """Build the MCP server (``MCPServer`` on mcp 2.x, ``FastMCP`` on mcp 1.x)."""
    families = "\n".join(f"- {k}: {v}" for k, v in FAMILY_DOCS.items())
    mcp = _server_class()(
        "trial-ledger",
        instructions=(
            "Backtest single-asset trading rules. Every evaluation is recorded in a "
            "tamper-evident ledger and counts as a trial for multiple-testing corrections."
        ),
    )

    @mcp.tool()
    def describe_dataset() -> dict[str, Any]:
        """Dataset, sample period, costs and the allowed strategy families with parameter ranges."""
        return service.describe()

    @mcp.tool(
        description=(
            "Backtest one strategy on the configured dataset and return annualised Sharpe, "
            "return, volatility, max drawdown, turnover. `strategy` is an object with "
            "`family` and its parameters, plus optional `long_only` and `vol_target` "
            f"booleans. Families:\n{families}"
        )
    )
    def run_backtest(strategy: dict[str, Any]) -> dict[str, Any]:
        return service.run_backtest(strategy)

    @mcp.tool(
        description=(
            "Backtest every combination of a parameter grid for one family (at most "
            f"{MAX_SWEEP} combinations) and return the best `top_k`. Each combination is "
            "logged as a separate trial. Example: family='ma_crossover', "
            "grid={'fast': [5, 10], 'slow': [50, 100]}."
        )
    )
    def run_sweep(
        family: str,
        grid: dict[str, list[float]],
        long_only: bool = False,
        vol_target: bool = False,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return service.run_sweep(family, grid, long_only, vol_target, top_k)

    def list_trials(limit: int = 20) -> dict[str, Any]:
        """Number of trials logged in this session and the most recent ones."""
        return service.list_trials(limit)

    def ledger_report(claimed_n: int | None = None) -> dict[str, Any]:
        """Multiple-testing report for this session: best Sharpe, PSR, Deflated Sharpe Ratio
        with the ledger's trial count (and with `claimed_n` if given), and PBO."""
        return service.report(claimed_n)

    if not research_only:
        mcp.tool()(list_trials)
        mcp.tool()(ledger_report)
    return mcp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="trial-ledger-mcp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", default=os.environ.get("TRIAL_LEDGER_DATA"))
    ap.add_argument("--ledger", default=os.environ.get("TRIAL_LEDGER_DIR", ".trial-ledger"))
    ap.add_argument("--periods-per-year", type=float, default=365.0)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--holdout-from", default=os.environ.get("TRIAL_LEDGER_HOLDOUT_FROM"))
    ap.add_argument("--agent", default=os.environ.get("TRIAL_LEDGER_AGENT"))
    ap.add_argument("--session", default=os.environ.get("TRIAL_LEDGER_SESSION"))
    ap.add_argument("--tool-budget", type=int, default=None)
    ap.add_argument("--research-only", action="store_true", help="hide accounting/report tools")
    ap.add_argument(
        "--hide-trial-ids",
        action="store_true",
        help="do not return the ledger's trial id with results (used in the experiment so the "
        "agent's self-reported N is its own count)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    service = build_service(args)
    create_server(service, research_only=args.research_only).run()


if __name__ == "__main__":
    main()
