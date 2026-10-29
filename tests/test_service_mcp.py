import asyncio
import json

import numpy as np
import pytest

from trial_ledger.backtest import Market
from trial_ledger.ledger import Ledger
from trial_ledger.service import BacktestService


@pytest.fixture
def service(tmp_path):
    rng = np.random.default_rng(0)
    r = rng.standard_t(4, 1200) * 0.02
    dates = [f"d{i:05d}" for i in range(r.size)]
    return BacktestService(Market("null", r, dates), Ledger(tmp_path / "tl"), session_id="s1")


def test_every_evaluation_is_logged(service):
    out = service.run_backtest({"family": "momentum", "lookback": 20})
    assert out["trial"] == 1 and "sharpe_annualized" in out
    sweep = service.run_sweep("ma_crossover", {"fast": [5, 10, 20], "slow": [50, 100]})
    assert sweep["n_evaluated"] == 6
    assert len(service.ledger.records()) == 7
    assert service.list_trials()["n_trials"] == 7
    rep = service.report(claimed_n=2)
    assert rep["n_trials"] == 7 and rep["dsr_claimed"] >= rep["dsr_ledger"]
    assert service.ledger.verify(check_returns=True).ok


def test_invalid_specs_are_not_trials(service):
    assert "error" in service.run_backtest({"family": "magic"})
    assert "error" in service.run_sweep("momentum", {"lookback": list(range(2, 100))})
    s = service.run_sweep("ma_crossover", {"fast": [5, 60], "slow": [50]})
    assert s["n_evaluated"] == 1 and s["n_invalid"] == 1
    assert len(service.ledger.records()) == 1


def test_sweep_results_sorted(service):
    s = service.run_sweep("momentum", {"lookback": [5, 10, 20, 40, 80]}, top_k=5)
    srs = [t["sharpe_annualized"] for t in s["top"]]
    assert srs == sorted(srs, reverse=True)


def test_research_budget_counts_calls_not_grid_points(service):
    service.tool_budget = 2
    assert service.run_sweep("momentum", {"lookback": [5, 10, 20]})["n_evaluated"] == 3
    assert "error" not in service.run_backtest({"family": "momentum"})
    assert "error" in service.run_backtest({"family": "momentum"})
    assert "error" in service.run_sweep("momentum", {"lookback": [30]})
    assert len(service.ledger.records()) == 4


def test_research_catalog_hides_accounting(service):
    pytest.importorskip("mcp")
    from mcp import Client

    from trial_ledger.mcp_server import create_server

    async def go():
        async with Client(create_server(service, research_only=True)) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert names == {"describe_dataset", "run_backtest", "run_sweep"}

    asyncio.run(go())


def test_mcp_server_roundtrip(service):
    pytest.importorskip("mcp")
    from trial_ledger.mcp_server import create_server

    try:
        from mcp import Client
    except ImportError:  # mcp 1.x has no high-level in-process client
        pytest.skip("needs mcp>=2 for the in-process client")

    server = create_server(service)

    async def go():
        async with Client(server) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert {"run_backtest", "run_sweep", "list_trials", "ledger_report"} <= tools
            res = await client.call_tool(
                "run_backtest", {"strategy": {"family": "rsi", "period": 14}}
            )
            payload = json.loads(res.content[0].text)
            assert payload["trial"] == 1
            await client.call_tool(
                "run_sweep", {"family": "breakout", "grid": {"lookback": [10, 20, 30]}}
            )
            res = await client.call_tool("list_trials", {})
            assert json.loads(res.content[0].text)["n_trials"] == 4

    asyncio.run(go())
    assert service.ledger.verify().ok


def test_mcp_server_over_stdio(tmp_path):
    """Launch trial-ledger-mcp as a subprocess, like Claude Code or Codex would."""
    pytest.importorskip("mcp")
    try:
        from mcp import Client, StdioServerParameters
    except ImportError:
        pytest.skip("needs mcp>=2")
    import sys

    rng = np.random.default_rng(1)
    csv = tmp_path / "prices.csv"
    closes = 100 * np.cumprod(1 + rng.standard_t(4, 400) * 0.02)
    csv.write_text("date,close\n" + "".join(f"d{i:04d},{c}\n" for i, c in enumerate(closes)))
    ledger_dir = tmp_path / "tl"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "trial_ledger.mcp_server", "--data", str(csv), "--ledger", str(ledger_dir),
              "--hide-trial-ids", "--holdout-from", "d0300"],
    )  # fmt: skip

    async def go():
        async with Client(params) as client:
            desc = json.loads((await client.call_tool("describe_dataset", {})).content[0].text)
            assert desc["n_obs"] == 300  # rows from the holdout date on are hidden
            res = await client.call_tool(
                "run_sweep", {"family": "momentum", "grid": {"lookback": [5, 10, 20]}}
            )
            payload = json.loads(res.content[0].text)
            assert payload["n_evaluated"] == 3
            assert all("trial" not in t for t in payload["top"])

    asyncio.run(go())
    lg = Ledger(ledger_dir)
    assert len(lg.records()) == 3 and lg.verify(check_returns=True).ok
