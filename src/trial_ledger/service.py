"""Backtest service shared by the MCP server and the experiment's agent loop.

Every successful evaluation is appended to the ledger *before* its result is returned, so the
caller cannot see a result that is not counted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backtest import ENGINE_CODE_HASH, FAMILIES, Market, expand_grid, normalise_spec, run_backtest
from .ledger import Ledger
from .report import build_report

MAX_SWEEP = 50


class BacktestService:
    def __init__(
        self,
        market: Market,
        ledger: Ledger,
        *,
        agent: str | None = None,
        session_id: str | None = None,
        source: str = "mcp",
        store_returns: bool = True,
        max_sweep: int = MAX_SWEEP,
    ):
        self.market = market
        self.ledger = ledger
        self.agent = agent
        self.session_id = session_id
        self.source = source
        self.store_returns = store_returns
        self.max_sweep = max_sweep
        self.n_calls = 0
        self.returns_by_seq: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ helpers
    def _period(self) -> dict[str, Any] | None:
        d = self.market.dates
        return {"start": d[1], "end": d[-1]} if d else None

    def _evaluate(self, spec: dict[str, Any], via: str) -> dict[str, Any]:
        res = run_backtest(spec, self.market)
        rec = self.ledger.append(
            strategy=res.spec,
            returns=res.returns,
            periods_per_year=self.market.periods_per_year,
            source=self.source,
            agent=self.agent,
            session_id=self.session_id,
            code_hash=ENGINE_CODE_HASH,
            dataset_hash=self.market.dataset_hash,
            period=self._period(),
            extra={"tool": via, "dataset": self.market.name},
            store_returns=self.store_returns,
        )
        self.returns_by_seq[rec["seq"]] = res.returns
        return {"trial": rec["seq"], "strategy": res.spec, **res.summary()}

    # ------------------------------------------------------------------ tools
    def run_backtest(self, strategy: dict[str, Any]) -> dict[str, Any]:
        self.n_calls += 1
        try:
            spec = normalise_spec(strategy)
        except ValueError as exc:
            return {"error": str(exc)}
        return self._evaluate(spec, "run_backtest")

    def run_sweep(
        self,
        family: str,
        grid: dict[str, list[Any]],
        long_only: bool = False,
        vol_target: bool = False,
        top_k: int = 5,
    ) -> dict[str, Any]:
        self.n_calls += 1
        if family not in FAMILIES:
            return {"error": f"unknown family {family!r}; choose one of {sorted(FAMILIES)}"}
        if not isinstance(grid, dict) or not grid:
            return {"error": "grid must be a non-empty object {param: [values]}"}
        specs = expand_grid(family, grid, long_only=long_only, vol_target=vol_target)
        if len(specs) > self.max_sweep:
            return {"error": f"grid has {len(specs)} combinations; the maximum is {self.max_sweep}"}
        valid, errors = [], []
        for s in specs:
            try:
                valid.append(normalise_spec(s))
            except ValueError as exc:
                errors.append(str(exc))
        results = [self._evaluate(s, "run_sweep") for s in valid]
        results.sort(key=lambda r: r["sharpe_annualized"], reverse=True)
        out: dict[str, Any] = {
            "n_evaluated": len(results),
            "top": results[: max(1, int(top_k))],
            "worst_sharpe_annualized": results[-1]["sharpe_annualized"] if results else None,
        }
        if errors:
            out["n_invalid"] = len(errors)
            out["invalid_example"] = errors[0]
        return out

    def list_trials(self, limit: int = 20) -> dict[str, Any]:
        recs = [r for r in self.ledger.records() if r.get("session_id") == self.session_id]
        return {
            "n_trials": len(recs),
            "last": [
                {"trial": r["seq"], "strategy": r["strategy"], "sharpe_annualized": r["sharpe_ann"]}
                for r in recs[-int(limit) :]
            ],
        }

    def report(self, claimed_n: int | None = None) -> dict[str, Any]:
        recs = [r for r in self.ledger.records() if r.get("session_id") == self.session_id]
        return build_report(self.ledger, recs, claimed_n=claimed_n).to_dict()

    def describe(self) -> dict[str, Any]:
        r = self.market.returns
        d = self.market.dates
        return {
            "dataset": self.market.name,
            "n_obs": int(r.size),
            "start": d[0] if d else None,
            "end": d[-1] if d else None,
            "periods_per_year": self.market.periods_per_year,
            "fee_bps_per_unit_turnover": self.market.fee_bps,
            "families": {
                f: {p: [lo, hi] for p, (_, lo, hi, _) in ps.items()} for f, ps in FAMILIES.items()
            },
            "modifiers": ["long_only", "vol_target"],
        }
