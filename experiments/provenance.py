"""Small, credential-free provenance receipts for generated research artifacts."""

from __future__ import annotations

import hashlib
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    sources = sorted((ROOT / "experiments").glob("*.py")) + sorted(
        (ROOT / "src/trial_ledger").glob("*.py")
    )
    return {
        "command": shlex.join(sys.orig_argv),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "python": platform.python_version(),
        "config": config,
        "hosted_model_revision": "unknown; requested alias and available CLI version recorded",
    }


def write_manifest() -> None:
    import json

    from .agent import RESULTS

    paths = sorted((RESULTS / "runs").glob("*/*/episode.json"))
    episodes = [json.loads(p.read_text()) for p in paths]
    artifacts = [p for p in RESULTS.iterdir() if p.is_file() and p.name != "provenance.json"]
    artifacts += sorted((RESULTS / "figures").glob("*.png"))
    receipt = provenance(
        {
            "evaluation_version": "causal-v2",
            "protocol_commit": "5dcf789ab54c0c6b0605f334f6edaa3612131dfe",
            "volatility_target": "fixed 0.02 per period; lagged 20-period denominator; cap 2",
            "btc_diagnostic_note": "previously exposed period; not an untouched test",
            "models": sorted({e["model"] for e in episodes}),
            "seeds_by_model": {
                m: sorted({e["seed"] for e in episodes if e["model"] == m})
                for m in sorted({e["model"] for e in episodes})
            },
            "nim_temperature": 0.7,
            "tool_budget": 30,
            "sharpe_target": 1.0,
            "fee_bps": 5.0,
            "is_period": ["2018-01-01", "2023-12-31"],
            "oos_period": ["2024-01-01", "2026-09-30"],
            "pbo_partitions": 16,
        }
    )
    receipt["artifact_sha256"] = {
        str(p.relative_to(RESULTS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts
    }
    receipt["episode_receipt_sha256"] = {
        str(p.relative_to(RESULTS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    receipt["legacy_provenance_note"] = (
        "Superseded first-version NIM episodes did not record their launch command "
        "or deployment revision. "
        "Their original caches, ledgers and seeds remain at the superseded result path; "
        "this manifest records the exact current metric-regeneration command and source."
    )
    (RESULTS / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
