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
        "command": shlex.join(["python", *sys.argv]),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "python": platform.python_version(),
        "config": config,
        "hosted_model_revision": "not exposed by the provider; model alias and CLI version recorded",
    }
