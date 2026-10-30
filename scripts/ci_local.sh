#!/bin/bash
set -eu
cd -- "$(dirname -- "$0")/.."

# Mirrors .github/workflows/ci.yml on the current Mac.
# Skip GitHub checkout/setup/cache/upload actions and the Linux/x86_64 matrix;
# build and check native macOS wheels and the source archive locally instead.
# The manually dispatched benchmark.yml is a performance study, not push CI.
# Use --no-sync after each explicit sync so uv keeps that job's selected extras.
export CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python3 - <<'PY'
import datetime
import json
import platform
import subprocess
import time
from pathlib import Path

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
short_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"]))
steps = [
    ("sync Python", "uv sync --extra mcp --group dev --group experiments"),
    ("lint", "uv run --no-sync ruff check ."),
    ("format", "uv run --no-sync ruff format --check ."),
    ("Python tests", "uv run --no-sync pytest"),
    ("Rust tests", "cargo test --release --manifest-path rust/Cargo.toml"),
    ("sync Rust extension", "uv sync --all-extras --group dev --group experiments"),
    ("import Rust extension", "uv run --no-sync python -c 'import trial_ledger_core; print(trial_ledger_core.__version__)'"),
    ("full tests", "uv run --no-sync pytest"),
    ("replay", "uv run --no-sync python -m experiments.run replay-check"),
    ("ledger PBO", "uv run --no-sync trial-ledger --dir experiments/results/causal-v2/runs/random-search/signflip-s0-r0/ledger pbo"),
    ("native wheel", "uv run --no-sync --with 'maturin>=1.7,<2' maturin build --release --manifest-path rust/Cargo.toml --out rust/dist"),
    ("source archive", "uv run --no-sync --with 'maturin>=1.7,<2' maturin sdist --manifest-path rust/Cargo.toml --out rust/sdist"),
    ("wheel notices", "uv run --no-sync python rust/check_licenses.py rust/dist/*.whl"),
    ("install wheel", "uv pip install rust/dist/*.whl"),
    ("wheel parity", "uv run --no-sync pytest tests/test_rust_parity.py"),
    ("source notices", "uv run --no-sync python rust/check_licenses.py rust/sdist/*.tar.gz"),
]
receipt = {
    "commit": commit,
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
    "working_tree_dirty": dirty,
    "steps": [],
    "overall": "PASS",
}
for name, command in steps:
    print(f"ci_local: {name}", flush=True)
    started = time.monotonic()
    result = subprocess.run(["/bin/bash", "-c", command])
    receipt["steps"].append({
        "name": name,
        "exit_code": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
    })
    if result.returncode:
        receipt["overall"] = "FAIL"
        break

path = Path("results/ci_local") / f"{short_sha}.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(receipt, indent=2) + "\n")
print(f"ci_local: {receipt['overall']} ({path})", flush=True)
raise SystemExit(0 if receipt["overall"] == "PASS" else 1)
PY
