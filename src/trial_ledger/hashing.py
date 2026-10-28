"""Canonical serialisation and content hashing helpers.

Every hash in the ledger is a hex SHA-256 digest. Records are hashed over a
canonical JSON encoding (sorted keys, no whitespace, UTF-8, finite floats only)
so that the same record always produces the same digest on any machine.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

GENESIS_HASH = "0" * 64


def _normalise(obj: Any) -> Any:
    """Convert numpy scalars/arrays and tuples into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_normalise(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            # JSON has no NaN/inf; encode them explicitly so hashing is stable.
            return {"__float__": repr(x)}
        return x
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding used for hashing."""
    return json.dumps(
        _normalise(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_json(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def hash_array(values: Any) -> str:
    """Hash a numeric series as little-endian float64 bytes (shape-independent of dtype)."""
    arr = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    h = hashlib.sha256()
    h.update(f"f8:{arr.shape}".encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def hash_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def chain_hash(prev_hash: str, body: dict[str, Any]) -> str:
    """SHA-256(prev_hash || canonical_json(body))."""
    return sha256_hex(prev_hash + canonical_json(body))
