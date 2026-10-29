"""Append-only, hash-chained JSONL ledger of backtest trials.

Layout of a ledger directory (default ``.trial-ledger/``)::

    ledger.jsonl      one JSON record per line, append-only
    HEAD              "<seq> <hash>" of the last record, rewritten after each append
    returns/<h>.npz  compressed content-addressed store (legacy .npy also readable)
    RETURNS_STORE    optional relative path to a shared returns store
    .lock             advisory lock serialising concurrent writers

Each record carries ``prev_hash`` and ``hash = SHA-256(prev_hash || canonical_json(body))``
where ``body`` is the record without its ``hash`` field. Editing, deleting, inserting or
reordering any line breaks the chain from that point on. Truncating the *tail* of the file
cannot be detected from the chain alone, which is why the head is also written to ``HEAD``
and can be anchored externally (e.g. committed to git or pasted in a report) and checked with
``trial-ledger verify --expect-head``. This makes the ledger tamper-evident, not
tamper-proof: someone with write access can rebuild a consistent chain, but not one that
matches a head hash recorded elsewhere earlier.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import GENESIS_HASH, canonical_json, chain_hash, hash_array, hash_json
from .stats import annualise_sharpe, sharpe_moments

SCHEMA_VERSION = 1
DEFAULT_DIR = ".trial-ledger"
ENV_DIR = "TRIAL_LEDGER_DIR"


def default_ledger_dir() -> Path:
    return Path(os.environ.get(ENV_DIR, DEFAULT_DIR))


@dataclass
class VerifyIssue:
    line: int  # 1-based line number in ledger.jsonl (0 for file-level issues)
    kind: str
    message: str


@dataclass
class VerifyResult:
    n_records: int
    head_hash: str
    issues: list[VerifyIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


class Ledger:
    """A hash-chained trial ledger stored in a directory."""

    def __init__(
        self,
        directory: str | os.PathLike[str] | None = None,
        *,
        returns_directory: str | os.PathLike[str] | None = None,
    ):
        self.dir = Path(directory) if directory is not None else default_ledger_dir()
        self.path = self.dir / "ledger.jsonl"
        self.head_path = self.dir / "HEAD"
        self.store_pointer = self.dir / "RETURNS_STORE"
        if returns_directory is not None:
            self.returns_dir = Path(returns_directory).resolve()
        elif self.store_pointer.exists():
            self.returns_dir = self.dir / self.store_pointer.read_text().strip()
        else:
            self.returns_dir = self.dir / "returns"

    # ------------------------------------------------------------------ io helpers
    def _ensure(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.returns_dir.mkdir(parents=True, exist_ok=True)
        if self.returns_dir.resolve() != (self.dir / "returns").resolve():
            self.store_pointer.write_text(os.path.relpath(self.returns_dir, self.dir) + "\n")
        self.path.touch(exist_ok=True)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._ensure()
        with open(self.dir / ".lock", "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _last_record(self) -> dict[str, Any] | None:
        """Read the last non-empty line without scanning the whole file."""
        if not self.path.exists():
            return None
        with open(self.path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            end = fh.tell()
            if end == 0:
                return None
            block = 4096
            buf = b""
            pos = end
            while pos > 0:
                step = min(block, pos)
                pos -= step
                fh.seek(pos)
                buf = fh.read(step) + buf
                lines = buf.rstrip(b"\n").split(b"\n")
                if len(lines) > 1 or pos == 0:
                    return json.loads(lines[-1])
        return None

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
        return out

    def head(self) -> tuple[int, str]:
        last = self._last_record()
        if last is None:
            return 0, GENESIS_HASH
        return int(last["seq"]), str(last["hash"])

    # ------------------------------------------------------------------ returns store
    def store_returns(self, returns: np.ndarray) -> str:
        arr = np.ascontiguousarray(np.asarray(returns, dtype="<f8"))
        h = hash_array(arr)
        self._ensure()
        p = self.returns_dir / f"{h}.npz"
        if not p.exists():
            tmp = p.with_name(f".{h}.{uuid.uuid4().hex}.npz")
            try:
                np.savez_compressed(tmp, returns=arr)
                os.replace(tmp, p)
            finally:
                tmp.unlink(missing_ok=True)
        return h

    def load_returns(self, returns_hash: str) -> np.ndarray | None:
        if len(returns_hash) != 64 or any(c not in "0123456789abcdef" for c in returns_hash):
            raise ValueError("invalid returns hash")
        compressed = self.returns_dir / f"{returns_hash}.npz"
        p = self.returns_dir / f"{returns_hash}.npy"
        if compressed.exists():
            with np.load(compressed, allow_pickle=False) as stored:
                arr = stored["returns"]
        elif p.exists():
            arr = np.load(p, allow_pickle=False)
        else:
            return None
        if hash_array(arr) != returns_hash:
            raise ValueError(f"return series content does not match {returns_hash}")
        return arr

    # ------------------------------------------------------------------ append
    def append(
        self,
        *,
        strategy: dict[str, Any] | None = None,
        returns: np.ndarray | None = None,
        sharpe: float | None = None,
        n_obs: int | None = None,
        periods_per_year: float = 252.0,
        source: str = "python",
        agent: str | None = None,
        session_id: str | None = None,
        code_hash: str | None = None,
        dataset_hash: str | None = None,
        period: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        store_returns: bool = True,
    ) -> dict[str, Any]:
        """Append one trial and return the stored record.

        If ``returns`` is given, the Sharpe ratio, skewness, kurtosis, number of observations
        and returns hash are computed from it (and the series is stored for PBO unless
        ``store_returns=False``). Otherwise ``sharpe``/``n_obs`` may be supplied directly,
        e.g. by a hook that only sees a script's printed output.
        """
        skew = kurt = None
        returns_hash = None
        if returns is not None:
            r = np.asarray(returns, dtype=float)
            m = sharpe_moments(r)
            sharpe, n_obs, skew, kurt = m.sharpe, m.n_obs, m.skew, m.kurtosis
            returns_hash = self.store_returns(r) if store_returns else hash_array(r)

        body_core = {
            "schema": SCHEMA_VERSION,
            "trial_id": uuid.uuid4().hex,
            "timestamp": datetime.now(UTC).isoformat(timespec="microseconds"),
            "source": source,
            "agent": agent or os.environ.get("TRIAL_LEDGER_AGENT"),
            "session_id": session_id or os.environ.get("TRIAL_LEDGER_SESSION"),
            "strategy": strategy or {},
            "strategy_hash": hash_json(strategy or {}),
            "code_hash": code_hash,
            "dataset_hash": dataset_hash,
            "period": period,
            "n_obs": n_obs,
            "returns_hash": returns_hash,
            "sharpe": sharpe,
            "sharpe_ann": None if sharpe is None else annualise_sharpe(sharpe, periods_per_year),
            "periods_per_year": periods_per_year,
            "skew": skew,
            "kurtosis": kurt,
            "extra": extra or {},
        }
        with self._locked():
            seq, prev = self.head()
            body = {"seq": seq + 1, "prev_hash": prev, **body_core}
            record = {**body, "hash": chain_hash(prev, body)}
            line = canonical_json(record) + "\n"
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            tmp = self.head_path.with_suffix(".tmp")
            tmp.write_text(f"{record['seq']} {record['hash']}\n")
            os.replace(tmp, self.head_path)
        return record

    # ------------------------------------------------------------------ verify
    def verify(self, expect_head: str | None = None, check_returns: bool = False) -> VerifyResult:
        """Re-hash the chain and report every inconsistency found.

        ``expect_head`` is an externally anchored head hash (or ``"<seq> <hash>"``). When
        omitted, the ``HEAD`` file is used as the anchor.
        """
        issues: list[VerifyIssue] = []
        prev = GENESIS_HASH
        expected_seq = 1
        n = 0
        last_hash = GENESIS_HASH
        if not self.path.exists():
            return VerifyResult(0, GENESIS_HASH, [VerifyIssue(0, "missing", f"{self.path}")])
        with open(self.path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    issues.append(VerifyIssue(lineno, "parse_error", str(exc)))
                    continue
                n += 1
                stored = rec.pop("hash", None)
                recomputed = chain_hash(rec.get("prev_hash", ""), rec)
                if stored != recomputed:
                    issues.append(
                        VerifyIssue(lineno, "hash_mismatch", "record content was modified")
                    )
                if rec.get("prev_hash") != prev:
                    issues.append(
                        VerifyIssue(
                            lineno,
                            "broken_link",
                            "prev_hash does not match the previous record "
                            "(a record was deleted, inserted or reordered)",
                        )
                    )
                if rec.get("seq") != expected_seq:
                    issues.append(
                        VerifyIssue(
                            lineno,
                            "seq_gap",
                            f"expected seq {expected_seq}, found {rec.get('seq')}",
                        )
                    )
                if check_returns and rec.get("returns_hash"):
                    try:
                        arr = self.load_returns(rec["returns_hash"])
                    except (ValueError, OSError):
                        issues.append(VerifyIssue(lineno, "returns_mismatch", rec["returns_hash"]))
                    else:
                        if arr is None:
                            issues.append(
                                VerifyIssue(lineno, "returns_missing", rec["returns_hash"])
                            )
                prev = stored if stored is not None else recomputed
                last_hash = prev
                expected_seq = (rec.get("seq") or expected_seq) + 1

        anchor = expect_head
        if anchor is None and self.head_path.exists():
            anchor = self.head_path.read_text().strip()
        if anchor:
            parts = anchor.split()
            anchor_hash = parts[-1]
            anchor_seq = int(parts[0]) if len(parts) == 2 else None
            if anchor_hash != last_hash:
                issues.append(
                    VerifyIssue(
                        0,
                        "head_mismatch",
                        f"ledger ends at {last_hash[:16]}... but the anchor is "
                        f"{anchor_hash[:16]}... (records truncated or chain rebuilt)",
                    )
                )
            elif anchor_seq is not None and anchor_seq != n:
                issues.append(
                    VerifyIssue(0, "head_mismatch", f"anchor says {anchor_seq} records, found {n}")
                )
        return VerifyResult(n_records=n, head_hash=last_hash, issues=issues)


def record_trial(returns: np.ndarray, strategy: dict[str, Any] | None = None, **kwargs: Any):
    """Convenience API for backtest scripts: ``record_trial(strategy_returns, {"lookback": 20})``.

    Writes to ``$TRIAL_LEDGER_DIR`` (default ``.trial-ledger``).
    """
    return Ledger().append(returns=returns, strategy=strategy, source="python", **kwargs)
