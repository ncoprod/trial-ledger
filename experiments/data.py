"""Download BTCUSDT daily klines from Binance public data (https://data.binance.vision).

No API key is needed. Every monthly archive is verified against Binance's published
SHA-256 ``.CHECKSUM`` file, and the processed series is written to
``experiments/data/btcusdt_1d.csv`` (committed, so the experiment runs offline).
"""

from __future__ import annotations

import csv
import hashlib
import io
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

BASE = "https://data.binance.vision/data/spot/monthly/klines/{sym}/1d/{sym}-1d-{ym}.zip"
HERE = Path(__file__).parent / "data"
RAW = HERE / "raw"
OUT = HERE / "btcusdt_1d.csv"


def _months(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    out = []
    while (y, m) <= (ye, me):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "trial-ledger-research"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def download(symbol: str = "BTCUSDT", start: str = "2018-01", end: str = "2026-09") -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    rows: dict[str, float] = {}
    for ym in _months(start, end):
        url = BASE.format(sym=symbol, ym=ym)
        zpath = RAW / Path(url).name
        if not zpath.exists():
            blob = _get(url)
            expected = _get(url + ".CHECKSUM").decode().split()[0]
            if hashlib.sha256(blob).hexdigest() != expected:
                raise RuntimeError(f"checksum mismatch for {url}")
            zpath.write_bytes(blob)
        with zipfile.ZipFile(zpath) as zf:
            name = zf.namelist()[0]
            for line in io.TextIOWrapper(zf.open(name), encoding="utf-8"):
                parts = line.strip().split(",")
                if not parts[0].isdigit():
                    continue  # header row in some archives
                ts = int(parts[0])
                ts = ts / 1e6 if ts > 1e14 else ts / 1e3  # microseconds from 2025 on
                day = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
                rows[day] = float(parts[4])  # close
    days = sorted(rows)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "close"])
        for d in days:
            w.writerow([d, f"{rows[d]:.2f}"])
    return OUT


if __name__ == "__main__":
    p = download()
    print(p, sum(1 for _ in open(p)) - 1, "rows")
