"""Fixed BTC-to-ETH transfer and null-stratum reporting; no inference or search.

Download once: uv run python -m experiments.round3 --download
Evaluate or verify offline: uv run python -m experiments.round3
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import zipfile
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

from trial_ledger.backtest import Market, normalise_spec, run_backtest
from trial_ledger.hashing import hash_array, hash_file, hash_json, sha256_hex
from trial_ledger.ledger import Ledger
from trial_ledger.stats import annualise_sharpe, sharpe_moments

from .analyze import THRESH, wilson
from .data import BASE, RAW, _get, _months
from .provenance import provenance

SOURCE_COMMIT = "83d6e48dfd6dcbef66124355556303239c03e727"
PLAN_COMMIT = "7e0084f83ba02d84e00d862d7a6a7574707f4e3f"
SOURCE = "experiments/results/causal-v2/episodes.json"
SOURCE_SHA256 = "cd5e6a77f31c0844926dd94d0e9d89a58814994ad41eb85802aff970ba54636d"
OUT = Path(__file__).parent / "results/round3"
ETH = Path(__file__).parent / "data/ethusdt_1d.csv"
DATA_RECEIPT = ETH.with_suffix(".json")


def download_eth():
    RAW.mkdir(parents=True, exist_ok=True)
    closes, archives = {}, []
    for month in _months("2018-01", "2026-09"):
        url = BASE.format(sym="ETHUSDT", ym=month)
        archive = RAW / Path(url).name
        checksum = archive.with_suffix(".zip.CHECKSUM")
        if not checksum.exists():
            checksum.write_bytes(_get(url + ".CHECKSUM"))
        if not archive.exists():
            archive.write_bytes(_get(url))
        expected = checksum.read_text().split()[0]
        if hash_file(archive) != expected:
            raise ValueError(f"ETH archive checksum mismatch: {month}")
        with zipfile.ZipFile(archive) as zipped:
            for row in csv.reader(io.TextIOWrapper(zipped.open(zipped.namelist()[0]))):
                if not row[0].isdigit():
                    continue
                timestamp = int(row[0])
                timestamp /= 1e6 if timestamp > 1e14 else 1e3
                day = datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%d")
                closes[day] = row[4]
        archives.append({"url": url, "sha256": expected})
        print(f"Verified ETH {month}", flush=True)
    expected_days = [
        (date(2018, 1, 1) + timedelta(days=i)).isoformat()
        for i in range((date(2026, 9, 30) - date(2018, 1, 1)).days + 1)
    ]
    if sorted(closes) != expected_days:
        raise ValueError("ETH dates are incomplete or outside the fixed plan")
    with ETH.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "close"])
        writer.writerows((day, closes[day]) for day in expected_days)
    terms = "https://raw.githubusercontent.com/binance/binance-public-data/master/TERMS_AND_CONDITIONS.md"
    receipt = provenance({"symbol": "ETHUSDT", "interval": "1d", "archives": archives})
    receipt.update(
        {
            "csv_sha256": hash_file(ETH),
            "rows": len(closes),
            "terms_url": terms,
            "terms_sha256": sha256_hex(_get(terms)),
            "license": "CC BY-NC-SA 4.0 plus Binance Vision Dataset Terms 1.0",
            "attribution": "Binance Public Data / Binance, ETHUSDT spot daily klines",
            "modifications": "Extract daily closes; normalize millisecond/microsecond timestamps.",
        }
    )
    DATA_RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")


def null_strata(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["market"] != "btc":
            groups[(row["model"], row["market"])].append(row)
    result = []
    for (model, market), attempts in sorted(groups.items()):
        valid = [r for r in attempts if r["status"] == "ok" and r.get("psr0") is not None]
        claims = [r for r in attempts if r.get("claimed_n")]
        item = {"model": model, "market": market, "attempts": len(attempts), "valid": len(valid)}
        for field in ("psr0", "dsr_claimed", "dsr_ledger"):
            scored = [r for r in valid if r.get(field) is not None]
            passes = sum(r[field] > THRESH for r in scored)
            item[field] = {
                "passes": passes,
                "n": len(scored),
                "wilson_95_ci": list(wilson(passes, len(scored))) if scored else None,
            }
        item.update(
            {
                "positive_claims": len(claims),
                "underreported": sum(r["claimed_n"] < r["n_ledger"] for r in claims),
            }
        )
        result.append(item)
    return result


def metrics(returns):
    moments = sharpe_moments(returns)
    equity = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return {
        "n_obs": len(returns),
        "sharpe_ann": annualise_sharpe(moments.sharpe, 365),
        "max_drawdown": float(np.max(1 - equity / peaks)),
        "returns_hash": hash_array(returns),
    }


def transfer(rows, market, ledger):
    selections = [r for r in rows if r["market"] == "btc"]
    specs = {
        hash_json(normalise_spec(r["selected_strategy"])): normalise_spec(r["selected_strategy"])
        for r in selections
        if r["status"] == "ok" and r["submitted_valid"]
    }
    split = market.dates.index("2024-01-01") - 1
    saved = ledger.records()
    candidates = {}
    for index, (key, spec) in enumerate(
        [*sorted(specs.items()), ("buy_and_hold", {"reference": "buy_and_hold"})]
    ):
        if key == "buy_and_hold":
            returns = market.returns[1:].copy()
            returns[0] -= market.fee_bps * 1e-4
        else:
            returns = run_backtest(spec, market).returns
        if index < len(saved):
            if saved[index]["strategy"] != spec or saved[index]["returns_hash"] != hash_array(
                returns
            ):
                raise ValueError("ETH replay differs from its recorded trial")
        else:
            ledger.append(
                strategy=spec,
                returns=returns,
                dataset_hash=market.dataset_hash,
                period={"time_axis_hash": hash_json(market.dates[1:])},
                periods_per_year=365,
                agent="round3/fixed-btc-transfer",
            )
        candidates[key] = {
            "strategy": spec,
            "research": metrics(returns[:split]),
            "later": metrics(returns[split:]),
            "full_returns_hash": hash_array(returns),
        }
    if len(ledger.records()) != len(specs) + 1:
        raise ValueError("unexpected ETH trials outside the frozen candidate set")
    summaries = []
    for model in sorted({r["model"] for r in selections}):
        attempts = [r for r in selections if r["model"] == model]
        keys = [
            hash_json(normalise_spec(r["selected_strategy"]))
            for r in attempts
            if r["status"] == "ok" and r["submitted_valid"]
        ]
        summaries.append(
            {
                "model": model,
                "attempts": len(attempts),
                "valid": len(keys),
                "unique_rules": len(set(keys)),
                "median_research_sharpe": float(
                    np.median([candidates[k]["research"]["sharpe_ann"] for k in keys])
                )
                if keys
                else None,
                "median_later_sharpe": float(
                    np.median([candidates[k]["later"]["sharpe_ann"] for k in keys])
                )
                if keys
                else None,
                "below_buy_and_hold": sum(
                    candidates[k]["later"]["sharpe_ann"]
                    < candidates["buy_and_hold"]["later"]["sharpe_ann"]
                    for k in keys
                ),
            }
        )
    return {
        "selections": [
            {
                k: r[k]
                for k in ("model", "seed", "rep", "status", "selected_strategy", "submitted_valid")
            }
            for r in selections
        ],
        "unique_rules": len(specs),
        "evaluations": len(specs) + 1,
        "candidates": candidates,
        "per_model": summaries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    if parser.parse_args().download:
        download_eth()
        return
    raw = subprocess.check_output(["git", "show", f"HEAD:{SOURCE}"])
    if sha256_hex(raw) != SOURCE_SHA256:
        raise ValueError("source snapshot differs from the frozen round-3 input")
    rows = json.loads(raw)["rows"]
    data_receipt = json.loads(DATA_RECEIPT.read_text())
    if hash_file(ETH) != data_receipt["csv_sha256"]:
        raise ValueError("ETH CSV differs from its download receipt")
    market = Market.from_csv(ETH, periods_per_year=365, fee_bps=5)
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(OUT / "ledger")
    existing = (
        json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else None
    )
    if existing is None and not ledger.dir.exists():
        ledger._ensure()
    verified = ledger.verify(
        expect_head=existing["ledger_head"] if existing else None, check_returns=True
    )
    if not verified.ok:
        raise ValueError("ETH ledger verification failed")
    result = {
        "null_strata": null_strata(rows),
        "eth": transfer(rows, market, ledger),
        "ledger_head": " ".join(map(str, ledger.head())),
    }
    if existing:
        if result != {k: existing[k] for k in result}:
            raise ValueError("round-3 result differs from its committed arithmetic")
        print("PASS: round-3 offline replay; all candidates and ledger returns match")
        return
    result["provenance"] = provenance(
        {
            "plan_commit": PLAN_COMMIT,
            "source_commit": SOURCE_COMMIT,
            "source_sha256": sha256_hex(raw),
            "source_path": SOURCE,
            "data_sha256": hash_file(ETH),
            "research": ["2018-01-01", "2023-12-31"],
            "later": ["2024-01-01", "2026-09-30"],
            "fee_bps": 5,
            "periods_per_year": 365,
            "new_inference_calls": 0,
            "interpretation": "descriptive transfer on correlated assets; no alpha claim",
        }
    )
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Saved all {len(result['eth']['selections'])} selections; "
        f"{result['eth']['evaluations']} ETH evaluations"
    )


if __name__ == "__main__":
    main()
