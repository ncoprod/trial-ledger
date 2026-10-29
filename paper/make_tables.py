"""Generate paper inputs from committed results; never read in-flight experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/generated"
RESULTS = "experiments/results/causal-v2/"
SOURCES = {}
NAMES = {
    "codex-cli/gpt-6.1-sol": "Codex (gpt-6.1-sol)",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "nvidia/nemotron-3.5-lightning-30b-a3b": "Nemotron Lightning",
    "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron Ultra",
    "random-search": "Random control",
}


def committed(path):
    data = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)
    SOURCES[path] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def tex(value):
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def table(name, caption, headers, rows, columns):
    lines = [
        r"\begin{table}[tbp]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}\label{{tab:{name}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        *(" & ".join(map(str, row)) + r" \\" for row in rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def rate(result):
    lo, hi = result["wilson_95_ci"]
    return rf"{result['passes']}/{result['n']} [{100 * lo:.1f}, {100 * hi:.1f}]"


def pvalue(value):
    return f"{value:.3g}"


def acceptance_plot(headlines):
    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    x = np.arange(2)
    for offset, key, label, color in [
        (-1, "psr0", "Naive PSR", "#cc792c"),
        (0, "dsr_claimed", "DSR: claimed N", "#4d94bd"),
        (1, "dsr_ledger", "DSR: ledger N", "#34534b"),
    ]:
        values = [headlines[c][key] for c in ("cli", "nim")]
        means = np.array([r["passes"] / r["n"] for r in values])
        lower, upper = np.array([r["wilson_95_ci"] for r in values]).T
        ax.bar(
            x + offset * 0.24,
            means * 100,
            width=0.24,
            label=label,
            color=color,
            yerr=np.array([means - lower, upper - means]) * 100,
            capsize=3,
        )
    ax.set(xticks=x, xticklabels=["Codex CLI", "NIM models"], ylim=(0, 108))
    ax.set_ylabel("Valid null picks accepted (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "acceptance.png", dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = committed(RESULTS + "episodes.json")["rows"]
    paired = committed(RESULTS + "paired_tests.json")
    store = committed(RESULTS + "returns_integrity.json")
    provenance = committed(RESULTS + "provenance.json")
    benchmark = committed("benchmarks/results/linux/pbo_benchmark.json")
    kernel = committed("benchmarks/results/linux/kernel_interleaved.json")
    claude = committed(RESULTS + "attempts/claude-cli__haiku/signflip-s0-r0/episode.json")
    assert len(episodes) == 265, "the fixed sample is incomplete"
    assert all(r["ledger_ok"] and not r["replay_mismatches"] for r in episodes)
    assert store["ledgers"] == len(episodes)
    h = paired["headlines"]
    cli, nim = h["cli"], h["nim"]
    by_model = {m: [r for r in episodes if r["model"] == m] for m in NAMES}
    tests = paired["per_model_paired_tests"]
    discordance = sum(
        t["claimed_only_pass"] + t["ledger_only_pass"]
        for model, t in tests.items()
        if model != "random-search"
    )
    pairs = sum(t["episodes"] for m, t in tests.items() if m != "random-search")
    sentence = (
        f"Claimed- and ledger-count DSR decisions differ in {discordance}/{pairs} "
        "valid paired null picks at the primary threshold."
    )
    speeds = [r["speedup_rust_vs_pypbo"] for r in benchmark["rows"] if r["pypbo_s"]]
    largest = max(kernel["rows"], key=lambda r: r["T"] * r["N"])
    config = provenance["config"]
    numbers = {
        "CorrectedEpisodes": len(episodes),
        "CliTotal": len(by_model["codex-cli/gpt-6.1-sol"]),
        "CliValidNull": cli["episodes"],
        "CliNaive": cli["psr0"]["passes"],
        "CliLedger": cli["dsr_ledger"]["passes"],
        "NimValidNull": nim["episodes"],
        "NimNaive": nim["psr0"]["passes"],
        "NimLedger": nim["dsr_ledger"]["passes"],
        "NimUnder": nim["underreported"],
        "NimClaims": nim["claims_available"],
        "MaxUnder": f"{nim['max_ledger_over_claimed']:.3f}",
        "PrimaryDecisionSentence": sentence,
        "SpeedLow": str(int(min(speeds))),
        "SpeedHigh": str(int(max(speeds))),
        "ResearchStart": config["is_period"][0],
        "ResearchEnd": config["is_period"][1],
        "TestStart": config["oos_period"][0],
        "TestEnd": config["oos_period"][1],
        "ProtocolCommit": "5dcf789a",
        "AnalysisCommit": provenance["git_commit"][:12],
        "StoredLedgers": store["ledgers"],
        "StoredTrials": f"{store['trials']:,}",
        "StoredObjects": f"{store['unique_compressed_objects']:,}",
        "RustOriginalMs": f"{largest['medians_seconds']['original'] * 1000:.3f}",
        "RustFinalMs": f"{largest['medians_seconds']['branchless'] * 1000:.3f}",
    }
    (OUT / "numbers.tex").write_text(
        "\n".join(rf"\newcommand{{\{key}}}{{{value}}}" for key, value in numbers.items()) + "\n"
    )
    table(
        "cohorts",
        "Corrected null-market acceptance. Brackets are nominal Wilson 95\\% "
        "intervals in percent; fractions use valid picks.",
        ["Cohort", "Attempts", "Naive PSR", "DSR claimed $N$", "DSR ledger $N$"],
        [
            [
                name,
                h[c]["attempts"],
                *(rate(h[c][k]) for k in ("psr0", "dsr_claimed", "dsr_ledger")),
            ]
            for c, name in (("cli", "Codex"), ("nim", "NIM"))
        ],
        "lrrrr",
    )
    model_rows = []
    for model, rows in by_model.items():
        null = [r for r in rows if r["market"] != "btc"]
        valid = [r for r in null if r["submitted_valid"]]
        claims = [r for r in null if r.get("claimed_n")]
        under = sum(r["claimed_n"] < r["n_ledger"] for r in claims)
        model_rows.append(
            [
                NAMES[model],
                len(null),
                len(valid),
                f"{np.median([r['n_ledger'] for r in null]):g}",
                f"{np.median([r['claimed_n'] for r in claims]):g}" if claims else "--",
                f"{under}/{len(claims)}",
            ]
        )
    table(
        "models",
        "Null attempts, valid submissions and count accounting. Medians "
        "are across attempts or available positive claims; invalid picks retain their claims.",
        ["Subject", "Attempts", "Valid", "Median $N_L$", "Median $N_C$", "Under-reports"],
        model_rows,
        "lrrrrr",
    )
    table(
        "paired",
        "Primary paired comparisons at 0.95. Discordant decisions and exact "
        "McNemar $p$ are distinct from continuous-score signed-rank tests (Holm adjusted).",
        ["Subject", "Pairs", "Claim only", "Ledger only", "$p_{McN}$", "$p_{W,Holm}$"],
        [
            [
                NAMES[m],
                t["episodes"],
                t["claimed_only_pass"],
                t["ledger_only_pass"],
                pvalue(t["mcnemar_exact_two_sided_p"]),
                pvalue(t["wilcoxon_holm_p"]),
            ]
            for m, t in tests.items()
        ],
        "lrrrrr",
    )
    sensitivity = [
        [
            f"Observed claims, threshold {threshold}",
            result["episodes"],
            result["claimed_only_pass"],
            result["ledger_only_pass"],
        ]
        for threshold, result in paired["exploratory_threshold_sensitivity"].items()
    ]
    sensitivity += [
        [
            rf"Hypothetical $N_C=\lfloor {r['claimed_fraction_of_ledger_n']}N_L\rfloor$",
            r["episodes"],
            r["claimed_only_pass"],
            0,
        ]
        for r in paired["counterfactual_underreporting_not_observed_claims"]
    ]
    table(
        "sensitivity",
        "Exploratory conditions for differing decisions. Counterfactual counts "
        "are floored at one and evaluated at 0.95; they are not observed agent claims.",
        ["Scenario", "Picks", "Claim only", "Ledger only"],
        sensitivity,
        "lrrr",
    )
    outside = sum(r["submitted_valid"] and not r["submitted_in_ledger"] for r in episodes)
    (OUT / "submission_notes.tex").write_text(
        f"Across all markets, {outside} valid submitted specifications were absent "
        "from their episode's recorded search. They are flagged and scored as submitted; "
        "no best recorded trial replaces them.\n"
    )
    failures = [
        [NAMES[m], tex(status), count]
        for m, rows in by_model.items()
        for status, count in sorted(Counter(r["status"] for r in rows).items())
    ]
    failures.append(["Claude Code Haiku pilot", tex(claude["status"]), 1])
    table(
        "failures",
        "Every attempt status, including the separate authentication pilot.",
        ["Subject", "Status", "Attempts"],
        failures,
        "llr",
    )
    btc_rows = []
    for model, rows in by_model.items():
        btc = [r for r in rows if r["market"] == "btc"]
        valid = [r for r in btc if r.get("oos_sharpe") is not None]
        specs = {json.dumps(r["selected_strategy"], sort_keys=True) for r in valid}
        btc_rows.append(
            [
                NAMES[model],
                len(btc),
                len(valid),
                len(specs),
                f"{np.median([r['is_sharpe'] for r in valid]):.3f}" if valid else "--",
                f"{np.median([r['oos_sharpe'] for r in valid]):.3f}" if valid else "--",
            ]
        )
    table(
        "btc",
        "BTC diagnostic on a previously exposed period. Annualized Sharpe "
        "medians use valid submitted rules; repeated rules are counted explicitly.",
        ["Subject", "Attempts", "Valid", "Rules", "Research", "Diagnostic"],
        btc_rows,
        "lrrrrr",
    )
    bench_rows = []
    for row in benchmark["rows"]:
        assert row["checks"]["numpy_pbo_exact"] and row["checks"]["rust_threads_byte_identical"]
        assert row["pypbo_s"] is None or row["checks"]["pypbo_checked"]
        bench_rows.append(
            [
                row["T"],
                row["N"],
                row["S"],
                f"{row['rust_parallel_s'] * 1000:.3f}",
                f"{row['pypbo_s'] * 1000:.3f}" if row["pypbo_s"] else "memory cap",
                f"{row['speedup_rust_vs_pypbo']:.1f}" if row["pypbo_s"] else "--",
            ]
        )
    table(
        "benchmark",
        "Isolated Linux median wall times (ms), two CPU threads. "
        "All saved samples contribute; the largest pypbo allocation is skipped.",
        ["$T$", "$N$", "$S$", "Rust", "pypbo", "Ratio"],
        bench_rows,
        "rrrrrr",
    )
    table(
        "provenance",
        "Committed source artifacts used by the paper generator. "
        "The complete SHA-256 map is in generated/manifest.json.",
        ["Artifact", "Source commit"],
        [
            ["Corrected protocol", r"\texttt{5dcf789a}"],
            ["Episode analysis and stored returns", rf"\texttt{{{provenance['git_commit'][:12]}}}"],
            [
                "Linux native benchmark",
                rf"\texttt{{{benchmark['meta']['provenance']['git_commit'][:12]}}}",
            ],
        ],
        "ll",
    )
    acceptance_plot(h)
    manifest = {
        "command": "uv run --no-sync python paper/make_tables.py",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "input_sha256": SOURCES,
        "generated_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(OUT.iterdir())
            if p.name != "manifest.json"
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated paper inputs from {len(episodes)} committed attempts")


if __name__ == "__main__":
    main()
