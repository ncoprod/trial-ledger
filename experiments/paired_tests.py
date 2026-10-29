"""Prespecified paired acceptance tests and explicitly exploratory sensitivity checks."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

from trial_ledger.stats import deflated_sharpe_ratio

from .agent import RESULTS
from .analyze import THRESH, wilson
from .provenance import provenance


def paired(rows: list[dict], threshold: float = THRESH) -> dict:
    valid = [
        r for r in rows if r.get("dsr_claimed") is not None and r.get("dsr_ledger") is not None
    ]
    claimed_only = sum(r["dsr_claimed"] > threshold >= r["dsr_ledger"] for r in valid)
    ledger_only = sum(r["dsr_ledger"] > threshold >= r["dsr_claimed"] for r in valid)
    discordant = claimed_only + ledger_only
    delta = np.array([r["dsr_claimed"] - r["dsr_ledger"] for r in valid])
    p = (
        float(wilcoxon(delta, alternative="two-sided", method="approx").pvalue)
        if np.any(delta)
        else 1.0
    )
    return {
        "episodes": len(valid),
        "threshold": threshold,
        "claimed_pass": sum(r["dsr_claimed"] > threshold for r in valid),
        "ledger_pass": sum(r["dsr_ledger"] > threshold for r in valid),
        "claimed_only_pass": claimed_only,
        "ledger_only_pass": ledger_only,
        "mcnemar_exact_two_sided_p": float(binomtest(claimed_only, discordant, 0.5).pvalue)
        if discordant
        else 1.0,
        "median_claimed_minus_ledger": float(np.median(delta)) if len(delta) else None,
        "wilcoxon_two_sided_p": p,
        "disagreement_wilson_ci": wilson(discordant, len(valid)) if valid else None,
    }


def main(rows: list[dict] | None = None) -> None:
    if rows is None:
        rows = json.loads((RESULTS / "episodes.json").read_text())["rows"]
    null = [r for r in rows if r["market"] != "btc" and r.get("status") == "ok"]
    groups = defaultdict(list)
    for r in null:
        groups[r["model"]].append(r)
    tests = {model: paired(rs) for model, rs in groups.items()}
    # Holm correction across model-specific continuous-score tests (no pooled iid claim).
    ordered = sorted(tests, key=lambda m: tests[m]["wilcoxon_two_sided_p"])
    running = 0.0
    for i, model in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * tests[model]["wilcoxon_two_sided_p"]))
        tests[model]["wilcoxon_holm_p"] = running
    llm = [r for r in null if r["model"] != "random-search"]
    cohorts = {
        "nim": [r for r in llm if "-cli/" not in r["model"]],
        "cli": [r for r in llm if "-cli/" in r["model"]],
    }
    headlines = {}
    for cohort, rs in cohorts.items():
        valid = [r for r in rs if r.get("psr0") is not None]
        n = len(valid)
        result = {"episodes": n}
        for key in ("psr0", "dsr_claimed", "dsr_ledger"):
            scored = [r for r in valid if r.get(key) is not None]
            k = sum(r[key] > THRESH for r in scored)
            ci = wilson(k, len(scored)) if scored else None
            result[key] = {
                "passes": k,
                "n": len(scored),
                "wilson_95_ci": ci,
                "half_width": (ci[1] - ci[0]) / 2 if ci else None,
            }
        claims = [r for r in valid if r.get("claimed_n")]
        under = [r for r in claims if r["claimed_n"] < r["n_ledger"]]
        result["underreported"] = len(under)
        result["claims_available"] = len(claims)
        result["max_ledger_over_claimed"] = max(
            (r["undercount_ratio"] for r in claims), default=None
        )
        headlines[cohort] = result
    sensitivity = {str(t): paired(llm, t) for t in (0.5, 0.8, 0.9, 0.95, 0.975, 0.99)}
    counterfactual = []
    for factor in (0.5, 0.25, 0.1, 0.02):
        examples = []
        for r in llm:
            m = r["selected_moments"]
            hypothetical_n = max(1, int(r["n_ledger"] * factor))
            hypothetical_dsr = deflated_sharpe_ratio(
                m["sharpe"],
                m["n_obs"],
                hypothetical_n,
                m["var_trials_sharpe"],
                m["skew"],
                m["kurtosis"],
            )
            if hypothetical_dsr > THRESH >= r["dsr_ledger"]:
                examples.append(
                    {
                        "model": r["model"],
                        "market": r["market"],
                        "seed": r["seed"],
                        "rep": r["rep"],
                        "ledger_n": r["n_ledger"],
                        "hypothetical_n": hypothetical_n,
                        "hypothetical_dsr": hypothetical_dsr,
                        "ledger_dsr": r["dsr_ledger"],
                    }
                )
        counterfactual.append(
            {
                "claimed_fraction_of_ledger_n": factor,
                "episodes": len(llm),
                "claimed_only_pass": len(examples),
                "examples": examples,
            }
        )
    boundaries = [
        {
            "model": r["model"],
            "market": r["market"],
            "seed": r["seed"],
            "rep": r["rep"],
            "claimed_n": r["claimed_n"],
            "ledger_n": r["n_ledger"],
            "dsr_claimed": r["dsr_claimed"],
            "dsr_ledger": r["dsr_ledger"],
            "disagreement_threshold_interval": [
                min(r["dsr_claimed"], r["dsr_ledger"]),
                max(r["dsr_claimed"], r["dsr_ledger"]),
            ],
        }
        for r in llm
        if r.get("dsr_claimed") is not None and abs(r["dsr_claimed"] - r["dsr_ledger"]) > 0.05
    ]
    output = {
        "provenance": provenance(
            {
                "threshold": THRESH,
                "wilson_z": 1.96,
                "sensitivity_thresholds": [0.5, 0.8, 0.9, 0.95, 0.975, 0.99],
            }
        ),
        "headlines": headlines,
        "per_model_paired_tests": tests,
        "exploratory_threshold_sensitivity": sensitivity,
        "counterfactual_underreporting_not_observed_claims": counterfactual,
        "exploratory_observed_boundary_intervals": boundaries,
        "caveats": [
            "Threshold sensitivity is exploratory; acceptance threshold remains 0.95.",
            "Wilson intervals describe episode proportions; models share market seeds.",
            "Pooled episodes are not independent; inference tests are model-specific.",
            "No difference in decisions is not evidence that counts are equivalent.",
        ],
    }
    (RESULTS / "paired_tests.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
