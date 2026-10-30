# Round 3: fixed transfer diagnostic and clearer evidence

Fixed before downloading or evaluating ETH. This is an internal versioned plan,
not external preregistration. Source selections and null metrics are the committed
`experiments/results/causal-v2/episodes.json` at
`83d6e48dfd6dcbef66124355556303239c03e727`.

## Choices and targets

1. Evaluate **all** BTC submissions from that source on Binance spot ETHUSDT daily
   closes, with the unchanged strategy engine and five-basis-point turnover cost.
   Keep the original research dates (2018-01-01 through 2023-12-31) and later dates
   (2024-01-01 through 2026-09-30). No search, parameter fitting, prompt change,
   threshold selection or seed selection on ETH. Repeated selections stay in
   episode accounting; evaluate each distinct normalized specification once over
   the full series, then slice its returns for both periods. Warm indicators and
   carry positions across the split. Log each evaluation and a fixed buy-and-hold
   reference, including compressed returns and date-axis identity. Missing or
   invalid submissions remain visible and are never substituted.
2. Decompose the existing null results by model and market, retaining attempt,
   valid-pick and positive-count-claim denominators. Report Wilson intervals for
   each stratum. Reconstruct the original headline counts exactly; no pooling
   models or markets to claim independent replication or narrower uncertainty.
3. Replace the seven-column README results table with at most four columns,
   emphasizing naive versus ledger-DSR decisions. Keep self-report accounting and
   intervals in separate compact tables/linked results. Generate these views from
   committed evidence, update the diagram, and generate the paper extension from
   the committed round-3 result rather than typing empirical cells.

Targets: complete selection accounting; verifiable ETH return objects; exact
headline reconciliation; offline deterministic regeneration; simpler README;
local CI passes. Stop when met. No strategy is selected for a second ETH attempt.
If an implementation defect prevents a target, log the defect and correction;
stop after two iterations without improvement and explain the remaining limit.

## Interpretation and multiplicity

The BTC selections already reflect adaptive search and examined BTC results.
ETH shares crypto-market exposure with BTC and public history may be represented
in pretrained models. This is a descriptive cross-asset transfer case, not an
untouched test or an independent demonstration of alpha. Report every model,
every selection, unique-rule counts, research/later Sharpe, later drawdown and
buy-and-hold comparison. Do not add an ETH significance claim or reuse BTC trial
N as if ETH were a new independently searched candidate universe. Record every
distinct evaluation as an additional trial, separate from both prior studies.

Additional NIM cohorts are deferred: the existing nominal headline half-widths
already meet the power target; more models on the same market seeds would change
the cohort mixture without creating independent market replication. No NIM call,
local ML training or Kaggle GPU is needed for this round.

Binance source checksums, normalized CSV, exact commands, source commit, settings,
source-file hashes, result hashes and every ledger head must accompany results.
Keep the existing Binance attribution and non-commercial/share-alike restrictions.
