# Corrected evaluation protocol: causal-v2

This plan is fixed before the corrected searches. It is an internal, versioned
protocol, not an externally registered preregistration. Earlier outcomes and the
BTC test period have already been examined. The corrected BTC rerun is therefore
a post-review diagnostic on a previously exposed period, not an untouched test.
No sizing parameter, prompt, seed, strategy family or threshold is tuned using it.

## Correction and fixed rerun plan

- Volatility targeting uses a fixed per-period standard deviation target of 0.02,
  the trailing 20 observations available at the close, and a 2x leverage cap.
  Positions are shifted one observation before earning returns; five-basis-point
  turnover costs use actual prior holdings. The target is not fitted to any window.
- Fresh NIM searches: all three existing models, seeds 0-19 on both null markets,
  plus five BTC episodes each. Same temperature, budgets and acceptance threshold.
- Fresh Codex CLI searches: seeds 0-49 on both null markets plus five BTC episodes,
  each in a new isolated workspace, gpt-6.1-sol with medium reasoning. The fixed
  larger sample targets Wilson half-width <=10 points even near a 50% rate.
- Random controls: seeds 0-9 on both nulls plus five BTC episodes.
- Total plan: 265 new episodes (135 NIM, 105 Codex, 25 random). Failures are retained,
  never replaced. Claude Haiku is attempted again; an authentication failure is
  recorded separately and does not replace a scheduled subject.
- New outputs live in `results/causal-v2/`; the 225 original receipts and results
  at `results/` remain superseded, immutable evidence. Protocol text in prompts
  distinguishes new inference cache keys. Both studies' search trials remain
  counted separately, with no pooling, best-run choice or claim of independent
  confirmation between versions. There are two evaluation versions, not two
  opportunities to select a favourable result.
- No predictive model is trained or validated. This fixes an implementation defect;
  all strategy selection uses the research window, and each fixed BTC candidate
  receives one test evaluation after submission. Reproduction repeats arithmetic.

## Original protocol (superseded where it claims clean lagging/holdout)


The fixed extension plan and all engineering iterations are recorded in
[`../ITERATIONS.md`](../ITERATIONS.md). This protocol does not optimize a model,
prompt, acceptance threshold or seed set against held-out BTC results.

## Subjects and sample

- NIM: `openai/gpt-oss-20b`, `nvidia/nemotron-3.5-lightning-30b-a3b`,
  `nvidia/nemotron-3-ultra-550b-a55b`; seeds 0–19 on signflip and GARCH nulls.
  Existing five BTC episodes per model are retained. Temperature 0.7; the market
  seed and cache salt are not a provider-side inference seed.
- Codex CLI: `gpt-6.1-sol`, medium reasoning, seeds 0–29 on both nulls, then five
  BTC episodes labelled seeds 0–4. Existing local sign-in, fresh temporary
  workspace and configuration per episode. No home instructions or other MCPs.
- Claude Code: headless Haiku attempted with MCP and hooks. OAuth refresh failed
  before inference; no successful Claude outcome is counted.
- Original random-search control: seeds 0–9 on both nulls, five BTC episodes.
  It uses the same backtest-call budget and reports its exact configuration count.

The CLI sessions are real coding-agent processes, but the research space is
constrained: five predefined rule families, two modifiers and at most 30 backtest
tool calls. Sweeps can evaluate up to 50 configurations per call. Scratch code is
allowed; strategy evaluation must use MCP. This is not an arbitrary-program
strategy-discovery benchmark. The live shell hook is a capture smoke check;
scored research evaluations in this experiment come through MCP.

## Data and selection

BTCUSDT daily closes, 2018-01-01 through 2023-12-31, are the research window.
The tools withhold 2024-01-01 through 2026-09-30. This does not exclude
knowledge of public BTC history from model pretraining. The final submitted candidate
alone is evaluated on that held-out window; later reproduction only repeats its
arithmetic. No prompt, strategy, model or acceptance rule is revised from test
performance. There is no separately tuned predictive model. CSCV partitions the
research window; it is not another independent test set. The candidate universe
was discovered using the whole research window, and agent search is not rerun
inside each CSCV fold. PBO therefore describes the logged universe conditionally;
it does not validate end-to-end agent generalisation.

Both synthetic markets have zero conditional mean before costs. Signflip uses
bootstrapped BTC magnitudes and independent symmetric signs; GARCH uses symmetric
Student-t innovations and symmetric clipping. Frozen input bytes avoid numerical
sampler drift across CPU architectures. Null data have 2,190 observations; all
strategies use lagged signals and five basis points per unit of turnover.

## Counts, failures and acceptance

Ledger N is the number of distinct return-series hashes within an episode.
Repeated or equivalent evaluations count once for N, while the ledger retains
every evaluation record. Claimed N is the agent's positive integer self-report.
The running ledger counter is hidden; individual sweep sizes remain visible.
Count-accounting summaries retain every positive claim, including invalid final
strategies; missing claims are not treated as accurate claims.

Invalid submissions, timeouts and provider failures are retained, never replaced
with the best ledger trial or a different seed. Startup/authentication pilots are
retained under `results/attempts/`. Primary acceptance rates describe valid picks
from completed episodes; planned attempts and statuses are reported alongside
the denominator. A failure is not a successful strategy or a DSR rejection.

Prespecified acceptance is strictly `PSR(SR > 0) > 0.95` or `DSR > 0.95`.
DSR uses logged cross-sectional Sharpe variance, skewness and kurtosis, once with
claimed N and once with ledger N. Each search includes all its logged trials;
there is no best-seed selection. All seeds and engineering iterations, including
failed ones, remain in the evidence. Engineering/profile changes are not trading
strategy candidates and do not contribute a financial significance claim.

## Uncertainty and the paired question

Wilson 95% intervals describe episode proportions. NIM models reuse market seeds,
so pooled episodes are not claimed to be iid: a market/seed cluster bootstrap
(5,000 replicates, seed 20261029) is also reported. The target half-width applies
to each cohort's headline rate, not each model's smaller subsample.

For each model, exact two-sided McNemar tests compare claimed-N and ledger-N
decisions at 0.95. Wilcoxon signed-rank tests compare continuous paired scores,
with Holm correction across models; their interpretation assumes symmetric
paired differences and independent market draws. Zero discordance does not prove
equivalence. Wilson intervals on discordance show the finite-sample limitation.

Thresholds 0.5, 0.8, 0.9, 0.975 and 0.99, observed score-boundary intervals, and
claimed-N fractions 0.5, 0.25, 0.1 and 0.02 are **exploratory sensitivity analyses**.
Counterfactual counts are not observed agent self-reports, and do not change the
headline threshold. These analyses explain when N can change a decision without
claiming that those conditions occurred in the primary experiment.

## Reproduction and provenance

`python -m experiments.run reproduce` replays NIM responses from committed cache,
audits CLI ledgers and re-evaluates every logged spec against its return hash.
CLI inference is not replayed offline. Compressed content-addressed return objects
also support `trial-ledger pbo` directly, without regenerating markets.

`results/episodes.json` preserves full precision. `results/provenance.json` maps
artifact checksums, commands, configurations, source commits/hashes and episode
receipts. Hosted model weight revisions were not pinned or recorded; model-card revisions in
the licence audit are not deployment revisions. Original NIM receipts lacked an
exact launch command; caches and original committed ledgers are retained, and this
limitation is stated rather than reconstructed as a known fact. Workspace hashes
in old long-running batch receipts describe the snapshot at receipt creation;
the loaded CLI harness hash is recorded separately when available.
