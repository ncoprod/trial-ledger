# Trial-ledger improvement log

## Targets and stopping rule

- GitHub Actions green on the final pushed main commit.
- At least 40 completed real coding-agent episodes, each with an externally stored,
  verified MCP ledger. Codex CLI gpt-6.1-sol, medium reasoning; Claude Code Haiku if
  authentication works. Failed attempts are retained separately and never counted as successes.
- Headline null-market naive and ledger-N DSR acceptance: Wilson 95% interval
  half-width <= 0.10. Retain all seeds and distinguish CLI subjects from NIM tool loops.
- Paired tests of claimed-N versus ledger-N DSR, including exact McNemar decisions
  at the prespecified 0.95 threshold. Boundary sensitivity is exploratory, not a
  new headline threshold. Report negative and mixed results.
- Every scored trial in committed experiment ledgers has a compressed return series;
  CLI PBO and integrity verification run without regenerating market data.
- Rust benchmark rerun against fixed inputs with identical-output checks; Linux
  x86_64 and macOS arm64 wheels built and tested by CI, never published to PyPI.
- README headline/provenance link to committed results with exact command, config,
  seeds, model revision (or explicit unknown hosted revision), and source commit.

Loop: run -> measure -> diagnose -> one focused change -> rerun. Stop when targets
are met, or after two consecutive focused iterations without real improvement and
an evidence-backed explanation of a remaining structural gap. Statistical targets
are evidence targets, never permission to tune seed selection or acceptance rates.
The originally designated BTC period was exposed before the independent review,
and future-dependent sizing invalidated its clean-holdout interpretation. The
corrected study therefore labels it a previously exposed diagnostic. No strategy
or experiment-protocol tuning uses diagnostic returns. Reproduction repeats
arithmetic on fixed candidates, rather than selecting new ones. Earlier entries
retain the terminology and outcomes of their historical evaluation version.

## Iteration 0 — baseline

- Hypothesis: cache replay is portable as documented.
- Command: `gh run view 40818134945 --log-failed`; read README, RESULTS and git log.
- Result: FAIL, Linux replay 60/100 identical, 40 GARCH episodes differ. Local suite
  originally 98 pytest / 3 cargo tests. Source: Actions run 40818134945; committed
  `experiments/results/RESULTS.md` at 4162d54.

## Iteration 1 — freeze the fitted calibration

- Hypothesis: cross-platform numerical optimizer drift causes the GARCH mismatch.
- Change: freeze original parameters with BTC checksum; add market fingerprint test.
- Command: `uv run pytest`; `uv run python -m experiments.run replay-check`.
- Result: local 99 tests pass, 100/100 episodes replay. Linux fingerprint still
  fails: fit freezing alone is insufficient; Student-t sampling also uses libm.
- Source: commit 0d4a761; Actions run 40847188920 (FAIL).

## Iteration 2 — freeze generated GARCH inputs

- Hypothesis: preserving input bytes removes sampler/libm drift without weakening
  hashes or changing any original episode.
- Change: commit compressed original seed 0-19 GARCH series alongside calibration.
- Command: `uv run python -m experiments.run replay-check`; GitHub CI.
- Result: local original replay 100/100 identical; Linux result pending.
- Source: commit 95051f7; `experiments/data/garch_markets.npz`, Actions 40847620374.

## Iteration 3 — CLI capture pilots

- Hypothesis: isolated CLI subjects can use the same MCP search protocol.
- Change: add restricted research MCP catalog, server-side 30-call budget, isolated
  temporary workspace/configuration, existing CLI sign-in, shell capture hook.
- Command: `uv run python -m experiments.cli_agents --agent codex --seeds 0 --markets signflip --reps 1 --timeout 900`
  and the same command with `--agent claude`.
- Result: both FAIL before trials. Codex MCP startup failed because NumPy scalar
  repr was written into CSV; fix by serializing native Python float. Claude's OAuth
  session expired and cannot refresh; skip further Claude episodes as authorized.
- Source: `experiments/results/attempts/`; raw diagnostics in ignored `runs/`.

## Iteration 4 — CLI MCP approvals

- Hypothesis: after fixing CSV scalar serialization the CLI can evaluate trials.
- Result: FAIL, hook smoke succeeded but Codex's noninteractive approval policy
  denied the MCP call. The empty strategy is invalid and excluded from episodes.
- Change: explicitly allow this experiment's three configured MCP tools; validate
  submitted strategy specs before accepting an episode.
- Command: repeat the Codex pilot command from iteration 3.
- Source: `experiments/results/attempts/codex-cli__gpt-6.1-sol/pilot-2/`.

Iteration 2 Linux outcome: PASS, all CI jobs green, Actions 40847620374 at 95051f7.

## Iteration 5 — return-store portability

- Hypothesis: one compressed shared content-addressed store avoids duplicated series
  while preserving existing hash-chain anchors and supporting offline CLI PBO.
- Initial result: FAIL on case-insensitive macOS: `RETURNS` collided with `returns/`.
- Change: use the distinct `RETURNS_STORE` pointer, keep legacy `.npy` reads,
  verify decompressed content hashes, use unique atomic temporary object filenames.
- Command: `uv run --no-sync pytest tests/test_ledger.py tests/test_service_mcp.py -x`;
  `uv run --no-sync python -m experiments.store_returns`.
- Result: 20 focused tests PASS; historical hydration running. Source:
  `experiments/results/returns_integrity.json` once hydration completes.

## Iteration 6 — rayon scratch-allocation granularity

- Hypothesis: fewer rayon leaves reduce per-task allocations and scheduling cost.
- Change: minimum split chunk size derived from split count and worker count.
- Command: `RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 uv run --no-sync python -m benchmarks.profile_core --output benchmarks/results/kernel_before.json`;
  rerun with `--compare benchmarks/results/kernel_before.json --output benchmarks/results/kernel_chunked.json` after rebuilding.
- Result: all three output arrays byte-identical on every fixed input. Mixed timing:
  small 1000x50 case median 0.106 -> 0.076 ms; largest case 87.54 -> 87.97 ms.
  No broad speedup claim. Three cargo and 15 targeted Python tests pass.
- Source: `benchmarks/results/kernel_before.json`, `kernel_chunked.json` (25 samples/case).

## Iteration 7 — fuse OOS scoring and ranking

- Hypothesis: removing one scratch vector and its memory pass improves larger cases.
- Change: compute the selected OOS score, then rank directly while scoring columns.
- Command: same fixed-input profile, `--output benchmarks/results/kernel_fused_rank.json`.
- Result: pending; reject this candidate if the large cases regress without benefit.
- Source: `benchmarks/results/kernel_fused_rank.json`.

Iteration 7 result: PASS on identical-output checks. Largest fixed-input median
87.54 -> 55.58 ms (two rayon threads); 2190x500: 0.673 -> 0.572 ms. Other cases mixed.
All five cases retain byte-identical logits, IS-best and OOS-best arrays. Keep this
candidate and rerun the full pypbo/NumPy/Rust comparison before updating headlines.

## Iteration 8 — paired N tests and live subjects

- Hypothesis: more episodes tighten intervals; accurate self-reports can leave the
  0.95 decisions unchanged even when the underlying DSR values differ.
- Change: expand NIM to seeds 0-19; run Codex CLI on the same two market types;
  add exact McNemar, Wilcoxon with Holm correction across models, Wilson intervals,
  exploratory threshold boundaries and explicitly hypothetical undercount scenarios.
- Commands: `python -m experiments.run agents --seeds 10-19 --markets signflip garch --btc-reps 0 --workers 1 --resume`;
  `python -m experiments.cli_agents --agent codex --seeds 0-19 --markets signflip garch --reps 1`.
- Result: running. NIM waits for and holds one shared mkdir slot, serial requests;
  all cached responses retained. Full local Python suite: 104 PASS.
- Source: committed per-episode receipts, `episodes.json`, `paired_tests.json` once
  regenerated. Models' hosted weight revisions are unavailable, explicitly recorded.

## Iteration 9 — committed-ledger PBO and licensing

- Hypothesis: hydration can close the missing-series gap without changing original
  trial counts, submissions, record bytes or anchors.
- Command: `python -m experiments.store_returns`; `trial-ledger --dir experiments/results/runs/openai__gpt-oss-20b/signflip-s0-r0/ledger pbo`.
- Result: initial snapshot restores 14,766 unique compressed objects (119,576,623
  bytes) for 17,752 distinct per-ledger series across 101 ledgers. Example CLI PBO:
  0.43954933954933956 over 172 trials, S=16; all required return hashes verified.
- Source: `experiments/results/returns_integrity.json` (final rerun will cover new episodes).
- Change: prevent a misleading partial-universe PBO when a return object is missing;
  scope ignored logs to root `/runs/`, retaining research `experiments/results/runs/`.
- Licensing: verify Hugging Face cards and upstream licences. Binance dataset terms
  explicitly impose CC BY-NC-SA 4.0; retain attributed data as authorized and flag
  non-commercial data separately from MIT code.

## Iteration 10 — invalid submissions and dependence-aware reporting

- Hypothesis: substituting the best logged strategy for an invalid final answer would
  overstate successful agent completion and could alter acceptance rates.
- Change: retain raw invalid submissions, score only valid submitted strategies;
  add a regression test. Add Wilson bars and market/seed cluster-bootstrap intervals
  (5,000 replicates, seed 20261029) alongside descriptive pooled Wilson intervals.
- Command: `uv run --no-sync pytest tests/test_cli_hooks.py tests/test_experiment.py tests/test_paired_tests.py`;
  `CARGO_BUILD_JOBS=2 cargo test --manifest-path rust/Cargo.toml --release`.
- Result: 41 Python tests and 3 cargo tests PASS. A missing stored trial now has a
  regression proving PBO refuses a partial trial universe.
- Source: `tests/test_experiment.py`, `tests/test_cli_hooks.py`, `paired_tests.json`
  after final regeneration. No changes to the prespecified 0.95 threshold or seed set.

CI milestone: Actions 40851148640 at 1da41f2 PASS, including both wheel builds and
committed-return PBO smoke test. Full benchmark is running under `runs/` with a
1.5 GB pypbo matrix cap; this non-ML job is below the shared >4 GB lock threshold.
The earlier wrapper waited unnecessarily for the heavy-ML lock and was stopped
before acquiring it; no other repo's lock was removed.

## Iteration 11 — robust benchmark measurement and fixed CLI sample plan

- Hypothesis: a single best timing on a shared machine is too sensitive to contention
  to establish a new speedup range. The first full rerun passes every equality check,
  but its pypbo timings varied strongly relative to the historical run.
- Change: report medians and retain every timing sample: 3 pypbo/generic, 7 NumPy
  block-statistics, 25 Rust samples. No numerical-kernel change from iteration 7.
- Commands: `RAYON_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --no-sync python -m benchmarks.bench_pbo`;
  `uv run --no-sync python -m experiments.freeze_markets --count 30`.
- Result: first full rerun PASS, retained in
  `benchmarks/results/pbo_benchmark_concurrent.json`; median rerun pending in
  `benchmarks/results/pbo_benchmark.json`. The pypbo case requiring approximately
  224 GB remains explicitly skipped, never extrapolated as a measured speedup.
- Fixed CLI sample extension: seeds 0-29 on both null markets (60 episodes), then
  five BTC runs (seeds 0-4). This plan is fixed before final cohort analysis and
  retains invalid submissions. GARCH inputs 0-19 stay byte-identical; 20-29 are
  archived before use. No strategy/prompt tuning or held-out BTC feedback.

## Iteration 12 — audit edge cases and timing-test diagnosis

- Hypothesis: scored hook trials without a return hash can still produce a misleading
  partial PBO, even after missing-object detection was added.
- Change: treat a missing hash as a missing series; exercise both cases. Use CPU time
  for the existing 100 ms parser-work guard so scheduler contention does not masquerade
  as parser complexity. Preserve the same time budget and pathological input cases.
- Command: `uv run --no-sync pytest`; then `uv run --no-sync pytest tests/test_cli_hooks.py`.
- Result: initial full run FAIL (104 pass, two wall-clock guards fail under concurrent
  benchmark load); focused rerun PASS (34 cases). This was a measurement failure, not
  a removed algorithmic performance requirement.
- Change: provenance now captures `sys.orig_argv`, preserving Python's `-m` invocation;
  earlier snapshots omitted that flag. Fixed launch commands remain in this log.
- Source: `tests/test_cli_hooks.py`, `experiments/PROTOCOL.md`; final suite rerun pending.

## Iteration 13 — resumable NIM scheduling

- Hypothesis: a queued thread pool is unnecessary for a one-request slot and makes
  termination wait for queued episodes instead of promptly releasing the mutex.
- Change: remove the batch thread pool; run serially in the main thread. A SIGTERM
  handler releases the owned mkdir slot in `finally`; an owner PID is recorded.
  Resume preserves completed failure receipts as well as successful episodes.
- Command: `uv run --no-sync pytest tests/test_nim_slot.py tests/test_paired_tests.py`.
- Result: 3 PASS, including simulated SIGTERM, handler restoration and proof that
  another occupied slot remains untouched. Existing cached prefixes will be reused.
- Source: `tests/test_nim_slot.py`; model batches use disjoint cache files and seed
  sets, with one worker and one atomic slot each (at most two machine-wide).

Iteration 11 median rerun outcome: PASS, all Rust thread outputs byte-identical;
NumPy/pypbo PBO and logit checks pass. Measured Rust/pypbo median range 95.07–1217.52x,
versus historical best-run range 160–1463x. This is mixed performance evidence,
not a uniformly faster end-to-end claim: measurement methods and shared load differ.
All timing samples remain in `benchmarks/results/pbo_benchmark.json`.

## Iteration 14 — interleaved native-build comparison

- Hypothesis: boolean rank reductions simplify the hot loop and can avoid branch
  costs without fast-math or altered floating-point accumulation order.
- Change: replace the OOS less/equal branch with two boolean increments (fewer lines).
  Build original (4162d54), fused (59a2d766), and branchless variants with the same
  compiler command; retain native-binary and Rust-source checksums per build.
- Commands: `CARGO_BUILD_JOBS=2 uv pip install --reinstall --no-deps ./rust` per
  variant; `RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 uv run --no-sync python -m benchmarks.compare_kernels runs/native-variants --output benchmarks/results/kernel_interleaved.json`.
- Result: all three output arrays byte-identical across all builds on every case.
  Randomized interleaving, 25 samples/build/case: largest original median 161.44 ms,
  fused 136.15 ms, branchless 115.30 ms (1.40024x original). The 2190x500 case regresses
  versus original (3.13 -> 3.62 ms), so this is not a universal speedup claim.
  Keep the simpler branchless candidate; full comparison rerun pending.
- Source: `benchmarks/results/kernel_interleaved.json`, raw standalone recheck and
  branchless profiles. Eight Rust parity tests and three cargo tests PASS.
- Provenance correction: old profile helper hashed the package's `__init__.py`, not
  its native binary. Relabel that field honestly; interleaved builds record actual
  native checksums. Older source/input/output hashes remain unchanged.

Iteration 13 restart outcome: cached prefix resumed with exactly the same 359
records, one additional API call, no extra independent selection. Checkpoint retained
under `experiments/results/attempts/nim-resume/`. An initial maintenance invocation
used system Python and failed to import the package; rerunning with `uv run` fixed
that before resuming. No cache or ledger data were lost.

Verification milestone: full suite 108 PASS in 8.99 s; Actions 40853488417 PASS.

## Iteration 15 — remove verified duplicate return objects

- Hypothesis: CLI-local return objects remain duplicated after migration to the
  shared content-addressed store, adding repository bytes without audit value.
- Change: copy verified local objects to the shared store when available; regenerate
  only historical missing series. Remove a local duplicate only after its shared
  object verifies and the anchored ledger verifies with all required series.
- Command: `uv run --no-sync pytest tests/test_experiment.py`;
  `uv run --no-sync python -m experiments.store_returns` after committing the source.
- Result: 7 focused tests PASS; new regression preserves exact ledger bytes, head
  anchor and decompressed return hash while removing the duplicate local object.
  Full migration pending; source: `experiments/results/returns_integrity.json`.

## Iteration 16 — isolate performance measurement from shared-machine load

- Hypothesis: the shared M1 cannot currently give a stable end-to-end latency ratio.
  Final candidate's full checks pass, but pypbo's 1000x10/S16 median rose to 72.10 s,
  versus 1.98 s historically. The resulting 46,273x ratio is not a credible portable
  speedup claim. Memory pressure is suspected, not established by per-run telemetry.
- Change: add a manually triggered isolated Ubuntu benchmark workflow. It builds
  all three recorded source revisions with the same compiler, interleaves the fixed
  cases, then measures pypbo/NumPy/Rust with all samples and equality checks retained.
  This is a measurement change, not a strategy/kernel tuning iteration.
- Command: `gh workflow run benchmark.yml --ref main`; retrieve its result artifact.
- Result: PASS on isolated Linux, all five NumPy/PBO and thread equality checks.
  Four feasible pypbo comparisons give 182.17–1135.50x median speedups. Original
  versus final Rust medians improve on every interleaved case (1.09–1.17x); largest
  219.863 -> 189.664 ms. The largest pypbo matrix is skipped by the prespecified
  memory cap. Keep all shared-M1 results, including regressions and inflated ratios.
  This is a new hardware/method baseline, not a comparison with the old best-of-M1
  160–1463x headline. Source: `benchmarks/results/pbo_benchmark.json`; isolated artifacts
  will be stored under `benchmarks/results/linux/` with the Actions run reference.
- Reproduce native variants: `uv run --no-sync python -m benchmarks.build_variants runs/native-variants`.

## Iteration 17 — retain invalid NIM submissions in the enlarged analysis

- Hypothesis: a completed API exchange can still contain an invalid final strategy.
- Change: derive `invalid_submission` status during analysis, preserve its ledger
  and claimed count, and exclude unscored picks from score sensitivity calculations.
  Do not replace the final answer with a successful strategy from the search.
- Command: `uv run --no-sync python -m experiments.run analyze`.
- Result: first enlarged analysis failed with `KeyError: selected_moments` for
  gpt-oss-20b/signflip/seed 11. The receipt remains unchanged. Both invalid-status
  paths and paired analysis now pass regression tests (10 focused tests). Re-run
  the full analysis from the committed fix.
- Result files: `experiments/results/episodes.json`, `paired_tests.json`; failed
  command log: ignored `runs/cli-cohort-analysis.log`. This is an analysis repair,
  with no new strategy selection or held-out test evaluation.

## Iteration 18 — completed Codex cohort and isolated performance evidence

- Hypothesis: extending the predeclared CLI sample improves precision without
  changing the prompt, strategy families or acceptance threshold.
- Change: finish seeds 20–29 on both null markets and five BTC repetitions,
  following the initial seeds 0–19; do not retry invalid submissions.
- Command: `uv run --no-sync python -m experiments.cli_agents --agent codex --seeds 20-29 --markets signflip garch --reps 1`;
  `uv run --no-sync python -m experiments.cli_agents --agent codex --seeds 0-4 --markets btc --reps 1`.
- Result: 65 receipts, including 60 null attempts (58 valid) and five BTC attempts.
  Naive acceptance is 49/58, ledger and claimed DSR each 0/58. Wilson half-widths
  are 9.275 and 3.106 percentage points, satisfying the target. Two invalid null
  answers remain in the denominator of attempts, not substituted picks.
- Result files: `experiments/results/runs/codex-cli__gpt-6.1-sol/`,
  `experiments/results/paired_tests.json` and `episodes.json` after analysis.
  NIM extension is still running; final aggregation will retain every attempt.

### Completed-data milestone

- All 65 Codex ledgers verified against external receipt anchors and stored return
  bytes. Staging snapshot also includes completed NIM extensions, never an in-flight
  episode. The early signflip/seed 0 pilot's missing tool-call count was recovered
  from its retained `calls.jsonl` (30 research calls); historical launch/source
  metadata remain explicitly unknown. No inference was rerun.
- The acceptance figure had overlapping model labels and a legend over an interval.
  Wrap long labels, move the legend outside the plot, then inspect the exported PNG.
  Values and intervals are unchanged. Full local suite: 110 passed; ruff clean.
- A provenance wording edit exceeded the lint line limit in 09dfaf00; 04f9d05a
  restores lint. The failed CI run is retained (40858774253).

### Interim documentation and analysis snapshot

- Command: `uv run --no-sync python -m experiments.run analyze`;
  `uv run --no-sync python -m experiments.run figures`.
- Result: 176 analyzed receipts, 65 Codex episodes included. The NIM extension
  is incomplete and explicitly labelled interim in README. Full-precision scores,
  Wilson/cluster intervals and paired/counterfactual results are now committed.
- Metadata: the early CLI pilot lacks historical source/launch provenance; its
  ledger, submission, CLI version and call trace remain available. No unknown
  source or hosted weight revision is invented.
- Verification: 110 pytest and three cargo tests pass. The 941d2e49 milestone
  passes all seven GitHub CI jobs (40859304489). Exported figures visually checked;
  local README links resolve. Final NIM aggregation and final-main CI remain open.

## Iteration 19 — clarify boundary uncertainty and missing deployment metadata

- Hypothesis: a zero-pass percentile-bootstrap interval could be mistaken for
  proof of zero risk, despite the nonzero Wilson upper bound.
- Change: explicitly document the bootstrap degeneracy; retain Wilson bounds.
  Record the unchanged Sharpe search target in the generation manifest and call
  hosted revisions unrecorded rather than implying a verified service capability.
- Command: `uv run --no-sync ruff check .`; `uv run --no-sync ruff format --check .`.
- Result: PASS. This changes explanatory metadata, not scores, thresholds,
  prompts, candidates or held-out evaluations. Final paired output will include
  the caveat. Source: `experiments/paired_tests.py`, `provenance.py`, README.

## Iteration 20 — resume after the machine reboot

- Cause: the machine reboot killed background jobs; completed Codex and benchmark
  work was already committed and remains unchanged. Stale mutexes were cleared
  by the owner; new batches acquire the shared slots normally.
- Check: `git status`, cached JSONL parsing and episode receipts show 15/20 new
  gpt-oss episodes complete; Lightning/Ultra extensions had not started.
- Change: preserve the interrupted gpt-oss GARCH seed-17 ledger (169 records) under
  `experiments/results/attempts/reboot-checkpoints/garch-s17-r0/`, then resume all
  three model batches with `--seeds 10-19 --markets signflip garch --btc-reps 0
  --workers 1 --resume` and their original `--models` aliases. Logs and exit codes
  are recorded in ignored `runs/nim-*-reboot.*`.
- Result: cache JSONL valid; completed receipts are skipped. The incomplete prefix
  is replayed from cache, not treated as a new independent selection. Verify its
  strategy/return fingerprints against the completed episode before final analysis.
- Current source: a634544a. No prompt, strategy space, threshold or held-out
  evaluation changes. Final sample remains fixed, with all failures retained.

## Iteration 21 — preserve score decisions through CSV export

- Hypothesis: six-significant-digit CSV formatting can change a strict acceptance
  decision when figures/reporting reload a score close to 0.95. Full-precision
  JSON and primary paired analysis were already unaffected.
- Regression command: `uv run --no-sync pytest tests/test_experiment.py::test_csv_preserves_acceptance_boundary`.
- Initial result: FAIL, 0.950000001 was exported/reloaded as 0.95.
- Change: one-line float formatting change to shortest round-trip representation.
  Display rounding remains in chart/report labels, after decisions are computed.
- Re-run: experiment/paired tests plus ruff. Final artifact regeneration will use
  the corrected CSV precision; no strategy or inference run is repeated.
- Source: `experiments/analyze.py`, `tests/test_experiment.py`; final `episodes.csv`.

Iteration 20 recovery outcome: GARCH seed 17 completed with 270 records; all 169
pre-reboot strategy/return fingerprints are identical. One provider timeout
recovered with backoff. The checkpoint includes the completed receipt reference.
Iteration 21 verification outcome: 11 focused tests PASS; ruff check/format PASS.
Main cfdcaaf1 passes all GitHub CI jobs (40876756870).

## Iteration 22 — retain invalid strategies' usable trial-count claims

- Hypothesis: count accuracy remains measurable even when a final strategy cannot
  be scored. Filtering claims by valid strategy status hides available accounting
  evidence and disagrees with the per-model summary population.
- Initial result: regression FAIL, an invalid strategy with a positive claim was
  counted as zero available claims. The actual NIM signflip seed-11 episode
  claimed 12 against 113 distinct logged return series.
- Change: one-line cohort accounting filter uses all positive claims from
  attempts. Acceptance and paired DSR comparisons still require valid scores.
- Command: `uv run --no-sync pytest tests/test_experiment.py -k invalid_submission --maxfail=1`;
  rerun experiment/paired tests and ruff after the fix.
- Result files: final `paired_tests.json`, `episodes.json`; regression in
  `tests/test_experiment.py`. This retains more failure evidence without
  introducing an accepted/rejected score for an invalid submission.

Iteration 22 verification: 11 focused tests PASS, ruff check/format PASS. The
trial-count figure title also identifies its actual null-plus-BTC population;
acceptance figures remain restricted to valid null-market picks.

Iteration 22 interim refresh: `python -m experiments.paired_tests` on the existing
176-row snapshot gives 25/71 NIM under-reports and 0/60 Codex under-reports when
all usable claims are retained. Acceptance denominators remain 70 and 58.
The refreshed manifest references committed/staged episode receipts, verified
against index blob hashes; the newly completed signflip/seed-18 receipt and its
returns are included. Full local suite: 111 passed in 19.32 s
(`runs/pytest-post-reboot.log`, ignored).

Iteration 22 reporting clarification: label the older pooled paragraph as valid
picks, and show the usable-claim denominator beside cohort under-report counts.
This makes the two populations explicit without changing any score or test.

Iteration 20 model milestone: the resumed gpt-oss batch exits 0 after all five
remaining episodes. The full model now has the prespecified 40 null episodes
(seeds 0-19 on both markets) plus five BTC episodes. The 359-record maintenance
checkpoint from the earlier serial-runner fix also matches the completed
signflip seed-12 trial fingerprints. These are replayed prefixes, not new
independent selections. Lightning and Ultra remain queued for shared NIM slots.

Gpt-oss-complete interim refresh: `uv run --no-sync python -m experiments.store_returns`
then `uv run --no-sync python -m experiments.run reproduce` both exit 0. All 185
receipts replay/audit without mismatches; 26,157 distinct per-ledger return series
verify. The NIM snapshot has 80 attempts, 78 valid picks, 68 naive passes, one
pass under each DSR, and 32/80 under-reports. Both invalid gpt-oss answers
(signflip seeds 11 and 16) retain their claims. The other 40 planned NIM episodes
are queued, so README remains explicitly interim. Sources: `episodes.json`,
`paired_tests.json`, `returns_integrity.json`, `provenance.json`; all artifact
and episode hashes verified, all three exported figures inspected.

## Iteration 23 — reduce idle slot-acquisition delay

- Hypothesis: ten-second mutex polling misses short releases when other repositories
  poll more frequently. Two slot-2 ownership changes occurred while both trial-ledger
  extension batches stayed queued. This is scheduling latency, not an API failure.
- Change: one-second polling of the same two atomic mkdir slots. Stop only the two
  idle queued clients and resume their unchanged commands; no API request or episode
  had started in either batch. Keep one serial client per acquired slot.
- Command: `uv run --no-sync pytest tests/test_nim_slot.py`; rerun both pending model
  batches with the original seeds 10-19, two null markets, one worker and `--resume`.
- Result: pending acquisition; logs in ignored `runs/nim-*-queue.log`, with exact
  launch receipts and exit files. All research settings and completed outcomes remain
  unchanged. Source: `experiments/llm.py`; final receipts will record the source hash.

Iteration 22 denominator follow-up: the per-model null table now shows attempts
with the valid-pick count in parentheses, matching the cohort table. For example,
gpt-oss has 40 attempts and 38 scored picks in the current snapshot. Generated
artifacts will refresh from this display-only change with the final analysis.

Iteration 23 outcome: one-second polling did not produce an acquisition before
both slots transferred to other repositories again. No request or episode started
in either pending model; their idle clients exited 143 for the next queue restart.

## Iteration 24 — final scheduling-latency check

- Hypothesis: a 0.1-second interval reduces missed short release windows while
  preserving the same atomic mutex and one serial requester per slot.
- Change: only the idle wait constant; restart the two still-unstarted batches.
- Command: `uv run --no-sync pytest tests/test_nim_slot.py`; the unchanged pending
  model commands are recorded in `runs/nim-*-fair-launch.json` with logs/exit files.
- Result: pending. This is not a prompt, seed, threshold or strategy change, and
  no timing improvement is claimed merely from changing the constant.
- Source: `experiments/llm.py`; final completed model receipts/results.

## Iteration 25 — uncertainty for headline count accuracy

- Hypothesis: zero observed under-reports can be misread as a population rate of
  zero when only an observed count is displayed.
- Change: reuse the existing Wilson method for each cohort's under-report share
  among all usable claims, including claims from invalid final strategies. Show
  these intervals beside the counts in README and the generated results table.
- Command: `uv run --no-sync pytest tests/test_experiment.py tests/test_paired_tests.py`;
  final `python -m experiments.run reproduce` regenerates all metric artifacts.
- Result file: `experiments/results/paired_tests.json` after final regeneration.
  No inference, candidate, selection threshold or sample-plan changes. The same
  shared-market dependence caveat applies to these descriptive intervals.

Iteration 25 focused verification: 11 tests PASS; ruff check/format PASS. The
report renderer derives its displayed bounds from stored counts with the same
Wilson function, so existing result snapshots remain readable before regeneration.

Iteration 24 acquisition outcome: Lightning acquired slot 2, then Ultra acquired
slot 1. Both clients remain serial; no third slot or request was introduced.
The observed acquisition does not establish a causal wall-clock speedup from
the poll change because release timing was controlled by the other repositories.
All stopped predecessor clients had acquired no slot and issued no requests.

Lightning extension milestone: all 20 new planned episodes completed; the batch
exits 0. Transient connection/time-out errors recovered with backoff. All 45
Lightning receipts (40 null plus five BTC) verify against their anchored heads
and stored returns. The cache and complete receipts are checkpointed without
staging the active Ultra batch. Source: `experiments/results/runs/nvidia__nemotron-3.5-lightning-30b-a3b/`.

## Fixed-sample completion

- Ultra finished its remaining 20 episodes and exits 0. Connection errors,
  timeouts and HTTP 500s recovered with backoff; no terminal API-error receipt
  was produced in either pending model batch.
- Exact planned set verified: 225 receipts, comprising 45 per NIM model
  (40 null plus five BTC), 65 Codex, and 25 random controls. No seed substituted.
- All 45 Ultra anchored ledgers and their stored returns verify before staging.
  Source: `experiments/results/runs/nvidia__nemotron-3-ultra-550b-a55b/`.
- Commands unchanged from the fixed plan; launch/exit logs are in ignored
  `runs/nim-*-fair.*`. Final aggregate analysis and full return-store audit follow
  from the complete raw-data checkpoint.

## Final verification and stopping decision

Source checkpoint: `7fd10ce8cce0fd7f9da95e6cfcb221931106d4f7`. Commands:
`uv run --no-sync python -m experiments.store_returns`, then
`uv run --no-sync python -m experiments.run reproduce`, with RAYON_NUM_THREADS=2,
OPENBLAS_NUM_THREADS=1 and VECLIB_MAXIMUM_THREADS=1. Both exit 0.

- Exact fixed set: 225 receipts; 221 valid final specifications, four invalid
  submissions retained (two Codex, two gpt-oss). One valid legacy Lightning
  final specification is outside its recorded search; this flag is retained,
  and analysis evaluates the submitted spec rather than substituting a ledger pick.
- Reproduction: 160 cache/baseline replays plus 65 CLI anchored-ledger audits;
  zero mismatches. All 225 analyzed ledgers verify, with zero trial-return
  re-evaluation mismatches. Every final artifact/episode provenance hash matches.
- Return store: 225 ledgers, 35,108 distinct per-ledger trial series, 28,613 shared
  compressed objects, 234,188,281 bytes. No missing object or noncanonical temporary
  compressed object. Source: `experiments/results/returns_integrity.json`.
- CSV exports retain the existing standard CRLF dialect. Plain `git diff --check`
  flags those line endings; the check with `cr-at-eol` recognition passes.
- All scored CSV values round-trip exactly to JSON values, including the strict
  acceptance decision. A verification helper initially indexed missing scores
  on invalid rows and raised KeyError; using the existing missing-score convention
  corrected the helper, then the audit passed. No result was changed by that check.
- Full native CI suite: 111 pytest and three cargo tests PASS at 7fd10ce8
  (Actions 40930159172), versus the original 98 pytest / three cargo baseline.
- All three final exported figures inspected; README local links resolve. Every
  README empirical number links to its result artifact.

| Target | Verified outcome | Evidence |
|---|---|---|
| CI green | All seven raw-data milestone jobs PASS; final documentation/results commit checked after push | Actions 40930159172 |
| >=40 real coding-agent episodes | 65 Codex attempts, 63 valid; 60 null attempts/58 valid, five BTC; every ledger verified | `experiments/results/episodes.json` |
| Wilson half-width <=10 pp | CLI naive 9.274912821 pp / DSR 3.105999845 pp; NIM naive 5.319482695 pp / DSR 2.247460607 pp | `experiments/results/paired_tests.json` |
| Claimed-N versus ledger-N decisions | Zero discordance at 0.95 for every model; exact McNemar p=1; no NIM continuous-score Holm test rejects at 0.05 | `experiments/results/paired_tests.json` |
| Conditions where N matters | Actual maximum 274/16=17.125x gives claimed DSR 0.7645027071 versus ledger 0.3331662428, both reject; controlled N=7 versus ledger N=30 crosses 0.95 | `experiments/results/episodes.json`, sensitivity in `paired_tests.json`, `tests/test_paired_tests.py` |
| Committed-ledger PBO | Every stored trial return verifies; CLI PBO smoke passes in CI | `experiments/results/returns_integrity.json`, `pbo_cli_example.json` |
| Rust output-checked benchmark | All outputs checked; all three native variants byte-identical; isolated original/final speedups 1.09-1.17x | `benchmarks/results/linux/kernel_interleaved.json` |
| Portable wheels, no PyPI publication | Linux x86_64 and macOS arm64 wheels built, installed, parity-tested and retained as CI artifacts | Actions 40930159172 |

Iteration outcomes that were pending in their original chronological entries:
2: portable GARCH replay PASS; 5/9/15: full storage migration/audit PASS;
7/11/14/16: isolated benchmark PASS, shared-M1 regressions/unstable timings retained;
8/18/20: fixed sample complete with all outcomes retained; 12: full suite/CI PASS;
17: two invalid NIM submissions handled without substitution; 19/21/22/25:
full-precision, dependence-caveated final reporting regenerated and verified.
23: one-second scheduling change did not obtain a slot before further transfers;
24: both batches subsequently acquired slots, without a causal timing claim.

Stopping rule (a): all research, storage, performance, documentation and licensing
targets are met; stop after the final pushed main commit passes Actions.
No prompt, strategy-space, seed or acceptance-threshold tuning used held-out BTC.
The five Codex BTC candidates (three distinct specs) remain fixed; median Sharpe
1.11503256061408 in sample versus 0.5682778598437503 held out, with every pick declining.
Sources: `experiments/results/episodes.json` and README's linked result tables.

The well-powered mixed result remains: NIM naive acceptance is 107/118 versus
1/118 under either DSR; 51/120 claims under-report N, maximum 17.125x. Codex
naive acceptance is 49/58 versus 0/58 under either DSR, with 0/60 under-reports.
Count capture changes scores but adds no primary-threshold decision advantage
over claimed N in this sample. Boundary sensitivity is exploratory, not a new
headline threshold. Zero discordance does not establish equivalence.

Claude Haiku remains skipped after its preserved OAuth refresh failure. Hosted
weight revisions and the earliest CLI pilot's exact launch/source metadata remain
explicitly unknown. Provider connection/time-out/500 errors recovered; no terminal
API-error episode is dropped. Reboot and maintenance prefixes (169 and 359
strategy/return fingerprints) matched their resumed episodes exactly. All owned
NIM slots were released; foreign locks were never removed.


## Independent review remediation (causal-v2)

Review intake at 94196b21: all eight findings accepted. The untracked review is
excluded locally and will never be staged. Previous completion claims do not
survive the newly demonstrated sizing defect. The old artifacts are superseded,
not deleted, relabelled as fresh searches or silently rescored.

Fixed plan before inference: `experiments/PROTOCOL.md`, 265 corrected episodes.
No outcome-based stopping or tuning; no untouched-BTC-test claim. Two evaluation
versions and every logged strategy trial are disclosed; the defective first
version is not a candidate eligible for headline selection.

| Item | Hypothesis and focused change | Verification / outcome | Fix commit |
|---|---|---|---|
| B1 future sizing | Fixed 0.02 per-period target; lagged rolling denominator; fresh searches in causal-v2 | Full positions/returns, research append invariance and OOS-prefix perturbation regressions; initial focused run exposed an unrelated guard-order test failure, corrected before rerun | 5dcf789a |
| M2 purge boundaries | Purge both label-interval boundaries, then embargo after their union | All-pairs interval intersections, zero/short embargo | 5dcf789a |
| M3 collapsed emissions | Deduplicate only proven equal return hashes within a coherent cohort; otherwise each scored evaluation counts | Actual emit/exec regression with changed script content, scores and repeated metadata | 5dcf789a |
| M4 mixed cohorts | Require dataset/time-axis/frequency/length identity; no majority-length selection; explicit legacy alignment assertion remains labelled | Mixed dataset, date axis, frequency and length rejection tests | 5dcf789a |
| M5 failed verification | Abort analysis/public report on chain/replay failure; stop reproduce before any writes | Corrupted score, return hash, anchored tail, and no-publication regressions | 5dcf789a, 21c36bb6, 6509fbc6, 5bf9ceea |
| M6 native licences | Collect locked dependency notices, package wheel/sdist bundles, inspect both CI artifacts | All 59 full notices verified in both platform wheels and Linux source archive; CI 40946558292 PASS | 4c36f6e5, 187ee8ee, 1ddf6ff7 |
| m7 window boundaries | Initial equity included in drawdown; slice actual full-history turnover | First OOS loss and unchanged holdings regression | 5dcf789a |
| m8 unused cache | Delete unused returns_by_seq and its numpy import | Existing MCP/service tests | 5dcf789a |

Core commands: `RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --no-sync pytest tests/test_backtest.py tests/test_stats.py
tests/test_cli_hooks.py tests/test_experiment.py tests/test_service_mcp.py -q`;
`uv run --no-sync ruff check .`; `uv run --no-sync ruff format --check .`.
First result: 105 focused tests pass, one new integrity-guard test fails because
summary grouping preceded its guard. Move the guard before grouping; rerun below.
Raw diagnostics: ignored `runs/review-core-tests.log`.

Core rerun outcome: all 133 tests PASS; ruff check/format PASS. Frozen GARCH
archive extended from 30 to 50 seeds, with all original 30 series hashes unchanged.
No corrected inference or selection has occurred before this protocol snapshot.
Missing-return effective-N estimates also refuse partial-universe estimation.

Baseline milestone: all 25 corrected random-search episodes completed, no skipped
seeds. Their ledgers/return objects are retained separately from the superseded
version. The fresh Claude Haiku pilot again failed authentication before a single
backtest, so its receipt is stored under causal-v2/attempts/. A small runner fix
routes future zero-evaluation auth failures there automatically. Codex and NIM
jobs remain resumable; NIM clients acquire only the shared atomic slots.

M6 implementation: the collector covers all 30 locked native packages, including
full BSD, MIT, Apache/LLVM and Unicode notices. An initial approach expected
.cargo-checksum.json files that this local cache does not contain; switched to
verifying the published .crate archive hash directly against Cargo.lock, then
reading notices from that archive. No unverified licence text was substituted.
Local `uv run --no-sync maturin` failed because the CLI is not installed in the
project environment; use the declared build backend via `uv build rust` instead.
Wheel/sdist inspections and both-platform CI checks follow. Raw local build log:
ignored `runs/review-wheel-build.log`.

M6 local verification: `CARGO_BUILD_JOBS=2 uv build rust --out-dir runs/review-dist`
PASS; `python rust/check_licenses.py runs/review-dist/*.whl
runs/review-dist/*.tar.gz` PASS for 59 full notice files in both the macOS arm64
wheel and source archive. Ruff check/format PASS (52 Python files). Exact CI
checks for the Linux and macOS artifacts are added before upload.

Review CI/push iteration: the first combined upload failed with HTTP 408. Pushing
the smaller core commit with HTTP/1.1 and a bounded request buffer succeeded;
the result checkpoint and licence bundle are retried separately. Core CI
40941478818 passes lint, Python, both Rust jobs and both wheels; PBO fails because
that smaller commit precedes the causal-v2 baseline ledger. The next checkpoint
contains the required ledger and compressed objects; no statistic is weakened.

Paper preparation: eight references verified against Crossref metadata or the
arXiv API. The first GARCH request returned 429, then succeeded on retry. A Holm
identifier returned Crossref 404, so no unverified bibliography entry is added;
the adjustment algorithm is stated explicitly. Sources and failed checks are
retained in `paper/reference_verification.json`.

Scheduling iteration: run the already fixed five BTC Codex sessions in a second,
isolated, serial queue while the 100 null sessions continue. This changes only
scheduling, not prompts, seeds, budgets or analysis. The original wrapper later
skips completed BTC receipts. No BTC metric is inspected to tune a search.
Command: `uv run --no-sync python -m experiments.cli_agents --agent codex
--seeds 0-4 --markets btc --reps 1`; launch/log/exit receipts are under ignored
`runs/v2-codex-btc*`. No NIM slot is used by a Codex CLI process.

Native notice guard: compare the notice manifest's package versions and archive
hashes directly with Cargo.lock, so a dependency update cannot silently bypass
the bundle. Existing wheel and source archive both still PASS the full check.

Paper inputs: `paper/make_tables.py` reads committed Git blobs, refuses the
incomplete fixed sample, and generates all empirical macros, tables, the
acceptance plot and a SHA-256 input/output manifest. The paper distinguishes
observed claims from every exploratory threshold and counterfactual-count case.
`uv run --no-sync ruff check .` and `ruff format --check .` PASS (53 files).
Numerical generation and PDF compilation wait for the complete committed sample;
no provisional or superseded counts are substituted into the draft.

Transport diagnosis: buffered HTTP/1.1, HTTP/2 and a small buffered object upload
all returned HTTP 408; the existing SSH identity has no access to this private
repository. A filtered diagnostic (no authentication headers or data logged)
successfully uploaded a temporary object-staging branch with HTTP/1.1 and
`http.postBuffer=0`. The full original main history is now retried with those
settings. No history was rewritten or force-pushed. The temporary branch is
removed once main is synchronized. Logs are under ignored `runs/review-push*`
and `runs/trace-upload.log`.

Publication follow-up: a failed CLI process could leave a syntactically valid
partial final answer. Preserve its specification validity and claim accounting,
but return an unscored failed row so every acceptance export uses the protocol's
completed-pick denominator. Regression cases cover both timeout and nonzero CLI
exit with a valid rule. `uv run --no-sync pytest tests/test_experiment.py -q`:
16 PASS; lint/format PASS. The corrected manifest explicitly records the protocol
commit, causal sizing rule and previously exposed BTC status. No search or test
outcome guided this reporting guard.

Local committed-ledger smoke command: `RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1 uv run --no-sync trial-ledger --dir
experiments/results/causal-v2/runs/random-search/signflip-s0-r0/ledger pbo`:
PASS, PBO 0.5737373737373738 from 140 trials, S=16, no alignment exception.
Raw output: ignored `runs/review-pbo-smoke.json`. The five corrected Codex BTC
sessions completed successfully; their held-out metrics are not used for tuning.

Review CI iteration: run 40946080229 passes reproduction, lint, Python, both Rust
jobs and the macOS wheel's complete-notice check. Linux source packaging fails
with permission denied: the manylinux container owns rust/dist, while the source
archive step runs on the host. Focused change: write the source archive to a
separate rust/sdist directory, verify wheel and source archives separately, upload
both. Raw failure retained in ignored `runs/review-license-ci-failed.log`.

Transport outcome: all 66 small streamed object batches uploaded. A temporary
merge made the original target reachable, then main advanced to its original
commits with zero objects retransmitted. The temporary branch was deleted;
187ee8ee and 21c36bb6 then pushed normally. Main history was neither rewritten nor
backdated. The single large streamed attempt also failed and is retained in the
ignored log. Request size remains an environment limitation, not a result change.

Review gate verified at 1ddf6ff7: Actions 40946558292, all seven jobs PASS.
Both native builds run 135 pytest tests; both cargo suites run three tests.
Both wheels and the Linux source archive contain all 59 full notice files.
Committed receipt: `experiments/results/causal-v2/review_verification.json`, with
the exact Actions metadata/log commands, source SHA and parsed assertions.
The independent review file remains locally excluded and untracked.

Tectonic bibliography smoke: `tectonic -X compile
runs/paper-bib-smoke/smoke.tex` exits 0. Text extraction confirms all eight
references and author accents resolve. Tectonic reports an internal rerun
consistency warning, stopping after six passes; no unresolved citation remains.
The full empirical paper is not yet built. Raw diagnostics are in ignored
`runs/paper-bib-smoke.log`; no placeholder counts enter the paper.

Corrected raw-evidence checkpoint: 45 successful Codex
CLI sessions, 4 gpt-oss sessions and all 25 fixed random controls
verified against their receipt anchors with return replay. The selected snapshot
contains 5993 distinct per-ledger series backed by 5935 shared objects.
Command: `uv run --no-sync python -m runs.stage_review_raw`; operational receipt
`runs/staged-review-raw.json` is ignored. Only complete episodes and their exact
referenced objects are staged; the live NIM cache is snapshotted at a complete
JSON line without overwriting it. No acceptance estimate is published before
the fixed 265-episode plan completes. Later sessions remain resumable.

Review documentation follow-up: update the shell-emission example with dataset,
scored time-axis and frequency identity. Explain that absent return hashes never
justify deduplication and that the labelled legacy alignment assertion cannot
override known cohort conflicts. The direct PBO smoke command now uses the
corrected committed control ledger. Correct fresh-run seed commands and label
the diagram's BTC period as previously examined. Clarify that missing historical
NIM launch metadata concerns the superseded first version, not newly recorded
receipts. No evaluated strategy or inference setting changes.

Documentation verification: the first lint pass rejected a long provenance string
and spacing in the Python README example; both are corrected. Ruff check/format
then PASS (53 files). The ignored transport helper initially passed an invalid
combined diff-tree flag; split it into the documented flags and resume from the
same base/target. No research receipt or main history changes for transport.

Publication wording iteration: the corrected RESULTS renderer and BTC figure now
explicitly identify the later period as previously examined. The generated report
opens with the sizing correction and distinguishes fresh searches from rescoring.
Only publication labels change; no prompt, inference, selection or backtest changes.
Figures will regenerate once the fixed sample completes.

Paper draft iteration: `paper/main.tex` now contains every requested section,
with established-statistic equations, explicit registration status, corrected
causal sizing, conditional-PBO scope and the exposed-BTC limitation. All measured
headlines use generated macros. Tables, figures and numerical macros remain
unpublished until the complete committed sample passes the generator's exact
265-key check. Added the frozen calibration table and review-CI provenance
inputs to that generator. Gençay's official arXiv abstract independently confirms
the stated restricted-tool/search-count comparison; no unverifiable citation is
added. Source-only draft commit precedes the final generated inputs and PDF.
Publication lint: two initially overlong table-label strings shortened; final
ruff check and format PASS (53 files).

Paper provenance clarification: fresh corrected inference reuses frozen null-market
seeds; it is not wholly new market data or an independent replication of the old
version. The paper states that explicitly. The numerical manifest also records
deterministic-generation settings and marks inference seed/revision as inapplicable.
Two bibliography-only build retries (retaining intermediates and compiling with
existing intermediates) still produce Tectonic 0.15.0's internal .bbl consistency
warning. Both exit 0 and resolve all references; no source-level bibliography defect
was found. Stop changing bibliography inputs to chase that engine warning. Logs:
ignored `runs/paper-bib-smoke-keep.log`, `runs/paper-bib-smoke-existing.log`.

Pushed checkpoints b759ad10 and 53ae244b: the full raw-evidence transport completed
without changing main history; the temporary branch is deleted. Actions
40950449242 at 53ae244b passes all seven jobs, including offline replay and both
platform wheels with full notices. All corrected inference jobs continue.

Failure replay iteration: the gpt-oss GARCH seed-2 request exhausts seven timeout
retries. A new offline fixture reproduces a real publication problem: a provider
RuntimeError becomes CacheMiss offline, despite the same cached evaluations.
Initial regression FAIL (both parameter cases); ignored
`runs/review-prefix-before.log`. Focused change: audit failed API attempts against
the exact transcript, trial fingerprints, tool/turn counts and first missing
response. Preserve the original failed status and exclude it from acceptance; do
not claim to replay a provider outage. Tampered feedback still fails the audit.
Command: `uv run --no-sync pytest tests/test_experiment.py -q`: 18 PASS; ruff
check/format PASS. Full suite and real failed-prefix verification follow. The
fixed sample is unchanged and failed attempts are never replaced.

Failed-prefix verification outcome: full local suite 137 PASS. Real offline check
`RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run
--no-sync python -m experiments.run replay-check`: 225 superseded ledgers integrity
audited; 128 corrected episodes checked (32 cache/control replays, one failed
cache-prefix audit, 95 CLI audits), zero mismatches. Raw output:
ignored `runs/review-prefix-real.log` / `.exit`. The failed gpt-oss GARCH seed-2
receipt retains 12 calls, 159 records, no claim or accepted pick.

Scheduling diagnostic: considered moving the planned Nemotron batch ahead after
the next fresh BTC request also timed out. Briefly stopped the gpt-oss process to
check its unfinished prefix before any handoff. The zero-prefix assertion FAILed:
23 new evaluations already existed, showing request progress had resumed. Sent
SIGCONT immediately, kept the same process/episode and slot, and made no handoff,
restart or trial deletion. No interruption receipt was written because the guard
stopped before that mutation. The fixed sample and settings remain unchanged.

Complete Codex milestone: all 105 fixed sessions finish with status ok, 100 null
sessions and five BTC sessions. Wrapper `runs/v2-codex.exit` is 0. No replacements
or extra seeds. The complete-episode return-store audit verifies 139 receipts
(105 Codex, nine gpt-oss including one provider failure, 25 controls), 9,355
distinct per-ledger trial series backed by 9,246 referenced shared objects.
Commands: `python -m experiments.store_returns`, then
`uv run --no-sync python -m runs.stage_review_raw`; ignored logs
`runs/review-codex-store.log`, `runs/stage-review-codex-complete.log`. The partial
store-count receipt is not published as the final fixed-sample result.

Actions 40953733922 at 6509fbc6 passes all seven jobs: 137 Python tests in each
native build, three Rust tests each, both wheels and the source archive containing
59 full upstream notices. Refresh `causal-v2/review_verification.json` from the
exact saved Actions metadata/log commands. Real failed-prefix coverage enters CI
with this next raw checkpoint; no acceptance aggregation occurs yet.

Complete Codex checkpoint `4a6ba60b` is pushed; Actions 40956120062 passes all
seven jobs. The committed snapshot checks 139 corrected receipts: 33 cache/control
replays, one failed cache-prefix audit and 105 CLI ledger audits, zero mismatches.
Both native suites pass 137 Python and three Rust tests; both wheels and the
source archive contain all 59 upstream notices. Commands and saved log checksum
are recorded in `causal-v2/review_verification.json`. README capture guidance now
clarifies direct `record_trial` versus emitted exec capture to avoid double counting.
No partial acceptance estimates are published.

Paper method audit: explicitly state the 365-period annualization convention
and distinguish descriptive annualized Sharpe from the daily PSR/DSR input.
Checked against `experiments/markets.py`, `run_backtest` and the committed
null-ledger metadata (`periods_per_year=365`, `n_obs=2190`). This is a prose
clarification; no evaluation, prompt, seed or inference setting changes.

Raw NIM midpoint checkpoint: 158 completed receipts (105 Codex, 28 gpt-oss,
25 controls), including the retained provider failure. All 12,196 distinct
per-ledger series verify against their anchored ledgers; 11,844 referenced shared
objects are staged. Command: `uv run --no-sync python -m experiments.store_returns`
(exit 0), then `uv run --no-sync python -m runs.stage_review_raw`. No objects needed
restoration. Partial store-count receipts and acceptance aggregates are excluded.
Transient malformed-response TypeErrors subsequently recover through the unchanged
retry policy; no extra episode or replacement seed is introduced.

Paper reporting iteration: pooled cohort rates can hide model heterogeneity and
leave the random control unassessed. Add a generated acceptance table for every
model and the random-search control, using statistic-specific valid denominators
and the recorded primary threshold. Describe the control's fixed sampling,
rounded-feedback selection and record-count reporting without changing them.
Command: `uv run --no-sync ruff check paper/make_tables.py` and format check:
PASS. Empirical generation remains gated on the full committed sample. Output:
`paper/generated/model_acceptance.tex` after final generation.

Traceability iteration: the 25 corrected control receipts omit launch provenance.
The contemporaneous `runs/v2-baseline-launch.json` records the exact command and
source commit 5dcf789a. Import it as `causal-v2/control_launch.json`; attach
explicitly recovered metadata to each control, including seed/config, no-inference
revision label, committed source-snapshot hashes and the original receipt SHA-256.
Command: `uv run --no-sync python -m runs.add_control_provenance`: 25 augmented.
Original receipt bytes match committed HEAD before augmentation; no trial record,
submission, status or anchor changes. The recovery note distinguishes reconstructed
launch-snapshot hashes from instrumented loaded-code hashes. Future baseline
generation still needs the same automatic provenance field after live NIM batches
finish; its selection algorithm and evaluation remain unchanged.

Reporting edge case: gpt-oss signflip seed 12 submits a valid strategy with N=0.
The fixed positive-integer claim rule already maps this to a missing count: naive
and ledger-N statistics may be scored, while claimed-N DSR is unavailable. Keep
this behavior; add generated missing/nonpositive-claim totals and explicit
statistic-specific denominator wording to the paper and final README renderer.
No claim is recoded as accurate, no hypothetical count is substituted, and no
partial acceptance estimate is published. Output: `paper/generated/submission_notes.tex`
from the complete committed sample.

Missing-claim reporting verification: summary/paired calculations already exclude
unavailable claimed-N DSR correctly. The older figure's single valid-n label and
pooled prose could obscure that smaller denominator. Show exact passes/scored
counts on every bar, show the claimed-N fraction, and name the positive-claim
denominator in count summaries. No statistic or inference behavior changes.
Command: `RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --no-sync pytest`: 137 PASS; ignored `runs/review-count-report-tests.log`.
Ruff check/format PASS for all three changed Python renderers. Final figures are
regenerated only from the full fixed sample.

Startup-failure iteration: a missing response before the first trial leaves no
ledger file, so a genuine zero-trial failure cannot pass the publication audit.
New offline regression `test_startup_api_failure_has_a_verifiable_empty_ledger`
FAILs before the fix (`runs/review-empty-first.log`). Initialize the empty ledger
before requesting model output; keep failed status, zero trials and no scores.
Command: `RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --no-sync pytest`: 138 PASS (`runs/review-empty-fixed.log`). Ruff check PASS;
initial format check flags the new multiline assertion, formatter applied and
format recheck PASS. This test makes no API calls or financial evaluations.
Long-running live NIM processes still execute their launch snapshot; they are not
restarted. Any genuine zero-trial receipt from that snapshot must be explicitly
hydrated against its external genesis anchor, with no invented evaluations.
Backtest feedback, fixed sample and all inference settings remain unchanged.

Control-provenance follow-through: future `run_random_search` receipts now record
the actual command, source/config, market seed, generator seed and a no-inference
revision label automatically. Existing recovered receipts (e3c4edd2) keep their
original generation source and are not relabelled as fresh runs. The docstring now
says record count rather than independent/distinct N. Command:
`uv run --no-sync pytest tests/test_experiment.py -q`: 19 PASS. A fixed seed-0
arithmetic replay verifies the submission and every trial fingerprint unchanged,
and checks the fresh provenance fields; `uv run --no-sync python -m
runs.check_control_provenance`, ignored `runs/review-control-provenance.log`: PASS.
Ruff check/format PASS. The pending automatic-provenance item is closed.

Long-job provenance audit: preserve the two contemporaneous NIM launch receipts
and their original checksums in `causal-v2/nim_launches.json`. Distinguish the
loaded launch source snapshot from workspace file hashes recorded at episode
completion. Command: `uv run --no-sync python -m runs.save_inference_launches`:
PASS; backtester, service and NIM client are byte-identical to 5dcf789a. The only
later agent changes initialize empty ledgers and attach control provenance.
No live batch is restarted and no inference setting or feedback changes.
Latest source checkpoint e2af89ee passes all seven Actions jobs in run 40980106091,
including 138 Python tests in each native build and the distribution notice checks.

Failure-prefix verification after the second retained API failure: command
`RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run
--no-sync python -m experiments.run replay-check` at f608a330: PASS. Audit of
170 completed corrected receipts finds 63 cache/control replays, two exact failed
cache-prefix audits and 105 CLI ledger audits, zero mismatches. Separately checks
all 225 superseded ledger anchors and stored series without rescoring them.
Result: ignored `runs/review-current-replay.log` (exit 0). No API calls or partial
acceptance estimates are made. Add a result-version index while preserving all
original artifact bytes; correct the stopping-rule paragraph's obsolete clean-BTC
wording to the explicitly exposed diagnostic interpretation.

Raw checkpoint after that audit: 170 completed receipts (105 Codex, 40 gpt-oss,
25 controls), 14,132 distinct per-ledger series and 13,698 referenced compressed
objects. Commands: `uv run --no-sync python -m experiments.store_returns`, then
`uv run --no-sync python -m runs.stage_review_raw`: PASS, no restoration needed.
Completed failure receipts and exact cache prefixes are included; active-episode
ledgers, partial store-count receipts and acceptance aggregates are excluded.
Logs: ignored `runs/review-current-store.log` and `runs/stage-review-current.log`.

Operational iteration: the combined Nemotron queue waits without requests and
would later serialize both models even if both shared slots became available.
Cancel only that unstarted process after checking its log, lack of model run
directories and lack of slot ownership. Requeue Lightning and Ultra as separate
serial `--workers 1 --resume` model batches with the same fixed seeds, tool budget,
prompt and inference settings. Original supervisor records termination; no episode
or financial trial is restarted or added. Commands and timestamps are preserved
in `causal-v2/nim_launches.json`, including explicit cancellation before zero
requests. The new queues load 4ed35896 (including empty-ledger initialization);
backtester, service and client remain byte-identical to the frozen protocol.
Verification command: `uv run --no-sync python -m runs.save_inference_launches`.
API concurrency remains at most two globally, one serial requester per held slot.

Paper reporting iteration: a continuous-score test needs an effect size alongside
its p-value. Generate each model's median claimed-minus-ledger DSR difference
directly from the committed paired result, with an explicit sign convention and
missing-value marker. No score, test or inference protocol changes. Commands:
`uv run --no-sync ruff check paper/make_tables.py` and `ruff format --check`:
PASS. Final table generation remains gated on the complete committed sample.

Bibliographic follow-up: the official JSTOR issue contents confirm Holm's article
exists, but the attempted identifier still returns HTTP 404 from both Crossref
and the DOI handle API. Retain the omission rather than invent an identifier;
the method's name and formula remain stated without a novelty claim. Command:
`uv run --no-sync python -` using `urllib.request` against both recorded endpoints.
Result and UTC/source metadata: `paper/reference_verification.json`. The eight
previously verified references are unchanged. This is not a research trial.

Paper figure follow-through: annotate exact passes/scored on every bar, since a
missing positive N claim can reduce the claimed-N denominator. Place labels above
their confidence bounds and state statistic-specific scoring in the caption.
No estimates change. Initial Ruff check FAILs B905 (zip without an explicit
strict flag); assert matching cohort lengths with `strict=True`, then check/format
PASS. Final visual verification occurs on the complete-sample figure and PDF.

Complete gpt-oss checkpoint: all 45 fixed attempts now have receipts, with raw
statuses 43 ok and two `api_error: RuntimeError` (GARCH seed 2 and signflip seed 17).
The batch exits 1 because those requests exhaust the unchanged timeout/retry
policy; no failed seed is replaced. Lightning acquires the released slot and
Ultra remains queued. Commands: original saved gpt-oss launch, then `uv run
--no-sync python -m experiments.store_returns` and `uv run --no-sync python -m
runs.stage_review_raw`: PASS. The staged 175-receipt snapshot verifies 15,103
distinct per-ledger series and 14,631 referenced shared objects, with no restoration.
Logs: ignored `runs/review-gpt-complete-store.log` and
`runs/stage-review-gpt-complete.log`. Partial count receipts and acceptance
aggregates remain excluded. Raw ok status does not substitute for final submission
validation in the later complete-sample analysis.

Interpretation audit: explicitly state that cohort contrasts confound model
identity, execution interface and chosen tool-use patterns. Add this limitation
to the paper and README. No causal interface comparison is claimed; the fixed
within-pick PSR/DSR and claimed/ledger comparisons remain as specified. Prose-only
change; no new financial evaluation or experiment setting change.

Multiplicity wording audit: label the existing model-specific exact McNemar
p-values as unadjusted and descriptive; distinguish them from the prespecified
Holm-adjusted continuous-score tests. No statistic, family or threshold changes,
and no family-wide rejection claim is based on the raw decision p-values.
Ruff check/format PASS for the table generator; final results remain gated.
State the existing signed-rank normal approximation, dropped zeros and lack of
continuity correction explicitly. These describe the unchanged SciPy call.
