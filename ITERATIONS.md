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
The held-out BTC period is evaluated once for each fixed final submitted candidate;
no strategy or experiment-protocol tuning uses that period. Reproduction repeats
arithmetic on the fixed candidates, not candidate selection.

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
