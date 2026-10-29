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
