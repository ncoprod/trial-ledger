# trial-ledger

Count the backtests your AI agent actually ran.

Work in progress.

## Security and privacy notes

- **No command text by default.** Shell hooks and `trial-ledger exec` record only
  `command_sha256` (hash of the exact command) and `program` (the executable, plus the
  script or module when it runs an interpreter such as `python`, `uv run` or `node`).
  `TRIAL_LEDGER_CAPTURE_COMMANDS=1` adds a token-level `command_preview` in which the values of
  `--*api*/*key*/*token*/*secret*/*pass*/*auth*/*credential*` flags, `VAR=value` assignments
  and any 20+ character token mixing letters and digits are redacted, truncated at a token
  boundary after redaction. Raw commands are never stored.
- **Logging fails closed.** Every `TRIAL_LEDGER` line counts as one attempt. A line that
  cannot be parsed or validated is still recorded, as an unscored record with
  `extra.malformed = true` and the error type, so a bad line never hides the trials around it
  and N is never undercounted.
