# trial-ledger

Count the backtests your AI agent actually ran.

Work in progress.

## Security and privacy notes

- **Commands are not stored verbatim by default.** Shell hooks and `trial-ledger exec` record
  `command_sha256` (the exact command, hashed) plus a short redacted `command_preview`. The
  preview strips `VAR=value` assignments and env prefixes, `Authorization:`/`Bearer` headers,
  `--api-key`/`--token`-style flags, credentials in URLs and token-shaped strings (`sk-`,
  `gho_`/`ghp_`, `nvapi-`, `hf_`, JWTs, long hex or base64 runs). Redaction is best effort;
  full command capture is opt-in with `TRIAL_LEDGER_CAPTURE_COMMANDS=1`.
- **Logging fails closed.** Every `TRIAL_LEDGER` line counts as one attempt. A line that
  cannot be parsed or validated is still recorded, as an unscored record with
  `extra.malformed = true` and the error type, so a bad line never hides the trials around it
  and N is never undercounted.
