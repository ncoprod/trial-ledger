# Research result versions

The agent-search evidence directly in this directory is **superseded**.
Independent review found future-dependent volatility sizing in its backtester.
Those searches saw contaminated feedback; rescoring their final choices cannot
repair the adaptive experiment. The original receipts, response caches, ledgers,
metrics and manifest remain unchanged for auditability.
The [original report](RESULTS.md), [episode metrics](episodes.json) and
[provenance manifest](provenance.json) describe that superseded version.

The corrected fixed plan and fresh adaptive searches are documented in
[the protocol](../PROTOCOL.md); their receipts and regenerated results are under
[`causal-v2/`](causal-v2/). Frozen null inputs are reused, so these are new search
trajectories rather than a wholly independent market replication. The BTC period
was previously examined and is diagnostic in the corrected study.

Native-performance benchmarks are a separate fixed-input experiment. Their
source and output checks are recorded under [`benchmarks/results/`](../../benchmarks/results/).
See [the iteration log](../../ITERATIONS.md) for review fixes and failures.
