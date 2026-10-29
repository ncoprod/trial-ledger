# Third-party notices

Verified 29 October 2026. Research use of hosted services does not redistribute their
weights. Hugging Face metadata revisions below establish the licence checked; they
**do not identify the unrecorded NIM deployment revision**. Machine-readable checks
and source URLs: [`experiments/results/license_sources.json`](experiments/results/license_sources.json).

## Models and coding-agent subjects

| Component used | Exact licence / terms | Attribution and handling |
|---|---|---|
| `openai/gpt-oss-20b`, NIM-hosted | Apache License 2.0; [model API](https://huggingface.co/api/models/openai/gpt-oss-20b), [LICENSE](https://huggingface.co/openai/gpt-oss-20b/blob/main/LICENSE) | OpenAI, gpt-oss. Model-card revision `6cee5e81ee83917806bbde320786a8fb61efebee`. We store requests' SHA-256 identifiers and responses, never weights or credentials. Redistributed Apache material requires its licence, applicable notices and change notices. |
| `nvidia/nemotron-3-ultra-550b-a55b`, NIM-hosted | OpenMDW License 1.1 ([model API](https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16), [licence](https://openmdw.ai/license/1-1/)) | NVIDIA, NVIDIA Nemotron 3 Ultra 550B A55B. Card revision `77df655d5e9f8362164ed14dd8b48f8bce657498`. Preserve OpenMDW notices/licence when distributing covered model material; no model material is redistributed here. |
| `nvidia/nemotron-3.5-lightning-30b-a3b`, NIM-hosted | OpenMDW License 1.1 ([model API](https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16), [licence](https://openmdw.ai/license/1-1/)) | NVIDIA, NVIDIA Nemotron 3.5 Lightning 30B A3B. Card revision `a9904d24bcc1d289a1950fa9d2b978c47cf903b9`. Same handling as above. |
| NVIDIA NIM endpoint | [NVIDIA API catalogue terms](https://build.nvidia.com/terms-of-service) in addition to model terms | Hosted inference only. Responses cached for exact research replay; deployment revision not pinned or recorded. |
| Codex CLI / `gpt-6.1-sol` | CLI: Apache License 2.0 ([upstream LICENSE](https://github.com/openai/codex/blob/main/LICENSE)); hosted model: proprietary OpenAI service terms | OpenAI Codex. Uses existing local sign-in; no CLI source, credentials or model weights redistributed. Record model alias, CLI version and reasoning setting per episode. |
| Claude Code / Haiku | Proprietary, copyright Anthropic PBC; [upstream licence notice](https://github.com/anthropics/claude-code/blob/main/LICENSE.md) points to [Commercial Terms](https://www.anthropic.com/legal/commercial-terms) | Anthropic Claude Code. Attempt retained; OAuth authentication failed before inference. No successful Claude episode or model-weight redistribution is claimed. |

## Market data — non-commercial restriction

BTCUSDT daily closes originate from **Binance Public Data**,
[data.binance.vision](https://data.binance.vision/), spot monthly daily-kline archives.

The repository README labels its downloader code MIT, but the **dataset** has separate
[Binance Vision Dataset Terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md),
version 1.0, updated 26 August 2026: **Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International (CC BY-NC-SA 4.0)** plus the stated dataset terms. The dataset terms
also cover derived data, charts and indicators. This restriction is material: the MIT
code licence does not grant commercial rights to Binance-derived research artifacts.

Attribution: Binance Public Data / Binance, BTCUSDT spot daily klines, downloaded
from Binance Vision. Modifications: extract closing prices, normalize millisecond /
microsecond timestamps, calculate returns, bootstrap return magnitudes or calibrate
GARCH, generate strategy return series and research figures. Retained prices,
Binance-derived null inputs, strategy returns and derived figures are provided for
non-commercial reproducibility under those dataset terms and CC BY-NC-SA 4.0.
Preserve source attribution, indicate modifications, link the licence and share
adapted covered data under the same licence. Commercial reuse requires the separate
permission described by Binance. See the [CC licence](https://creativecommons.org/licenses/by-nc-sa/4.0/).

`experiments/data.py` downloads each monthly archive and checks Binance's SHA-256
checksum before extracting the committed close series. The close CSV is retained
by the owner's decision for offline reproducibility. Frozen GARCH calibration and
input arrays preserve exact bytes across architectures; the statistical generator
uses symmetric clipping to keep simple returns valid without introducing a nonzero
conditional mean. No Yahoo, Kaggle or other market dataset is used by this repository.

## Reference software and mathematical methods

- **pypbo**, by its upstream contributors, commit
  `4d723f06498267a2a6280cb9d7d5649348e961d1`: **GNU Affero General Public License v3**,
  [upstream LICENSE](https://github.com/esvhd/pypbo/blob/4d723f06498267a2a6280cb9d7d5649348e961d1/LICENSE).
  The benchmark downloads and executes this reference under ignored
  `benchmarks/_vendor/`. It is not bundled in either wheel or committed source.
  Redistribution of covered source/binaries must preserve copyright/licence notices
  and meet AGPL corresponding-source obligations. This project implements CSCV and
  Sharpe statistics from the cited mathematical methods; no pypbo source is ported
  into the MIT runtime.
- Bailey & López de Prado, *The Sharpe Ratio Efficient Frontier* (2012) and
  *The Deflated Sharpe Ratio* (2014); Bailey, Borwein, López de Prado & Zhu,
  *The Probability of Backtest Overfitting* (2017). Methods are attributed in the
  README and source; papers are linked, not redistributed.
- Politis & Romano, *The Stationary Bootstrap* (1994), for null-market resampling;
  López de Prado, *Advances in Financial Machine Learning* (2018), for purged CV.
- Gençay (2026), *What survives honest evaluation?*,
  [arXiv:2608.27734](https://arxiv.org/abs/2608.27734), is related work, not an
  implementation or dataset dependency. Its text/code is not redistributed.

Runtime libraries are ordinary dependencies, not vendored: NumPy and SciPy
(BSD-3-Clause), MCP Python SDK (MIT), PyO3 (MIT OR Apache-2.0), rust-numpy (BSD-2-Clause),
and rayon (MIT OR Apache-2.0). Their distributions carry their own notices. Dev and
benchmark dependencies are declared separately in `pyproject.toml` / `uv.lock`.
