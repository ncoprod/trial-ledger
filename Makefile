.PHONY: install test lint rust bench data reproduce experiment

install:            ## dev environment incl. the Rust core (needs a Rust toolchain)
	uv sync --all-extras --group dev --group experiments

test:
	uv run pytest
	cd rust && cargo test --release

lint:
	uv run ruff check .
	uv run ruff format --check .

bench:              ## pypbo vs NumPy vs Rust (fetches pypbo at a pinned commit)
	uv sync --all-extras --group dev --group bench
	uv run python benchmarks/bench_pbo.py

data:               ## re-download BTCUSDT daily klines from data.binance.vision
	uv run python -m experiments.data

reproduce:          ## replay every agent episode from the cache, rebuild tables + figures
	uv run python -m experiments.run reproduce

experiment:         ## new API calls (needs NVIDIA_API_KEY)
	uv run python -m experiments.run agents --seeds 0-9
	uv run python -m experiments.run baseline --seeds 0-9
	uv run python -m experiments.run analyze
	uv run python -m experiments.run figures
