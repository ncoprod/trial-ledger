import numpy as np
import pytest

from trial_ledger.backtest import (
    FAMILIES,
    Market,
    expand_grid,
    normalise_spec,
    run_backtest,
    signal,
    strategy_returns,
)


@pytest.fixture
def market():
    r = np.random.default_rng(0).standard_t(4, 1500) * 0.02
    return Market("t", r, periods_per_year=365, fee_bps=5)


SPECS = [
    {"family": "momentum", "lookback": 30},
    {"family": "ma_crossover", "fast": 5, "slow": 40, "long_only": True},
    {"family": "mean_reversion", "lookback": 20, "entry_z": 1.5, "vol_target": True},
    {"family": "breakout", "lookback": 55},
    {"family": "rsi", "period": 7, "lower": 25, "upper": 75},
]


@pytest.mark.parametrize("spec", SPECS)
def test_no_lookahead(market, spec):
    """Positions decided up to t must not change when returns after t are altered."""
    cut = 900
    pos = signal(spec, market)
    r2 = market.returns.copy()
    r2[cut + 1 :] = np.random.default_rng(1).standard_normal(r2.size - cut - 1)
    pos2 = signal(spec, Market("t2", r2))
    np.testing.assert_array_equal(pos[: cut + 1], pos2[: cut + 1])
    changed = Market("t2", r2, fee_bps=market.fee_bps)
    np.testing.assert_array_equal(
        strategy_returns(spec, market)[:cut], strategy_returns(spec, changed)[:cut]
    )


@pytest.mark.parametrize("spec", SPECS)
def test_positions_bounded_and_returns_length(market, spec):
    pos = signal(spec, market)
    assert np.all(np.abs(pos) <= 2.0 + 1e-12)
    assert strategy_returns(spec, market).size == market.returns.size - 1


def test_costs_reduce_returns(market):
    spec = {"family": "momentum", "lookback": 3}
    free = Market("f", market.returns, fee_bps=0)
    assert strategy_returns(spec, market).sum() < strategy_returns(spec, free).sum()


def test_hand_checked_momentum():
    r = np.array([0.0, 0.01, 0.02, -0.05, 0.01, 0.01])
    m = Market("h", r, fee_bps=0)
    # lookback 2: mean(r[t-1:t+1]) signs from t=1: +, +, -, -, +
    out = strategy_returns({"family": "momentum", "lookback": 2}, m)
    expected = np.array([0.0, 0.02, -0.05, -0.01, -0.01])
    np.testing.assert_allclose(out, expected)


def test_zero_alpha_market_has_no_edge_on_average():
    rng = np.random.default_rng(3)
    srs = []
    for _ in range(200):
        mag = np.abs(rng.standard_t(3, 1000)) * 0.02
        r = mag * rng.choice([-1.0, 1.0], size=mag.size)
        srs.append(
            run_backtest({"family": "momentum", "lookback": 10}, Market("n", r, fee_bps=0)).sharpe
        )
    assert abs(np.mean(srs)) < 3 * np.std(srs) / np.sqrt(len(srs))


def test_normalise_spec_validation():
    s = normalise_spec({"family": "rsi", "params": {"period": 9.6}})
    assert s == {
        "family": "rsi",
        "period": 10,
        "lower": 30.0,
        "upper": 70.0,
        "long_only": False,
        "vol_target": False,
    }
    with pytest.raises(ValueError, match="unknown family"):
        normalise_spec({"family": "neural_net"})
    with pytest.raises(ValueError, match="fast < slow"):
        normalise_spec({"family": "ma_crossover", "fast": 50, "slow": 10})
    with pytest.raises(ValueError, match="out of range"):
        normalise_spec({"family": "momentum", "lookback": 1000})
    with pytest.raises(ValueError, match="unknown parameter"):
        normalise_spec({"family": "momentum", "stop_loss": 0.1})


def test_expand_grid():
    g = expand_grid("ma_crossover", {"fast": [5, 10], "slow": [50, 100, 200]}, long_only=True)
    assert len(g) == 6 and all(x["long_only"] for x in g)
    assert set(FAMILIES) == {"momentum", "ma_crossover", "mean_reversion", "breakout", "rsi"}


def test_eval_from_uses_warmup(market):
    full = run_backtest({"family": "momentum", "lookback": 50}, market)
    oos = run_backtest({"family": "momentum", "lookback": 50}, market, eval_from=1000)
    np.testing.assert_allclose(oos.returns, full.returns[999:])
    assert oos.n_obs == market.returns.size - 1000


@pytest.mark.parametrize("spec", [{**s, "vol_target": True} for s in SPECS])
def test_oos_prefix_and_research_returns_do_not_use_future(market, spec):
    split, prefix = 900, 200
    altered = market.returns.copy()
    altered[split + prefix :] *= 3
    changed = Market("changed", altered, fee_bps=market.fee_bps)
    np.testing.assert_array_equal(
        run_backtest(spec, market).returns[: split - 1],
        run_backtest(spec, market.slice(0, split)).returns,
    )
    np.testing.assert_array_equal(
        run_backtest(spec, market, eval_from=split).returns[:prefix],
        run_backtest(spec, changed, eval_from=split).returns[:prefix],
    )


def test_window_drawdown_includes_initial_equity_and_actual_turnover():
    market = Market("boundary", np.array([0, 0.1, 0.1, -0.1, 0, 0]), fee_bps=0)
    result = run_backtest({"family": "momentum", "lookback": 2}, market, eval_from=3)
    np.testing.assert_allclose(result.returns, [-0.1, 0, 0])
    assert result.max_drawdown == pytest.approx(0.1)
    assert result.turnover == pytest.approx(2 / 3)
