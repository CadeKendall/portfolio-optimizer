import numpy as np
import pandas as pd
import pytest

from app.backtest import (
    EQUAL_WEIGHT,
    MAX_SHARPE,
    MIN_VOL,
    compute_targets,
    drawdown_series,
    run_backtest,
    signal_positions,
    simulate,
)
from app.data import DataError
from app.optimizer import Constraints

LOOKBACK = 252


def test_simulate_constant_weights_matches_hand_calculation():
    prices = np.array([[100.0, 100.0], [110.0, 100.0], [110.0, 90.0]])
    equity, traded = simulate(prices, {0: np.array([0.5, 0.5])}, 0, cost_bps=0)
    # Day 1: 0.5*1.1 + 0.5*1.0 = 1.05, and holdings drift to 0.55/1.05 and 0.5/1.05.
    # Day 2: A is flat and B falls 10%, so 1.05 * (0.55/1.05 + 0.5/1.05 * 0.9) = 1.00.
    assert equity == pytest.approx([1.0, 1.05, 1.0])
    assert traded == [pytest.approx(1.0)]


def test_simulate_rebalance_applies_only_to_later_returns():
    prices = np.array([[100.0, 100.0], [110.0, 100.0], [121.0, 100.0]])
    trades = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
    equity, traded = simulate(prices, trades, 0, cost_bps=0)
    # Day 1 is earned by the old weights. Day 2 is earned by B, which is flat.
    assert equity == pytest.approx([1.0, 1.1, 1.1])
    assert traded == pytest.approx([1.0, 2.0])


def test_costs_reduce_equity_by_exact_amount():
    prices = np.array([[100.0, 100.0], [110.0, 100.0], [121.0, 100.0]])
    trades = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
    no_cost, _ = simulate(prices, trades, 0, cost_bps=0)
    with_cost, _ = simulate(prices, trades, 0, cost_bps=100)
    # 1% on the opening 100% purchase, then 1% on the 200% notional switch.
    assert with_cost[-1] / no_cost[-1] == pytest.approx(0.99 * 0.98)


def test_rebalance_back_to_target_trades_only_the_drift():
    prices = np.array([[100.0, 100.0], [150.0, 100.0], [150.0, 100.0]])
    target = np.array([0.5, 0.5])
    _, traded = simulate(prices, {0: target, 1: target}, 0, cost_bps=0)
    # After day 1 the holdings are 0.6/0.4, so returning to 0.5/0.5 trades 0.2.
    assert traded[1] == pytest.approx(0.2)


def test_drawdown_series():
    assert drawdown_series(np.array([1.0, 2.0, 1.0, 3.0])) == pytest.approx([0, 0, -0.5, 0])


def test_signals_are_period_ends_with_room_for_lookback_and_trade(prices):
    signals = signal_positions(prices.index, "monthly", LOOKBACK)
    for i in signals:
        assert prices.index[i].month != prices.index[i + 1].month
        assert i >= LOOKBACK
        assert i + 2 <= len(prices) - 1
    assert len(signal_positions(prices.index, "quarterly", LOOKBACK)) < len(signals)


def test_signal_excludes_final_partial_period():
    index = pd.bdate_range("2024-01-01", "2024-03-15")
    signals = signal_positions(index, "monthly", 5)
    assert [index[i].month for i in signals] == [1, 2]


def test_signal_on_second_to_last_row_is_excluded_because_new_weights_earn_nothing():
    index = pd.bdate_range("2024-01-01", "2024-03-01")  # Feb 29 is the penultimate row.
    signals = signal_positions(index, "monthly", 5)
    assert [index[i].month for i in signals] == [1]


def test_no_lookahead_future_prices_do_not_change_earlier_weights(prices):
    signals = signal_positions(prices.index, "monthly", LOOKBACK)
    cutoff = signals[len(signals) // 2]

    changed = prices.copy()
    changed.iloc[cutoff + 1 :] *= np.random.default_rng(1).uniform(0.3, 3.0, changed.iloc[cutoff + 1 :].shape)

    constraints = Constraints()
    original, _ = compute_targets(prices, signals, LOOKBACK, 0.0, constraints)
    perturbed, _ = compute_targets(changed, signals, LOOKBACK, 0.0, constraints)

    later_differs = False
    for name in (MAX_SHARPE, MIN_VOL):
        for i in signals:
            if i not in original[name] or i not in perturbed[name]:
                continue
            if i <= cutoff:
                assert original[name][i] == pytest.approx(perturbed[name][i], abs=1e-12)
            elif not np.allclose(original[name][i], perturbed[name][i], atol=1e-3):
                later_differs = True
    assert later_differs  # Proves that the perturbation reached the estimator.


def test_new_weights_do_not_earn_the_trade_day_return():
    """A jump on the signal's next day must not reach weights chosen before it."""
    index = pd.bdate_range("2023-01-02", "2023-03-31")
    values = np.ones((len(index), 2)) * 100.0
    prices = np.cumprod(np.where(np.arange(len(index)) % 2 == 0, 1.001, 0.999))
    values[:, 0] *= prices
    frame = pd.DataFrame(values, index=index, columns=["A", "B"])
    signal = next(i for i in range(len(index) - 1) if index[i].month != index[i + 1].month)

    frame.iloc[signal + 1 :, 0] *= 2  # A doubles on the trade day.
    equity, _ = simulate(frame.to_numpy(), {signal + 1: np.array([1.0, 0.0])}, signal + 1, 0)
    assert equity[0] == 1.0
    assert equity[1] == pytest.approx(frame.iloc[signal + 2, 0] / frame.iloc[signal + 1, 0])


def test_full_backtest_shape_and_metrics(prices):
    benchmark = prices["AAA"].rename("SPY")
    result = run_backtest(
        prices, benchmark=benchmark, benchmark_name="SPY", lookback=LOOKBACK, risk_free=0.0
    )
    assert [s.name for s in result.strategies] == [MAX_SHARPE, MIN_VOL, EQUAL_WEIGHT, "SPY"]
    assert result.rebalances > 30
    for strategy in result.strategies:
        assert len(strategy.equity) == len(result.dates) == len(strategy.drawdown)
        assert np.isfinite(strategy.equity).all()
        assert -1 < strategy.metrics.max_drawdown <= 0
        assert strategy.metrics.volatility > 0

    by_name = {s.name: s for s in result.strategies}
    assert by_name["SPY"].equity[0] == 1.0
    assert by_name["SPY"].metrics.annual_turnover == 0
    assert by_name[EQUAL_WEIGHT].metrics.annual_turnover > 0
    # Min volatility should really be the calmer strategy against a 25%-vol asset.
    assert by_name[MIN_VOL].metrics.volatility < by_name["SPY"].metrics.volatility


def test_higher_costs_lower_returns(prices):
    cheap = run_backtest(prices, lookback=LOOKBACK, cost_bps=0, risk_free=0.0)
    costly = run_backtest(prices, lookback=LOOKBACK, cost_bps=50, risk_free=0.0)
    for low, high in zip(cheap.strategies, costly.strategies):
        assert high.equity[-1] < low.equity[-1]


def test_unsolvable_rebalances_start_equal_weight_then_keep_weights_and_are_reported(prices):
    # Nothing beats a 500% risk-free rate, so max Sharpe fails at every rebalance.
    result = run_backtest(prices, lookback=LOOKBACK, risk_free=5.0)
    skips = [s for s in result.skips if s.strategy == MAX_SHARPE]
    assert len(skips) == result.rebalances
    assert "started equal weight" in skips[0].reason
    assert "kept previous weights" in skips[1].reason
    assert len(skips[0].date) == 10


def test_not_enough_history_raises(prices):
    with pytest.raises(DataError, match="Not enough history"):
        run_backtest(prices, lookback=len(prices))
