import numpy as np
import pandas as pd
import pytest

from app.backtest import (
    EQUAL_WEIGHT,
    INVERSE_VOL,
    MAX_SHARPE,
    MIN_VOL,
    compute_targets,
    drawdown_series,
    run_backtest,
    signal_positions,
    simulate,
)
from app.data import DataError
from app.optimizer import Constraints, OptimizationFailed, inverse_vol_weights

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


# Two full calendar years of business days: Mon 2022-01-03 to Fri 2023-12-29.
KNOWN_RANGE = pd.bdate_range("2022-01-03", "2023-12-29")


def test_monthly_rebalance_dates_on_known_range():
    signals = signal_positions(KNOWN_RANGE, "monthly", 5)
    # Every month-end from Jan 2022 to Nov 2023. Dec 2023 is the final partial period.
    expected = pd.bdate_range("2022-01-01", "2023-11-30", freq="BME")
    assert len(signals) == 23
    assert list(KNOWN_RANGE[signals]) == list(expected)


def test_quarterly_rebalance_dates_on_known_range():
    signals = signal_positions(KNOWN_RANGE, "quarterly", 5)
    # Quarter-ends Q1 2022 to Q3 2023. Q4 2023 is the final partial period.
    expected = pd.to_datetime(
        ["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-30", "2023-03-31", "2023-06-30", "2023-09-29"]
    )
    assert len(signals) == 7
    assert list(KNOWN_RANGE[signals]) == list(expected)


def test_lookback_drops_early_period_ends_on_known_range():
    # Row 60 is 2022-03-28, so the Jan (row 20) and Feb (row 39) month-ends lack a full
    # window, but the Mar 31 (row 63) month-end and quarter-end both qualify.
    assert KNOWN_RANGE[60] == pd.Timestamp("2022-03-28")
    monthly = signal_positions(KNOWN_RANGE, "monthly", 60)
    quarterly = signal_positions(KNOWN_RANGE, "quarterly", 60)
    assert len(monthly) == 21
    assert len(quarterly) == 7
    assert KNOWN_RANGE[monthly[0]] == KNOWN_RANGE[quarterly[0]] == pd.Timestamp("2022-03-31")


def test_run_backtest_reports_rebalance_count_per_frequency():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0003, 0.01, size=(len(KNOWN_RANGE), 3))
    frame = pd.DataFrame(
        100 * np.cumprod(1 + returns, axis=0), index=KNOWN_RANGE, columns=["A", "B", "C"]
    )
    monthly = run_backtest(frame, lookback=60, frequency="monthly", risk_free=0.0)
    quarterly = run_backtest(frame, lookback=60, frequency="quarterly", risk_free=0.0)
    assert monthly.rebalances == 21
    assert quarterly.rebalances == 7
    # Both start trading the day after the first signal (Mar 31 2022).
    assert monthly.dates[0] == quarterly.dates[0] == "2022-04-01"


# Inverse volatility. Daily returns alternate +s/-s, so each asset's sample std is exactly
# proportional to s and the weights can be checked by hand.
def alternating_prices(steps, days=21, start="2024-01-01"):
    signs = np.where(np.arange(days - 1) % 2 == 0, 1.0, -1.0)
    returns = np.outer(signs, steps)
    values = 100 * np.vstack([np.ones(len(steps)), np.cumprod(1 + returns, axis=0)])
    index = pd.bdate_range(start, periods=days)
    return pd.DataFrame(values, index=index, columns=[f"A{k}" for k in range(len(steps))])


def test_inverse_vol_weights_hand_checked():
    # Vols in ratio 1:2:4, so inverse vols 4:2:1 and weights 4/7, 2/7, 1/7.
    weights = inverse_vol_weights(alternating_prices([0.01, 0.02, 0.04]), Constraints())
    assert list(weights.index) == ["A0", "A1", "A2"]
    assert weights.to_numpy() == pytest.approx([4 / 7, 2 / 7, 1 / 7], abs=1e-12)


def test_inverse_vol_weights_equal_vols_give_equal_weight():
    weights = inverse_vol_weights(alternating_prices([0.03, 0.03]), Constraints())
    assert weights.to_numpy() == pytest.approx([0.5, 0.5], abs=1e-12)


def test_inverse_vol_weights_cap_redistributes_in_proportion():
    # 4/7 is above a 50% cap, so A0 gets 50% and the other 50% splits 2:1.
    prices = alternating_prices([0.01, 0.02, 0.04])
    weights = inverse_vol_weights(prices, Constraints(max_weight=0.5))
    assert weights.to_numpy() == pytest.approx([0.5, 1 / 3, 1 / 6], abs=1e-9)


def test_inverse_vol_weights_floor_redistributes_in_proportion():
    # 1/7 is below a 20% floor, so A2 gets 20% and the other 80% splits 2:1.
    prices = alternating_prices([0.01, 0.02, 0.04])
    weights = inverse_vol_weights(prices, Constraints(min_weight=0.2))
    assert weights.to_numpy() == pytest.approx([0.8 * 2 / 3, 0.8 / 3, 0.2], abs=1e-9)


def test_inverse_vol_weights_reject_zero_volatility():
    prices = alternating_prices([0.01, 0.0])
    with pytest.raises(OptimizationFailed, match="zero or undefined volatility"):
        inverse_vol_weights(prices, Constraints())


def test_inverse_vol_targets_use_only_the_lookback_window():
    # A1 is calm (+-1%) up to the Jan 31 signal and very volatile (+-10%) afterwards.
    # With a 20-day lookback the signal must see only the calm 1:2 vol ratio -> 2/3, 1/3.
    calm = alternating_prices([0.01, 0.02], days=40)
    signals = signal_positions(calm.index, "monthly", 20)
    i = signals[0]
    assert calm.index[i] == pd.Timestamp("2024-01-31")
    later = calm.pct_change().fillna(0.0)
    later.iloc[i + 1 :, 0] *= 10
    prices = 100 * (1 + later).cumprod()

    targets, skips = compute_targets(prices, signals, 20, 0.0, Constraints(), inverse_volatility=True)
    assert not [s for s in skips if s.strategy == INVERSE_VOL]
    assert targets[INVERSE_VOL][i] == pytest.approx([2 / 3, 1 / 3], abs=1e-12)


def test_inverse_vol_is_opt_in(prices):
    default = run_backtest(prices, lookback=LOOKBACK, risk_free=0.0)
    assert INVERSE_VOL not in [s.name for s in default.strategies]

    benchmark = prices["AAA"].rename("SPY")
    result = run_backtest(
        prices,
        benchmark=benchmark,
        benchmark_name="SPY",
        lookback=LOOKBACK,
        risk_free=0.0,
        inverse_volatility=True,
    )
    names = [s.name for s in result.strategies]
    assert names == [MAX_SHARPE, MIN_VOL, EQUAL_WEIGHT, INVERSE_VOL, "SPY"]
    by_name = {s.name: s for s in result.strategies}
    # Turning it on doesn't change the other strategies.
    for s in default.strategies:
        assert by_name[s.name].equity == pytest.approx(s.equity)
    inv = by_name[INVERSE_VOL]
    assert len(inv.equity) == len(result.dates)
    assert np.isfinite(inv.equity).all()
    assert inv.metrics.annual_turnover > 0
    # Leaning toward the low-vol assets makes it calmer than equal weight here.
    assert inv.metrics.volatility < by_name[EQUAL_WEIGHT].metrics.volatility


def test_inverse_vol_weights_tilt_toward_low_vol_assets(prices):
    signals = signal_positions(prices.index, "monthly", LOOKBACK)
    targets, _ = compute_targets(prices, signals, LOOKBACK, 0.0, Constraints(), inverse_volatility=True)
    assert len(targets[INVERSE_VOL]) == len(signals)
    for weights in targets[INVERSE_VOL].values():
        assert weights.sum() == pytest.approx(1)
        # Fixture vols are AAA 25%, BBB 10%, CCC 18%, DDD 5%.
        assert weights[3] > weights[1] > weights[2] > weights[0] > 0


def test_inverse_vol_respects_weight_bounds_in_backtest(prices):
    signals = signal_positions(prices.index, "quarterly", LOOKBACK)
    bounds = Constraints(min_weight=0.1, max_weight=0.4)
    targets, skips = compute_targets(prices, signals, LOOKBACK, 0.0, bounds, inverse_volatility=True)
    assert not [s for s in skips if s.strategy == INVERSE_VOL]
    for weights in targets[INVERSE_VOL].values():
        assert weights.sum() == pytest.approx(1)
        assert weights.min() >= 0.1 - 1e-9
        assert weights.max() <= 0.4 + 1e-9


def test_inverse_vol_no_lookahead(prices):
    signals = signal_positions(prices.index, "monthly", LOOKBACK)
    cutoff = signals[len(signals) // 2]

    changed = prices.copy()
    changed.iloc[cutoff + 1 :] *= np.random.default_rng(1).uniform(0.3, 3.0, changed.iloc[cutoff + 1 :].shape)

    args = (signals, LOOKBACK, 0.0, Constraints())
    original, _ = compute_targets(prices, *args, inverse_volatility=True)
    perturbed, _ = compute_targets(changed, *args, inverse_volatility=True)

    later_differs = False
    for i in signals:
        if i <= cutoff:
            assert original[INVERSE_VOL][i] == pytest.approx(perturbed[INVERSE_VOL][i], abs=1e-12)
        elif not np.allclose(original[INVERSE_VOL][i], perturbed[INVERSE_VOL][i], atol=1e-3):
            later_differs = True
    assert later_differs  # Proves that the perturbation reached the estimator.
