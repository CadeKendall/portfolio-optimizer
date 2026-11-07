"""Walk-forward backtest.

Timeline for a rebalance signalled at the close of day t:
  - Estimates use prices up to and including t.
  - The trade happens at the close of t+1, so t+1's return is earned by the old weights.
  - The new weights first earn the return of t+2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataError
from .optimizer import (
    Constraints,
    InfeasibleError,
    OptimizationFailed,
    TRADING_DAYS,
    check_feasible,
    estimate,
    max_sharpe_weights,
    min_vol_weights,
)

FREQUENCIES = {"monthly": "M", "quarterly": "Q"}
MAX_SHARPE = "Max Sharpe"
MIN_VOL = "Min volatility"
EQUAL_WEIGHT = "Equal weight"


@dataclass
class Skip:
    strategy: str
    date: str
    reason: str


@dataclass
class Metrics:
    cagr: float
    volatility: float
    sharpe: float | None
    max_drawdown: float
    annual_turnover: float


@dataclass
class StrategyResult:
    name: str
    equity: list[float]
    drawdown: list[float]
    metrics: Metrics


@dataclass
class BacktestResult:
    dates: list[str]
    strategies: list[StrategyResult]
    skips: list[Skip]
    rebalances: int


def signal_positions(index: pd.DatetimeIndex, frequency: str, lookback: int) -> list[int]:
    """Row positions of each period's last trading day that can be acted on."""
    periods = index.to_period(FREQUENCIES[frequency])
    is_period_end = periods[:-1] != periods[1:]
    ends = np.flatnonzero(is_period_end)
    # Needs a full lookback window, and a trade at i+1 that still earns a return at i+2.
    return [int(i) for i in ends if i >= lookback and i + 2 <= len(index) - 1]


def compute_targets(
    prices: pd.DataFrame,
    signals: list[int],
    lookback: int,
    risk_free: float,
    constraints: Constraints,
) -> tuple[dict[str, dict[int, np.ndarray]], list[Skip]]:
    """Target weights per strategy, keyed by signal position.

    A signal with no entry means "keep holding what you have". If the very first signal
    can't be solved there is nothing to hold yet, so that strategy starts equal weight.
    """
    n = prices.shape[1]
    equal = np.full(n, 1 / n)
    targets: dict[str, dict[int, np.ndarray]] = {
        MAX_SHARPE: {},
        MIN_VOL: {},
        EQUAL_WEIGHT: {i: equal for i in signals},
    }
    skips: list[Skip] = []

    for i in signals:
        window = prices.iloc[i - lookback : i + 1]
        mu, cov = estimate(window)
        solvers = {
            MAX_SHARPE: lambda: max_sharpe_weights(mu, cov, risk_free, constraints),
            MIN_VOL: lambda: min_vol_weights(mu, cov, constraints),
        }
        for name, solve in solvers.items():
            try:
                targets[name][i] = solve().to_numpy()
            except (InfeasibleError, OptimizationFailed) as exc:
                is_first = not targets[name]
                if is_first:
                    targets[name][i] = equal
                action = "started equal weight" if is_first else "kept previous weights"
                skips.append(Skip(name, prices.index[i].date().isoformat(), f"{exc} ({action})"))
    return targets, skips


def simulate(
    prices: np.ndarray, trades: dict[int, np.ndarray], start: int, cost_bps: float
) -> tuple[np.ndarray, list[float]]:
    """Simulate from row `start`, trading at the close of each row in `trades`.

    Returns the equity curve, with 1.0 invested before costs, and the traded notional
    (sum of absolute weight changes) of each trade. The first trade buys from cash.
    """
    rate = cost_bps / 10_000
    holdings = np.zeros(prices.shape[1])
    nav = 1.0
    equity = np.empty(len(prices) - start)
    traded = []
    for k in range(start, len(prices)):
        if k > start:
            growth = holdings * prices[k] / prices[k - 1]
            portfolio_growth = growth.sum()
            nav *= portfolio_growth
            holdings = growth / portfolio_growth
        if k in trades:
            notional = float(np.abs(trades[k] - holdings).sum())
            nav *= 1 - rate * notional
            holdings = trades[k]
            traded.append(notional)
        equity[k - start] = nav
    return equity, traded


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    return equity / np.maximum.accumulate(equity) - 1


def compute_metrics(
    equity: np.ndarray, dates: pd.DatetimeIndex, risk_free: float, traded: list[float]
) -> Metrics:
    years = (dates[-1] - dates[0]).days / 365.25
    daily = equity[1:] / equity[:-1] - 1
    volatility = float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS))
    mean = float(daily.mean() * TRADING_DAYS)
    # The opening purchase isn't turnover. One-way means half the traded notional.
    one_way = sum(traded[1:]) / 2
    return Metrics(
        cagr=float(equity[-1] ** (1 / years) - 1) if years > 0 else 0.0,
        volatility=volatility,
        sharpe=(mean - risk_free) / volatility if volatility > 0 else None,
        max_drawdown=float(drawdown_series(equity).min()),
        annual_turnover=one_way / years if years > 0 else 0.0,
    )


def run_backtest(
    prices: pd.DataFrame,
    benchmark: pd.Series | None = None,
    benchmark_name: str = "Benchmark",
    lookback: int = 756,
    frequency: str = "monthly",
    cost_bps: float = 5.0,
    risk_free: float = 0.04,
    constraints: Constraints = Constraints(),
) -> BacktestResult:
    check_feasible(prices.shape[1], constraints)
    signals = signal_positions(prices.index, frequency, lookback)
    if not signals:
        raise DataError(
            f"Not enough history: the {lookback}-day lookback uses up all"
            f" {len(prices)} shared trading days. Shorten the lookback or add history."
        )

    targets, skips = compute_targets(prices, signals, lookback, risk_free, constraints)
    start = signals[0] + 1
    dates = prices.index[start:]
    values = prices.to_numpy(dtype=float)

    strategies = []
    for name, by_signal in targets.items():
        trades = {i + 1: weights for i, weights in by_signal.items()}
        equity, traded = simulate(values, trades, start, cost_bps)
        strategies.append(
            StrategyResult(
                name,
                equity.tolist(),
                drawdown_series(equity).tolist(),
                compute_metrics(equity, dates, risk_free, traded),
            )
        )

    if benchmark is not None:
        series = benchmark.reindex(prices.index).to_numpy(dtype=float)[start:]
        equity = series / series[0]
        strategies.append(
            StrategyResult(
                benchmark_name,
                equity.tolist(),
                drawdown_series(equity).tolist(),
                compute_metrics(equity, dates, risk_free, []),
            )
        )

    return BacktestResult(
        dates=[d.date().isoformat() for d in dates],
        strategies=strategies,
        skips=skips,
        rebalances=len(signals),
    )
