"""compute_metrics checked against hand-calculated values on tiny equity series."""

import math

import numpy as np
import pandas as pd
import pytest

from app.backtest import Metrics, compute_metrics

# Five points spanning 2020-01-01 to 2024-01-01: 366 + 365 + 365 + 365 = 1461 days,
# which is exactly 4 * 365.25, so the backtest spans exactly 4 years.
DATES = pd.DatetimeIndex(["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"])
# Daily returns: +20%, -25%, +20%, +92%. Final value 2.0736 = 1.2 ** 4.
EQUITY = np.array([1.0, 1.2, 0.9, 1.08, 2.0736])


def test_compute_metrics_matches_hand_calculation():
    traded = [1.0, 0.3, 0.1]
    m = compute_metrics(EQUITY, DATES, risk_free=0.04, traded=traded)

    # CAGR: 2.0736 ** (1/4) - 1 = 1.2 - 1.
    assert m.cagr == pytest.approx(0.2)

    # Returns 0.2, -0.25, 0.2, 0.92 -> mean 1.07 / 4 = 0.2675.
    # Deviations -0.0675, -0.5175, -0.0675, 0.6525; squares sum to 0.702675.
    # Sample variance 0.702675 / 3 = 0.234225, annualized by 252 trading days.
    expected_vol = math.sqrt(0.234225 * 252)
    assert m.volatility == pytest.approx(expected_vol)

    # Sharpe: (annualized mean - risk free) / volatility.
    assert m.sharpe == pytest.approx((0.2675 * 252 - 0.04) / expected_vol)

    # Peak 1.2 then trough 0.9: 0.9 / 1.2 - 1 = -0.25. Later new highs don't erase it.
    assert m.max_drawdown == pytest.approx(-0.25)

    # Opening buy (1.0) excluded; one-way = (0.3 + 0.1) / 2 = 0.2 over 4 years.
    assert m.annual_turnover == pytest.approx(0.05)


def test_compute_metrics_returns_every_field_checked_above():
    # Guards against a new metric being added without a hand-checked test here.
    assert set(Metrics.__dataclass_fields__) == {
        "cagr",
        "volatility",
        "sharpe",
        "max_drawdown",
        "annual_turnover",
    }


def test_flat_growth_has_zero_volatility_and_no_sharpe():
    # Doubles every step: identical returns, so zero dispersion and Sharpe is None.
    # (Powers of two keep the returns exactly equal in floating point.)
    equity = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    m = compute_metrics(equity, DATES, risk_free=0.04, traded=[1.0])

    assert m.cagr == pytest.approx(1.0)  # 16 ** (1/4) - 1 over 4 years
    assert m.volatility == 0.0
    assert m.sharpe is None
    assert m.max_drawdown == 0.0  # never below its running peak
    assert m.annual_turnover == 0.0  # only the opening purchase


def test_benchmark_with_no_trades_has_zero_turnover():
    m = compute_metrics(EQUITY, DATES, risk_free=0.0, traded=[])
    assert m.annual_turnover == 0.0
    assert m.sharpe == pytest.approx(0.2675 * 252 / math.sqrt(0.234225 * 252))


def test_loss_gives_negative_cagr_and_drawdown_to_final_value():
    # Falls 10% a year for 4 years, then the whole path is one drawdown.
    equity = np.array([1.0, 0.9, 0.81, 0.729, 0.6561])
    m = compute_metrics(equity, DATES, risk_free=0.0, traded=[1.0])

    assert m.cagr == pytest.approx(-0.1)
    assert m.max_drawdown == pytest.approx(0.6561 - 1)


def test_zero_length_span_gives_zero_cagr_and_turnover():
    same_day = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-01"])
    m = compute_metrics(np.array([1.0, 1.1, 1.0]), same_day, risk_free=0.0, traded=[1.0, 0.5])
    assert m.cagr == 0.0
    assert m.annual_turnover == 0.0
