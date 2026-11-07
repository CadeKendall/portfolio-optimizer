import numpy as np
import pandas as pd
import pytest

from app.optimizer import (
    Constraints,
    InfeasibleError,
    check_feasible,
    estimate,
    max_feasible_return,
    max_sharpe_weights,
    min_vol_weights,
    optimize,
)


def two_assets(vol_a=0.2, vol_b=0.1, rho=0.3, mu_a=0.10, mu_b=0.05):
    names = ["A", "B"]
    mu = pd.Series([mu_a, mu_b], index=names)
    cov = pd.DataFrame(
        [[vol_a**2, rho * vol_a * vol_b], [rho * vol_a * vol_b, vol_b**2]],
        index=names,
        columns=names,
    )
    return mu, cov


def test_min_vol_matches_two_asset_closed_form():
    vol_a, vol_b, rho = 0.2, 0.1, 0.3
    mu, cov = two_assets(vol_a, vol_b, rho)
    covariance = rho * vol_a * vol_b
    expected_a = (vol_b**2 - covariance) / (vol_a**2 + vol_b**2 - 2 * covariance)
    weights = min_vol_weights(mu, cov, Constraints())
    assert weights["A"] == pytest.approx(expected_a, abs=1e-4)
    assert weights["B"] == pytest.approx(1 - expected_a, abs=1e-4)


def test_max_sharpe_matches_two_asset_closed_form():
    mu, cov = two_assets()
    risk_free = 0.02
    # The unconstrained tangency portfolio is proportional to inverse(cov) * excess returns.
    raw = np.linalg.solve(cov.to_numpy(), (mu - risk_free).to_numpy())
    expected = raw / raw.sum()
    weights = max_sharpe_weights(mu, cov, risk_free, Constraints())
    assert weights.to_numpy() == pytest.approx(expected, abs=1e-3)


def test_weights_sum_to_one_and_respect_bounds(prices):
    constraints = Constraints(min_weight=0.05, max_weight=0.40)
    result = optimize(prices, risk_free=0.0, constraints=constraints)
    portfolios = [result.max_sharpe, result.min_volatility, *result.frontier]
    assert result.max_sharpe is not None
    for portfolio in portfolios:
        weights = np.array(list(portfolio.weights.values()))
        assert weights.sum() == pytest.approx(1.0, abs=1e-9)
        assert (weights >= 0.05 - 1e-6).all()
        assert (weights <= 0.40 + 1e-6).all()


def test_frontier_rises_in_both_return_and_volatility(prices):
    result = optimize(prices, risk_free=0.0, frontier_points=20)
    assert len(result.frontier) >= 15
    returns = np.array([p.expected_return for p in result.frontier])
    vols = np.array([p.volatility for p in result.frontier])
    assert (np.diff(returns) > 0).all()
    assert (np.diff(vols) > -1e-5).all()


def test_key_portfolios_dominate_the_frontier(prices):
    result = optimize(prices, risk_free=0.01, frontier_points=25)
    assert all(p.volatility >= result.min_volatility.volatility - 1e-5 for p in result.frontier)
    assert all(p.sharpe <= result.max_sharpe.sharpe + 1e-4 for p in result.frontier)


@pytest.mark.parametrize(
    "n, constraints, message",
    [
        (4, Constraints(0.0, 0.20), "20.0% maximum across 4 assets"),
        (4, Constraints(0.30, 1.0), "30.0% minimum across 4 assets"),
        (4, Constraints(0.5, 0.2), "above the maximum"),
    ],
)
def test_infeasible_bounds_have_specific_messages(n, constraints, message):
    with pytest.raises(InfeasibleError, match=message):
        check_feasible(n, constraints)


def test_exactly_feasible_bounds_are_allowed():
    check_feasible(4, Constraints(0.0, 0.25))
    check_feasible(4, Constraints(0.25, 1.0))
    check_feasible(3, Constraints(0.0, 1 / 3))


def test_optimize_raises_on_infeasible_bounds(prices):
    with pytest.raises(InfeasibleError):
        optimize(prices, constraints=Constraints(0.0, 0.2))


def test_max_sharpe_with_no_excess_return_has_specific_message():
    mu, cov = two_assets(mu_a=0.02, mu_b=0.01)
    with pytest.raises(InfeasibleError, match="beat the 5.00% risk-free rate"):
        max_sharpe_weights(mu, cov, 0.05, Constraints())


def test_max_sharpe_accounts_for_bounds_when_checking_excess_return():
    # A beats the risk-free rate on its own, but the 50% cap forces in B.
    mu, cov = two_assets(mu_a=0.06, mu_b=0.00)
    assert max_feasible_return(mu, Constraints(0.0, 0.5)) == pytest.approx(0.03)
    with pytest.raises(InfeasibleError):
        max_sharpe_weights(mu, cov, 0.05, Constraints(0.0, 0.5))


def test_optimize_still_returns_min_vol_when_max_sharpe_is_impossible(prices):
    result = optimize(prices, risk_free=5.0)
    assert result.max_sharpe is None
    assert result.min_volatility is not None
    assert "risk-free rate" in result.warnings[0]


def test_estimate_annualizes_returns(prices):
    mu, cov = estimate(prices)
    daily = prices.pct_change().dropna()
    assert mu["AAA"] == pytest.approx(daily["AAA"].mean() * 252)
    # Shrinkage moves the sample variance, but it stays in the same range.
    assert cov.loc["AAA", "AAA"] == pytest.approx(daily["AAA"].var() * 252, rel=0.25)
