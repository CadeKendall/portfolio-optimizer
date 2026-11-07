"""Mean-variance optimization on top of PyPortfolioOpt."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models
from pypfopt.exceptions import OptimizationError

TRADING_DAYS = 252
_FEASIBILITY_TOLERANCE = 1e-9


class InfeasibleError(ValueError):
    """No portfolio can satisfy the request. The message is shown to the user."""


class OptimizationFailed(RuntimeError):
    """The solver didn't return a usable portfolio."""


@dataclass(frozen=True)
class Constraints:
    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass
class Portfolio:
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float | None


@dataclass
class OptimizationResult:
    max_sharpe: Portfolio | None
    min_volatility: Portfolio
    frontier: list[Portfolio]
    assets: dict[str, dict[str, float]]
    warnings: list[str]


def estimate(prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualized arithmetic mean returns and Ledoit-Wolf shrinkage covariance.

    Arithmetic means are used because a portfolio's arithmetic return is the weighted
    sum of its assets' returns, which mean-variance optimization relies on. Geometric
    returns don't add up that way.
    """
    returns = prices.pct_change().dropna(how="any")
    mu = returns.mean() * TRADING_DAYS
    cov = risk_models.CovarianceShrinkage(
        returns, returns_data=True, frequency=TRADING_DAYS
    ).ledoit_wolf()
    return mu, cov


def check_feasible(n_assets: int, constraints: Constraints) -> None:
    lo, hi = constraints.min_weight, constraints.max_weight
    if lo > hi:
        raise InfeasibleError(
            f"The minimum weight ({lo:.1%}) is above the maximum weight ({hi:.1%})."
        )
    if n_assets * hi < 1 - _FEASIBILITY_TOLERANCE:
        raise InfeasibleError(
            f"A {hi:.1%} maximum across {n_assets} assets only adds up to {n_assets * hi:.1%},"
            " and weights must total 100%."
        )
    if n_assets * lo > 1 + _FEASIBILITY_TOLERANCE:
        raise InfeasibleError(
            f"A {lo:.1%} minimum across {n_assets} assets adds up to {n_assets * lo:.1%},"
            " and weights must total 100%."
        )


def max_feasible_return(mu: pd.Series, constraints: Constraints) -> float:
    """The highest return the bounds allow: fill the best assets first."""
    weights = pd.Series(constraints.min_weight, index=mu.index, dtype=float)
    remaining = 1 - weights.sum()
    room = constraints.max_weight - constraints.min_weight
    for ticker in mu.sort_values(ascending=False).index:
        add = min(room, remaining)
        weights[ticker] += add
        remaining -= add
    return float(weights @ mu)


def _solve(mu, cov, constraints: Constraints, method: str, **kwargs) -> pd.Series:
    frontier = EfficientFrontier(
        mu, cov, weight_bounds=(constraints.min_weight, constraints.max_weight)
    )
    try:
        getattr(frontier, method)(**kwargs)
    except (OptimizationError, ValueError, ArithmeticError, cp.error.SolverError) as exc:
        raise OptimizationFailed(f"{method}: {exc}") from exc

    raw = np.asarray(frontier.weights, dtype=float)
    if not np.isfinite(raw).all():
        raise OptimizationFailed(f"{method}: the solver returned invalid weights.")
    # Solvers land slightly outside the bounds, so tidy up rounding noise.
    tidied = np.clip(raw, constraints.min_weight, constraints.max_weight)
    return pd.Series(tidied / tidied.sum(), index=mu.index)


def min_vol_weights(mu: pd.Series, cov: pd.DataFrame, constraints: Constraints) -> pd.Series:
    return _solve(mu, cov, constraints, "min_volatility")


def max_sharpe_weights(
    mu: pd.Series, cov: pd.DataFrame, risk_free: float, constraints: Constraints
) -> pd.Series:
    best = max_feasible_return(mu, constraints)
    if best <= risk_free:
        raise InfeasibleError(
            f"No allowed portfolio is expected to beat the {risk_free:.2%} risk-free rate"
            f" (the best is {best:.2%}), so a maximum Sharpe portfolio doesn't exist."
        )
    return _solve(mu, cov, constraints, "max_sharpe", risk_free_rate=risk_free)


def _portfolio(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, risk_free: float) -> Portfolio:
    expected = float(weights @ mu)
    volatility = float(np.sqrt(weights @ cov @ weights))
    sharpe = (expected - risk_free) / volatility if volatility > 0 else None
    return Portfolio({t: float(w) for t, w in weights.items()}, expected, volatility, sharpe)


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free: float,
    constraints: Constraints,
    low_return: float,
    points: int = 30,
) -> list[Portfolio]:
    # The very top of the range is only just feasible, so stay a hair below it.
    high_return = max_feasible_return(mu, constraints) - 1e-6
    if high_return <= low_return:
        return []
    portfolios = []
    for target in np.linspace(low_return, high_return, points):
        try:
            weights = _solve(mu, cov, constraints, "efficient_return", target_return=float(target))
        except OptimizationFailed:
            continue  # A gap in the curve is better than failing the whole request.
        portfolios.append(_portfolio(weights, mu, cov, risk_free))
    return portfolios


def optimize(
    prices: pd.DataFrame,
    risk_free: float = 0.04,
    constraints: Constraints = Constraints(),
    frontier_points: int = 30,
) -> OptimizationResult:
    check_feasible(prices.shape[1], constraints)
    mu, cov = estimate(prices)

    warnings = []
    min_vol = _portfolio(min_vol_weights(mu, cov, constraints), mu, cov, risk_free)
    max_sharpe = None
    try:
        max_sharpe = _portfolio(
            max_sharpe_weights(mu, cov, risk_free, constraints), mu, cov, risk_free
        )
    except InfeasibleError as exc:
        warnings.append(str(exc))
    except OptimizationFailed:
        warnings.append("The solver couldn't find a maximum Sharpe portfolio for these inputs.")

    frontier = efficient_frontier(
        mu, cov, risk_free, constraints, min_vol.expected_return, frontier_points
    )
    assets = {
        ticker: {
            "expected_return": float(mu[ticker]),
            "volatility": float(np.sqrt(cov.loc[ticker, ticker])),
        }
        for ticker in prices.columns
    }
    return OptimizationResult(max_sharpe, min_vol, frontier, assets, warnings)
