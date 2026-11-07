import numpy as np
import pandas as pd
import pytest


def make_prices(days=1500, seed=7, start="2015-01-01"):
    """Correlated random-walk prices for four assets with different risk and return."""
    rng = np.random.default_rng(seed)
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    annual_mean = np.array([0.12, 0.06, 0.09, 0.03])
    annual_vol = np.array([0.25, 0.10, 0.18, 0.05])
    correlation = np.array(
        [
            [1.0, 0.2, 0.5, 0.0],
            [0.2, 1.0, 0.1, 0.2],
            [0.5, 0.1, 1.0, 0.0],
            [0.0, 0.2, 0.0, 1.0],
        ]
    )
    daily_cov = np.outer(annual_vol, annual_vol) * correlation / 252
    returns = rng.multivariate_normal(annual_mean / 252, daily_cov, size=days)
    index = pd.bdate_range(start, periods=days)
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=index, columns=tickers)


@pytest.fixture
def prices():
    return make_prices()
