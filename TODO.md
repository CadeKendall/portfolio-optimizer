# TODO

- [ ] README: add Windows (PowerShell) setup, run and test commands next to the existing Mac/Linux ones.
- [ ] Tests for compute_metrics: check every metric it returns against hand-calculated values on a tiny equity series (5-10 points).
- [ ] Tests for request validation in schemas.py: ticker cleanup (spaces, lowercase, duplicates), invalid min/max weights, end date before start date. Each should give a clear 422.
- [ ] Tests for efficient_frontier edge cases: two assets, two nearly identical assets, and bounds tight enough that only one portfolio is feasible.
- [ ] Test that monthly vs quarterly rebalancing produces the expected number of rebalance dates on a known date range.
- [ ] Add Sortino and Calmar ratios to backtest metrics, with hand-checked tests. Add them to the API response. Don't change the frontend yet.
- [ ] Add annualized turnover to backtest metrics, with a hand-checked test, and add it to the API response.
- [ ] Add an inverse-volatility (simple risk parity) strategy to the backtest alongside equal weight, with tests including the no-lookahead check.
- [ ] Add an endpoint that returns backtest equity curves as CSV, with an API test.
- [ ] Add docstrings to every public function in app/ that doesn't have one. No behavior changes.
- [ ] Add missing type hints in app/. No behavior changes.

## Added by Claude
