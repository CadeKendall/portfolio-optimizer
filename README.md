# Portfolio Optimizer

A personal web app for mean-variance portfolio optimization and walk-forward backtesting.
It runs locally and listens on `127.0.0.1` only.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000.

Run the tests with `.venv/bin/python -m pytest`. They use synthetic prices and need no network access.

## What it does

**Optimize** estimates annualized arithmetic mean returns and Ledoit-Wolf shrinkage
covariance over a lookback window. It returns:
- the maximum Sharpe portfolio,
- the minimum volatility portfolio,
- the efficient frontier, with each asset's weight kept between your min and max.

**Backtest** compares max Sharpe, min volatility, equal weight and a benchmark over a test period.
- At each monthly or quarterly rebalance, it re-estimates using only prices up to the signal date, then trades at the **next** day's close, so there is no lookahead.
- Costs are charged as basis points on traded notional.
- If a rebalance can't be solved, the strategy keeps its previous weights, and the page lists that date. If the first rebalance fails, the strategy starts at equal weight.

## Data

- **Yahoo Finance** (via `yfinance`) supplies dividend- and split-adjusted closes, cached in `data/prices.db` for 12 hours.
  - Adjusted prices change whenever a dividend is paid, so each refresh replaces a ticker's whole history instead of appending to it.
  - If Yahoo is unavailable, the app uses cached prices and shows a warning.
- **CSV upload** takes a wide file: a `date` column plus one adjusted-price column per ticker. Uploads are saved in `data/uploads/`.
- Tickers are restricted to their shared date range. Dates where any ticker has no price are dropped, not filled. The page reports both, and names any ticker that shortens the range.

Set `PO_DATA_DIR` to keep the cache somewhere else.

## Layout

| Path | Purpose |
|---|---|
| `app/data.py` | Price sources, SQLite cache, CSV parsing, alignment |
| `app/optimizer.py` | Estimators, max Sharpe, min volatility, frontier |
| `app/backtest.py` | Walk-forward engine and metrics |
| `app/main.py`, `app/schemas.py` | FastAPI routes and request/response models |
| `static/` | Single page: plain JS and vendored Plotly 2.35.2 |

## Caveats

- Expected returns estimated from history are very noisy, and max Sharpe weights are sensitive to them.
- Backtests carry survivorship bias because you pick tickers that exist today.
- `yfinance` is an unofficial scraper and can break when Yahoo changes. Use CSV upload as the fallback.
- Nothing here is investment advice.
