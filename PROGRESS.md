# Progress log

(Claude adds a short entry here after each task.)

- 2026-11-17 â€” README: added Windows (PowerShell) setup/run/test commands beside the Mac/Linux ones, and a separate "Test it" section. Docs only, no tests needed; suite passes (58). Windows run uses `python -m uvicorn` (works without relying on the uvicorn.exe shim); I verified the module imports but did not launch the server.
- 2026-11-17 — Added tests/test_metrics.py: hand-checked CAGR, volatility, Sharpe, max drawdown and turnover on a 5-point, exactly-4-year series, plus edge cases (zero vol, no trades, losses, zero-length span). No app code changed; suite passes (64).
  Look at: constant non-power-of-two returns (+10%/step) give vol ~1.8e-15 so Sharpe is ~1.4e16 instead of None. I tested with an exact series and added a TODO under "Added by Claude" to fix it.
