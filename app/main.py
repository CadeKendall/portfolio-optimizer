"""FastAPI app: the JSON API plus the static page."""

from __future__ import annotations

import csv
import io
import os
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .backtest import BacktestResult, run_backtest
from .data import (
    MIN_ROWS,
    Aligned,
    CsvSource,
    DataError,
    Downloader,
    PriceSource,
    UploadStore,
    YFinanceSource,
    align_prices,
    yahoo_download,
)
from .optimizer import Constraints, InfeasibleError, OptimizationFailed, TRADING_DAYS, optimize
from .schemas import (
    BacktestRequest,
    BacktestResponse,
    CommonRequest,
    DataInfo,
    OptimizeRequest,
    OptimizeResponse,
    UploadResponse,
)

ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def years_before(end: date, years: float) -> date:
    """The date `years` years before `end`, counting 365.25 days per year."""
    return end - timedelta(days=round(365.25 * years))


def create_app(data_dir: Path | None = None, downloader: Downloader = yahoo_download) -> FastAPI:
    """Build the app: API routes, error handling and the static page.

    `data_dir` holds the price cache and uploads (default: $PO_DATA_DIR, else
    ./data). `downloader` fetches Yahoo prices; tests pass an offline one.
    """
    data_dir = Path(data_dir or os.environ.get("PO_DATA_DIR", ROOT / "data"))
    yahoo = YFinanceSource(data_dir / "prices.db", downloader)
    uploads = UploadStore(data_dir / "uploads")
    app = FastAPI(title="Portfolio Optimizer")

    @app.exception_handler(DataError)
    @app.exception_handler(InfeasibleError)
    @app.exception_handler(OptimizationFailed)
    async def user_error(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    def source_for(req: CommonRequest) -> PriceSource:
        if req.source == "upload":
            return CsvSource(uploads.load(req.upload_id))
        return yahoo

    def load(source: PriceSource, tickers: list[str], start: date, end: date, min_rows: int) -> Aligned:
        fetched = source.get_prices(tickers, start, end)
        aligned = align_prices(fetched.prices, tickers, min_rows)
        aligned.warnings = fetched.warnings + aligned.warnings
        return aligned

    def info(aligned: Aligned, tickers: list[str]) -> DataInfo:
        return DataInfo(
            start=aligned.start,
            end=aligned.end,
            rows=len(aligned.prices),
            tickers=tickers,
            warnings=aligned.warnings,
        )

    def constraints(req: CommonRequest) -> Constraints:
        return Constraints(req.min_weight, req.max_weight)

    # Plain `def` routes run in a worker thread, so slow yfinance calls don't block the server.
    @app.post("/api/optimize", response_model=OptimizeResponse)
    def optimize_route(req: OptimizeRequest) -> OptimizeResponse:
        source = source_for(req)
        end = req.end or source.default_end()
        aligned = load(source, req.tickers, years_before(end, req.lookback_years), end, MIN_ROWS)
        if aligned.prices.shape[1] < 2:
            raise DataError("At least two tickers with price data are needed.")

        result = optimize(aligned.prices, req.risk_free, constraints(req), req.frontier_points)
        return OptimizeResponse(
            data=info(aligned, list(aligned.prices.columns)),
            max_sharpe=result.max_sharpe,
            min_volatility=result.min_volatility,
            frontier=result.frontier,
            assets=result.assets,
            warnings=result.warnings,
        )

    def backtest(req: BacktestRequest) -> tuple[Aligned, list[str], BacktestResult]:
        source = source_for(req)
        end = req.end or source.default_end()
        lookback = round(req.lookback_years * TRADING_DAYS)
        # Fetch the lookback window before the test period, so the first trade is about `years` ago.
        start = years_before(end, req.years + req.lookback_years)
        wanted = req.tickers + ([req.benchmark] if req.benchmark and req.benchmark not in req.tickers else [])
        aligned = load(source, wanted, start, end, lookback + 3)

        assets = [t for t in req.tickers if t in aligned.prices.columns]
        if len(assets) < 2:
            raise DataError("At least two tickers with price data are needed.")
        has_benchmark = req.benchmark is not None and req.benchmark in aligned.prices.columns
        benchmark = aligned.prices[req.benchmark] if has_benchmark else None

        result = run_backtest(
            aligned.prices[assets],
            benchmark=benchmark,
            benchmark_name=req.benchmark or "Benchmark",
            lookback=lookback,
            frequency=req.frequency,
            cost_bps=req.cost_bps,
            risk_free=req.risk_free,
            constraints=constraints(req),
            inverse_volatility=req.inverse_volatility,
        )
        return aligned, assets, result

    @app.post("/api/backtest", response_model=BacktestResponse)
    def backtest_route(req: BacktestRequest) -> BacktestResponse:
        aligned, assets, result = backtest(req)
        return BacktestResponse(data=info(aligned, assets), result=result)

    @app.post("/api/backtest/csv")
    def backtest_csv_route(req: BacktestRequest) -> Response:
        """Same request as /api/backtest; returns the equity curves as CSV.

        One row per date, one column per strategy (in response order). Values are
        growth of 1.0, written with full float precision.
        """
        _, _, result = backtest(req)
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["date"] + [s.name for s in result.strategies])
        for k, day in enumerate(result.dates):
            writer.writerow([day] + [repr(float(s.equity[k])) for s in result.strategies])
        return Response(
            content=out.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="backtest_equity.csv"'},
        )

    @app.post("/api/prices/upload", response_model=UploadResponse)
    async def upload_route(file: UploadFile) -> UploadResponse:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise DataError("The CSV is larger than 20 MB.")
        upload_id, prices = uploads.save(raw)
        return UploadResponse(
            upload_id=upload_id,
            tickers=list(prices.columns),
            start=prices.index[0].date(),
            end=prices.index[-1].date(),
            rows=len(prices),
        )

    app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
    return app


app = create_app()
