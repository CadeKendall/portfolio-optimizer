"""Request and response models for the HTTP API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .backtest import BacktestResult
from .data import TICKER_RE
from .optimizer import Portfolio


def _clean_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if not TICKER_RE.match(ticker):
        raise ValueError(f"{value!r} isn't a valid ticker")
    return ticker


class CommonRequest(BaseModel):
    tickers: list[str] = Field(min_length=2, max_length=30)
    source: Literal["yahoo", "upload"] = "yahoo"
    upload_id: str | None = None
    end: date | None = None
    risk_free: float = Field(0.04, ge=-0.05, le=1.0)
    min_weight: float = Field(0.0, ge=0.0, le=1.0)
    max_weight: float = Field(1.0, gt=0.0, le=1.0)

    @field_validator("tickers")
    @classmethod
    def clean_tickers(cls, values: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(_clean_ticker(v) for v in values))
        if len(cleaned) < 2:
            raise ValueError("enter at least two different tickers")
        return cleaned

    @model_validator(mode="after")
    def upload_needs_id(self):
        if self.source == "upload" and not self.upload_id:
            raise ValueError("upload a CSV first, or switch the data source to Yahoo")
        return self


class OptimizeRequest(CommonRequest):
    lookback_years: float = Field(5.0, ge=0.5, le=30)
    frontier_points: int = Field(30, ge=5, le=100)


class BacktestRequest(CommonRequest):
    years: float = Field(10.0, ge=1, le=40)
    lookback_years: float = Field(3.0, ge=0.5, le=10)
    frequency: Literal["monthly", "quarterly"] = "monthly"
    cost_bps: float = Field(5.0, ge=0, le=500)
    benchmark: str | None = "SPY"

    @field_validator("benchmark")
    @classmethod
    def clean_benchmark(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return _clean_ticker(value)


class DataInfo(BaseModel):
    start: date
    end: date
    rows: int
    tickers: list[str]
    warnings: list[str]


class AssetStats(BaseModel):
    expected_return: float
    volatility: float


class OptimizeResponse(BaseModel):
    data: DataInfo
    max_sharpe: Portfolio | None
    min_volatility: Portfolio
    frontier: list[Portfolio]
    assets: dict[str, AssetStats]
    warnings: list[str]


class BacktestResponse(BaseModel):
    data: DataInfo
    result: BacktestResult


class UploadResponse(BaseModel):
    upload_id: str
    tickers: list[str]
    start: date
    end: date
    rows: int
