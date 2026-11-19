"""Request validation: bad input should give a clear 422 before any price data is fetched."""

from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import create_app
from app.schemas import BacktestRequest, OptimizeRequest
from tests.conftest import make_prices

ROUTES = ["/api/optimize", "/api/backtest"]


class RecordingDownloader:
    """Offline stand-in for Yahoo that serves synthetic prices and records every call."""

    def __init__(self):
        days = 3200
        start = pd.bdate_range(end=date.today(), periods=days)[0]
        universe = make_prices(days=days, start=start)
        self.universe = universe.rename(
            columns={"AAA": "SPY", "BBB": "TLT", "CCC": "QQQ", "DDD": "SHY"}
        )
        self.calls = []

    def __call__(self, tickers, start, end):
        self.calls.append((list(tickers), start, end))
        present = [t for t in tickers if t in self.universe.columns]
        return self.universe.loc[pd.Timestamp(start) : pd.Timestamp(end), present]


@pytest.fixture
def downloader():
    return RecordingDownloader()


@pytest.fixture
def client(tmp_path, downloader):
    return TestClient(create_app(tmp_path, downloader))


def messages(response):
    """All validation messages from a FastAPI 422 body, joined for easy matching."""
    detail = response.json()["detail"]
    if isinstance(detail, str):
        return detail
    return " | ".join(d["msg"] for d in detail)


# --- Ticker cleanup ---------------------------------------------------------


@pytest.mark.parametrize("model", [OptimizeRequest, BacktestRequest])
def test_tickers_are_stripped_uppercased_and_deduplicated_in_order(model):
    req = model(tickers=["  spy ", "tlt", "SPY", " Tlt", "qqq"])
    assert req.tickers == ["SPY", "TLT", "QQQ"]


def test_benchmark_is_cleaned_and_blank_means_none():
    assert BacktestRequest(tickers=["SPY", "TLT"], benchmark=" qqq ").benchmark == "QQQ"
    assert BacktestRequest(tickers=["SPY", "TLT"], benchmark="   ").benchmark is None
    assert BacktestRequest(tickers=["SPY", "TLT"], benchmark=None).benchmark is None


def test_duplicates_that_leave_one_ticker_are_rejected():
    with pytest.raises(ValidationError, match="at least two different tickers"):
        OptimizeRequest(tickers=["spy", " SPY ", "Spy"])


@pytest.mark.parametrize("bad", ["", "   ", "SP Y", "SPY!", "A" * 16, "$SPY"])
def test_invalid_ticker_is_rejected_with_its_name(bad):
    with pytest.raises(ValidationError, match="isn't a valid ticker"):
        OptimizeRequest(tickers=["TLT", bad])


@pytest.mark.parametrize("route", ROUTES)
def test_api_cleans_tickers(client, route):
    response = client.post(route, json={"tickers": [" spy", "tlt ", "SPY", "Tlt", "qqq"]})
    assert response.status_code == 200, response.text
    assert response.json()["data"]["tickers"] == ["SPY", "TLT", "QQQ"]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "tickers, expected",
    [
        (["spy", "SPY "], "at least two different tickers"),
        (["SPY", "bad ticker"], "'bad ticker' isn't a valid ticker"),
        (["SPY"], "at least 2 items"),
    ],
)
def test_api_ticker_errors_are_clear_422s(client, downloader, route, tickers, expected):
    response = client.post(route, json={"tickers": tickers})
    assert response.status_code == 422
    assert expected in messages(response)
    assert downloader.calls == []


def test_api_invalid_benchmark_is_a_clear_422(client, downloader):
    response = client.post("/api/backtest", json={"tickers": ["SPY", "TLT"], "benchmark": "no way"})
    assert response.status_code == 422
    assert "'no way' isn't a valid ticker" in messages(response)
    assert downloader.calls == []


# --- Min / max weights ------------------------------------------------------


def test_min_equal_to_max_is_allowed():
    req = OptimizeRequest(tickers=["SPY", "TLT"], min_weight=0.5, max_weight=0.5)
    assert (req.min_weight, req.max_weight) == (0.5, 0.5)


@pytest.mark.parametrize("model", [OptimizeRequest, BacktestRequest])
def test_min_above_max_is_rejected(model):
    with pytest.raises(ValidationError, match=r"minimum weight \(40\.0%\) is above the maximum weight \(30\.0%\)"):
        model(tickers=["SPY", "TLT", "QQQ", "SHY"], min_weight=0.4, max_weight=0.3)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "weights, expected",
    [
        ({"min_weight": 0.4, "max_weight": 0.3}, "minimum weight (40.0%) is above the maximum weight (30.0%)"),
        ({"min_weight": -0.1}, "greater than or equal to 0"),
        ({"min_weight": 1.5}, "less than or equal to 1"),
        ({"max_weight": 0}, "greater than 0"),
        ({"max_weight": -0.2}, "greater than 0"),
        ({"max_weight": 1.01}, "less than or equal to 1"),
    ],
)
def test_api_bad_weights_are_clear_422s_before_fetching(client, downloader, route, weights, expected):
    body = {"tickers": ["SPY", "TLT", "QQQ", "SHY"], **weights}
    response = client.post(route, json=body)
    assert response.status_code == 422
    assert expected in messages(response)
    assert downloader.calls == []


@pytest.mark.parametrize("route", ROUTES)
def test_api_weights_that_cannot_sum_to_one_are_clear_422s(client, route):
    # Valid on their own, but infeasible for this many assets: caught after loading data.
    too_low = client.post(route, json={"tickers": ["SPY", "TLT", "QQQ"], "max_weight": 0.2})
    assert too_low.status_code == 422
    assert "20.0% maximum across 3 assets" in messages(too_low)

    too_high = client.post(route, json={"tickers": ["SPY", "TLT", "QQQ"], "min_weight": 0.4})
    assert too_high.status_code == 422
    assert "40.0% minimum across 3 assets" in messages(too_high)


# --- Dates ------------------------------------------------------------------
# The API has no start date field: the start is derived as `end` minus the
# lookback (optimize) or `years + lookback_years` (backtest). So "end before
# start" can't be sent directly; the reachable cases are an unparseable end date
# and an end date so early that the derived window holds no prices.


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("end", ["not-a-date", "2024-13-01", "2024-02-30"])
def test_api_malformed_end_date_is_a_clear_422(client, downloader, route, end):
    response = client.post(route, json={"tickers": ["SPY", "TLT"], "end": end})
    assert response.status_code == 422
    assert "valid date" in messages(response)
    assert downloader.calls == []


@pytest.mark.parametrize("route", ROUTES)
def test_api_end_date_before_any_data_is_a_clear_422(client, route):
    response = client.post(route, json={"tickers": ["SPY", "TLT"], "end": "1990-01-01"})
    assert response.status_code == 422
    assert "No price data" in messages(response)


def test_end_date_is_parsed_and_window_is_derived_from_it(client, downloader):
    response = client.post(
        "/api/optimize", json={"tickers": ["SPY", "TLT"], "end": "2024-06-28", "lookback_years": 2}
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert date.fromisoformat(data["end"]) <= date(2024, 6, 28)
    assert date.fromisoformat(data["start"]) >= date(2022, 6, 27)
    assert date.fromisoformat(data["start"]) < date.fromisoformat(data["end"])
