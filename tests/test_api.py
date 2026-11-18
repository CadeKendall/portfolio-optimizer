from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import make_prices


@pytest.fixture
def client(tmp_path):
    days = 3200
    universe = make_prices(days=days, start=pd.bdate_range(end=date.today(), periods=days)[0])
    universe = universe.rename(columns={"AAA": "SPY", "BBB": "TLT", "CCC": "QQQ", "DDD": "SHY"})

    def fake_download(tickers, start, end):
        present = [t for t in tickers if t in universe.columns]
        return universe.loc[pd.Timestamp(start) : pd.Timestamp(end), present]

    return TestClient(create_app(tmp_path, fake_download))


def test_optimize_end_to_end(client):
    response = client.post(
        "/api/optimize",
        json={"tickers": ["spy", "TLT", "qqq", "SHY", "NOPE"], "risk_free": 0.0, "max_weight": 0.6},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["data"]["tickers"] == ["SPY", "TLT", "QQQ", "SHY"]
    assert any("NOPE" in w for w in body["data"]["warnings"])
    assert body["data"]["rows"] > 1000
    assert sum(body["max_sharpe"]["weights"].values()) == pytest.approx(1)
    assert max(body["min_volatility"]["weights"].values()) <= 0.6 + 1e-6
    assert len(body["frontier"]) > 10
    assert set(body["assets"]) == {"SPY", "TLT", "QQQ", "SHY"}


def test_backtest_end_to_end_with_separate_benchmark(client):
    response = client.post(
        "/api/backtest",
        json={"tickers": ["TLT", "QQQ", "SHY"], "benchmark": "SPY", "years": 8, "lookback_years": 2},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    names = [s["name"] for s in body["result"]["strategies"]]
    assert names == ["Max Sharpe", "Min volatility", "Equal weight", "SPY"]
    assert body["data"]["tickers"] == ["TLT", "QQQ", "SHY"]
    first_trade = date.fromisoformat(body["result"]["dates"][0])
    assert 7.5 < (date.today() - first_trade).days / 365.25 < 8.5


def test_backtest_with_unknown_benchmark_warns_and_continues(client):
    response = client.post(
        "/api/backtest", json={"tickers": ["TLT", "QQQ"], "benchmark": "NOPE", "years": 3}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [s["name"] for s in body["result"]["strategies"]][-1] == "Equal weight"
    assert any("NOPE" in w for w in body["data"]["warnings"])


def test_user_errors_return_422_with_message(client):
    infeasible = client.post("/api/optimize", json={"tickers": ["SPY", "TLT"], "max_weight": 0.3})
    assert infeasible.status_code == 422
    assert "30.0% maximum across 2 assets" in infeasible.json()["detail"]

    unknown = client.post("/api/optimize", json={"tickers": ["NOPE", "NADA"]})
    assert unknown.status_code == 422
    assert "No price data" in unknown.json()["detail"]

    one = client.post("/api/optimize", json={"tickers": ["SPY", "spy"]})
    assert one.status_code == 422

    no_upload = client.post("/api/optimize", json={"tickers": ["SPY", "TLT"], "source": "upload"})
    assert no_upload.status_code == 422


def test_upload_then_optimize_without_yahoo(client):
    prices = make_prices(days=400).rename(columns=str.lower)
    csv = prices.rename_axis("Date").to_csv().encode()
    upload = client.post("/api/prices/upload", files={"file": ("p.csv", csv, "text/csv")})
    assert upload.status_code == 200, upload.text
    meta = upload.json()
    assert meta["tickers"] == ["AAA", "BBB", "CCC", "DDD"]
    assert meta["rows"] == 400

    response = client.post(
        "/api/optimize",
        json={"tickers": ["AAA", "BBB", "CCC"], "source": "upload", "upload_id": meta["upload_id"]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["end"] == meta["end"]


def test_bad_upload_returns_message(client):
    response = client.post("/api/prices/upload", files={"file": ("p.csv", b"x,y\n1,2\n", "text/csv")})
    assert response.status_code == 422
    assert "named 'date'" in response.json()["detail"]


def test_serves_the_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Portfolio Optimizer" in response.text


def test_backtest_response_includes_annual_turnover(client):
    response = client.post(
        "/api/backtest",
        json={"tickers": ["TLT", "QQQ", "SHY"], "benchmark": "SPY", "years": 5, "lookback_years": 1},
    )
    assert response.status_code == 200, response.text
    by_name = {s["name"]: s["metrics"] for s in response.json()["result"]["strategies"]}
    for metrics in by_name.values():
        assert isinstance(metrics["annual_turnover"], float)
        assert metrics["annual_turnover"] >= 0
    assert by_name["SPY"]["annual_turnover"] == 0  # buy-and-hold benchmark never trades
    assert by_name["Equal weight"]["annual_turnover"] > 0  # monthly drift gets rebalanced


def test_backtest_inverse_volatility_is_opt_in(client):
    request = {"tickers": ["TLT", "QQQ", "SHY"], "benchmark": "SPY", "years": 5, "lookback_years": 1}
    default = client.post("/api/backtest", json=request)
    assert default.status_code == 200, default.text
    assert "Inverse volatility" not in [s["name"] for s in default.json()["result"]["strategies"]]

    response = client.post("/api/backtest", json={**request, "inverse_volatility": True})
    assert response.status_code == 200, response.text
    strategies = response.json()["result"]["strategies"]
    names = [s["name"] for s in strategies]
    assert names == ["Max Sharpe", "Min volatility", "Equal weight", "Inverse volatility", "SPY"]
    inv = strategies[3]
    assert len(inv["equity"]) == len(response.json()["result"]["dates"])
    assert set(inv["metrics"]) == set(strategies[2]["metrics"])
    assert inv["metrics"]["annual_turnover"] > 0
