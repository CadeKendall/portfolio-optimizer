from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.data import (
    CACHE_TTL_SECONDS,
    CsvSource,
    DataError,
    UploadStore,
    YFinanceSource,
    align_prices,
    parse_csv,
)

START, END = date(2024, 1, 1), date(2024, 12, 31)


def frame(tickers, multiplier=1.0, start="2024-01-02", days=100):
    index = pd.bdate_range(start, periods=days)
    return pd.DataFrame(
        {t: multiplier * (100 + np.arange(days, dtype=float)) for t in tickers}, index=index
    )


class FakeDownloader:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, tickers, start, end):
        self.calls.append((list(tickers), start, end))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result[[t for t in tickers if t in self.result.columns]]


def test_cache_hit_avoids_second_download(tmp_path):
    fake = FakeDownloader(frame(["SPY", "TLT"]))
    source = YFinanceSource(tmp_path / "c.db", fake)
    first = source.get_prices(["SPY", "TLT"], START, END)
    second = source.get_prices(["SPY", "TLT"], START, END)
    assert len(fake.calls) == 1
    pd.testing.assert_frame_equal(first.prices, second.prices, check_freq=False)
    assert first.prices.index[0] == pd.Timestamp("2024-01-02")


def test_expired_cache_replaces_history_instead_of_mixing_adjustments(tmp_path, monkeypatch):
    fake = FakeDownloader(frame(["SPY"]))
    source = YFinanceSource(tmp_path / "c.db", fake)
    source.get_prices(["SPY"], START, END)

    fake.result = frame(["SPY"], multiplier=0.5, days=50)  # A dividend rescales history.
    real_time = __import__("time").time
    monkeypatch.setattr("app.data.time.time", lambda: real_time() + CACHE_TTL_SECONDS + 1)
    refreshed = source.get_prices(["SPY"], START, END).prices

    assert len(fake.calls) == 2
    assert len(refreshed) == 50
    assert refreshed["SPY"].iloc[0] == 50.0


def test_outage_falls_back_to_cache_with_warning(tmp_path, monkeypatch):
    fake = FakeDownloader(frame(["SPY"]))
    source = YFinanceSource(tmp_path / "c.db", fake)
    source.get_prices(["SPY"], START, END)

    fake.result = RuntimeError("Yahoo is down")
    real_time = __import__("time").time
    monkeypatch.setattr("app.data.time.time", lambda: real_time() + CACHE_TTL_SECONDS + 1)
    result = source.get_prices(["SPY"], START, END)

    assert len(result.prices) == 100
    assert "couldn't refresh" in result.warnings[0]


def test_wider_request_refetches_and_keeps_order(tmp_path):
    fake = FakeDownloader(frame(["SPY", "TLT"]))
    source = YFinanceSource(tmp_path / "c.db", fake)
    source.get_prices(["SPY"], START, END)
    result = source.get_prices(["TLT", "SPY"], date(2023, 1, 1), END)
    assert fake.calls[1] == (["TLT", "SPY"], date(2023, 1, 1), END)
    assert list(result.prices.columns) == ["TLT", "SPY"]


def test_unknown_ticker_is_reported_not_fatal():
    aligned = align_prices(frame(["SPY", "TLT"]), ["SPY", "TLT", "NOPE"])
    assert aligned.unknown == ["NOPE"]
    assert list(aligned.prices.columns) == ["SPY", "TLT"]
    assert "NOPE" in aligned.warnings[0]


def test_all_unknown_raises():
    with pytest.raises(DataError, match="No price data"):
        align_prices(pd.DataFrame(), ["NOPE"])


def test_alignment_reports_the_ticker_that_shortens_the_range():
    prices = frame(["SPY", "NEW"])
    prices.loc[prices.index[:20], "NEW"] = np.nan
    aligned = align_prices(prices, ["SPY", "NEW"])
    assert aligned.start == prices.index[20].date()
    assert len(aligned.prices) == 80
    assert any("NEW only has data from" in w for w in aligned.warnings)


def test_alignment_drops_interior_gaps_instead_of_filling():
    prices = frame(["SPY", "TLT"])
    prices.loc[prices.index[50], "TLT"] = np.nan
    aligned = align_prices(prices, ["SPY", "TLT"])
    assert prices.index[50] not in aligned.prices.index
    assert aligned.prices.notna().all().all()
    assert any("Dropped 1 dates" in w for w in aligned.warnings)


def test_too_little_history_raises():
    with pytest.raises(DataError, match="Only 30 shared trading days"):
        align_prices(frame(["SPY"], days=30), ["SPY"])


def test_csv_parses_misaligned_dates_and_missing_values():
    raw = b"Date,spy,TLT\n2024-01-03,101,\n2024-01-02,100,50\n2024-01-04,,52\n2024-01-05,103,53\n"
    prices = parse_csv(raw)
    assert list(prices.columns) == ["SPY", "TLT"]
    assert prices.index[0] == pd.Timestamp("2024-01-02")  # sorted
    assert np.isnan(prices.loc["2024-01-03", "TLT"])
    aligned = align_prices(prices, ["SPY", "TLT"], min_rows=2)
    assert [d.day for d in aligned.prices.index] == [2, 5]


@pytest.mark.parametrize(
    "raw, message",
    [
        (b"", "read the CSV"),
        (b"SPY,TLT\n1,2\n", "named 'date'"),
        (b"date\n2024-01-01\n", "at least one price column"),
        (b"date,SPY\n", "no data rows"),
        (b"date,SPY,spy\n2024-01-01,1,2\n", "duplicate"),
        (b"date,SPY\nnot-a-date,1\n", "unreadable date"),
        (b"date,SPY\n2024-01-01,1\n2024-01-01,2\n", "more than one row"),
        (b"date,SPY\n2024-01-01,abc\n", "aren't numbers"),
        (b"date,SPY\n2024-01-01,-5\n", "positive"),
        (b"date,SPY\n2024-01-01,inf\n", "positive"),
        (b"date,bad ticker\n2024-01-01,1\n", "Invalid ticker"),
    ],
)
def test_csv_rejects_bad_files_with_specific_messages(raw, message):
    with pytest.raises(DataError, match=message):
        parse_csv(raw)


def test_upload_round_trip_and_path_traversal(tmp_path):
    store = UploadStore(tmp_path)
    upload_id, prices = store.save(b"date,SPY\n2024-01-02,100\n2024-01-03,101\n")
    pd.testing.assert_frame_equal(store.load(upload_id), prices, check_freq=False)
    with pytest.raises(DataError, match="no longer available"):
        store.load("../../etc/passwd")
    with pytest.raises(DataError, match="no longer available"):
        store.load("0" * 16)


def test_csv_source_slices_to_range_and_known_tickers():
    source = CsvSource(frame(["SPY", "TLT"]))
    result = source.get_prices(["SPY", "NOPE"], date(2024, 2, 1), date(2024, 2, 29))
    assert list(result.prices.columns) == ["SPY"]
    assert result.prices.index.min() >= pd.Timestamp("2024-02-01")
    assert result.prices.index.max() <= pd.Timestamp("2024-02-29")
    assert source.default_end() == frame(["SPY"]).index.max().date()
