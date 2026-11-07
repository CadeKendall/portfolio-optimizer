"""Price data: sources, the SQLite cache, CSV parsing and date alignment."""

from __future__ import annotations

import hashlib
import io
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

MIN_ROWS = 60
CACHE_TTL_SECONDS = 12 * 60 * 60
TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")
UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{16}$")


class DataError(ValueError):
    """Price data is missing or unusable. The message is shown to the user."""


@dataclass
class Fetched:
    prices: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


class PriceSource(Protocol):
    def default_end(self) -> date: ...

    def get_prices(self, tickers: list[str], start: date, end: date) -> Fetched: ...


Downloader = Callable[[list[str], date, date], pd.DataFrame]


def yahoo_download(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Download dividend- and split-adjusted closes. Yahoo's `end` is exclusive."""
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    index = pd.to_datetime(close.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    close.index = index.normalize()
    close.columns = [str(c) for c in close.columns]
    return close.dropna(axis=1, how="all")


class YFinanceSource:
    """Yahoo prices with a per-ticker SQLite cache.

    Adjusted closes are rescaled whenever a dividend is paid, so prices fetched at
    different times can't be mixed. A refresh therefore replaces a ticker's whole
    history rather than appending to it.
    """

    def __init__(self, cache_path: Path, downloader: Downloader = yahoo_download):
        self.cache_path = Path(cache_path)
        self.downloader = downloader
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS prices (ticker TEXT, date TEXT, close REAL,"
                " PRIMARY KEY (ticker, date))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS coverage (ticker TEXT PRIMARY KEY, start TEXT,"
                " end TEXT, fetched_at REAL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.cache_path, timeout=30)

    def default_end(self) -> date:
        return date.today()

    def get_prices(self, tickers: list[str], start: date, end: date) -> Fetched:
        with self._connect() as db:
            coverage = {
                row[0]: (date.fromisoformat(row[1]), date.fromisoformat(row[2]), row[3])
                for row in db.execute(
                    f"SELECT ticker, start, end, fetched_at FROM coverage"
                    f" WHERE ticker IN ({','.join('?' * len(tickers))})",
                    tickers,
                )
            }

        now = time.time()
        stale = []
        for ticker in tickers:
            cov = coverage.get(ticker)
            is_fresh = (
                cov is not None
                and cov[0] <= start
                and cov[1] >= end
                and now - cov[2] < CACHE_TTL_SECONDS
            )
            if not is_fresh:
                stale.append(ticker)

        warnings: list[str] = []
        if stale:
            warnings += self._refresh(stale, start, end, coverage, now)
        return Fetched(self._read(tickers, start, end), warnings)

    def _refresh(self, tickers, start, end, coverage, now) -> list[str]:
        # Widen to anything already cached, because the old rows get replaced.
        fetch_start = min([start] + [coverage[t][0] for t in tickers if t in coverage])
        fetch_end = max([end] + [coverage[t][1] for t in tickers if t in coverage])
        try:
            frame = self.downloader(tickers, fetch_start, fetch_end)
        except Exception:  # yfinance raises many unrelated types during outages
            frame = pd.DataFrame()

        warnings = []
        with self._connect() as db:
            for ticker in tickers:
                series = frame[ticker].dropna() if ticker in frame.columns else None
                if series is None or series.empty:
                    if ticker in coverage:
                        warnings.append(
                            f"{ticker}: couldn't refresh from Yahoo, using cached prices"
                            f" through {coverage[ticker][1].isoformat()}."
                        )
                    continue
                db.execute("DELETE FROM prices WHERE ticker = ?", (ticker,))
                db.executemany(
                    "INSERT INTO prices VALUES (?, ?, ?)",
                    [(ticker, d.date().isoformat(), float(v)) for d, v in series.items()],
                )
                db.execute(
                    "INSERT OR REPLACE INTO coverage VALUES (?, ?, ?, ?)",
                    (ticker, fetch_start.isoformat(), fetch_end.isoformat(), now),
                )
        return warnings

    def _read(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        with self._connect() as db:
            rows = pd.read_sql_query(
                f"SELECT ticker, date, close FROM prices"
                f" WHERE ticker IN ({','.join('?' * len(tickers))}) AND date >= ? AND date <= ?",
                db,
                params=[*tickers, start.isoformat(), end.isoformat()],
            )
        if rows.empty:
            return pd.DataFrame()
        frame = rows.pivot(index="date", columns="ticker", values="close")
        frame.index = pd.to_datetime(frame.index)
        frame.columns.name = None
        return frame[[t for t in tickers if t in frame.columns]].sort_index()


class CsvSource:
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices

    def default_end(self) -> date:
        return self.prices.index.max().date()

    def get_prices(self, tickers: list[str], start: date, end: date) -> Fetched:
        present = [t for t in tickers if t in self.prices.columns]
        sliced = self.prices.loc[pd.Timestamp(start) : pd.Timestamp(end), present]
        return Fetched(sliced)


def parse_csv(raw: bytes) -> pd.DataFrame:
    """Parse a wide CSV with a `date` column and one price column per ticker."""
    try:
        header = pd.read_csv(io.BytesIO(raw), header=None, nrows=1, dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataError(f"Couldn't read the CSV: {exc}") from exc

    names = [str(c).strip().upper() if pd.notna(c) else "" for c in header.iloc[0]]
    if "DATE" not in names:
        raise DataError("The CSV needs a column named 'date'.")
    tickers = [n for n in names if n != "DATE"]
    if not tickers:
        raise DataError("The CSV needs at least one price column.")
    bad = [n for n in tickers if not TICKER_RE.match(n)]
    if bad:
        raise DataError(f"Invalid ticker column names: {', '.join(map(repr, bad))}.")
    if len(set(names)) != len(names):
        raise DataError("The CSV has duplicate column names.")

    try:
        frame = pd.read_csv(io.BytesIO(raw), header=None, skiprows=1, names=names, dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataError(f"Couldn't read the CSV: {exc}") from exc
    if frame.empty:
        raise DataError("The CSV has no data rows.")

    try:
        dates = pd.to_datetime(frame["DATE"].str.strip(), errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataError(f"The CSV has an unreadable date: {exc}") from exc
    if dates.isna().any():
        raise DataError("The CSV has rows with no date.")
    if dates.duplicated().any():
        duplicate = dates[dates.duplicated()].iloc[0].date()
        raise DataError(f"The CSV has more than one row for {duplicate}.")

    prices = pd.DataFrame(index=pd.DatetimeIndex(dates.dt.normalize(), name=None))
    for ticker in tickers:
        original = frame[ticker].str.strip().replace("", np.nan).reset_index(drop=True)
        numeric = pd.to_numeric(original, errors="coerce")
        if (numeric.isna() & original.notna()).any():
            raise DataError(f"Column {ticker} has values that aren't numbers.")
        values = numeric.to_numpy(dtype=float)
        if np.isinf(values).any() or (values[~np.isnan(values)] <= 0).any():
            raise DataError(f"Column {ticker} has prices that aren't positive finite numbers.")
        prices[ticker] = values
    return prices.sort_index()


class UploadStore:
    """Keeps uploaded prices on disk so they survive a server restart."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, raw: bytes) -> tuple[str, pd.DataFrame]:
        prices = parse_csv(raw)
        upload_id = hashlib.sha256(raw).hexdigest()[:16]
        prices.rename_axis("date").to_csv(self.directory / f"{upload_id}.csv")
        return upload_id, prices

    def load(self, upload_id: str) -> pd.DataFrame:
        path = self.directory / f"{upload_id}.csv"
        if not UPLOAD_ID_RE.match(upload_id) or not path.exists():
            raise DataError("That uploaded file is no longer available. Please upload it again.")
        return parse_csv(path.read_bytes())


@dataclass
class Aligned:
    prices: pd.DataFrame
    warnings: list[str]
    unknown: list[str]

    @property
    def start(self) -> date:
        return self.prices.index[0].date()

    @property
    def end(self) -> date:
        return self.prices.index[-1].date()


def align_prices(prices: pd.DataFrame, requested: list[str], min_rows: int = MIN_ROWS) -> Aligned:
    """Restrict tickers to their shared date range and to dates all of them traded.

    Dates missing for any ticker are dropped rather than forward-filled. A fill would
    invent zero-return days, which understates volatility and correlation.
    """
    usable = prices.dropna(axis=1, how="all") if not prices.empty else prices
    unknown = [t for t in requested if t not in usable.columns]
    known = [t for t in requested if t in usable.columns]
    if not known:
        raise DataError(f"No price data found for: {', '.join(requested)}.")

    warnings = []
    if unknown:
        warnings.append(f"No price data found for {', '.join(unknown)}, so excluded.")

    usable = usable[known]
    firsts = {t: usable[t].first_valid_index() for t in known}
    lasts = {t: usable[t].last_valid_index() for t in known}
    shared_start, shared_end = max(firsts.values()), min(lasts.values())
    if shared_start > shared_end:
        raise DataError("These tickers have no dates in common.")

    if shared_start > min(firsts.values()):
        limiting = [t for t in known if firsts[t] == shared_start]
        warnings.append(
            f"{', '.join(limiting)} only has data from {shared_start.date()},"
            " which shortens the range for every ticker."
        )
    if shared_end < max(lasts.values()):
        limiting = [t for t in known if lasts[t] == shared_end]
        warnings.append(
            f"{', '.join(limiting)} only has data through {shared_end.date()},"
            " which shortens the range for every ticker."
        )

    in_range = usable.loc[shared_start:shared_end]
    complete = in_range.dropna(how="any")
    dropped = len(in_range) - len(complete)
    if dropped:
        warnings.append(f"Dropped {dropped} dates where at least one ticker had no price.")
    if len(complete) < min_rows:
        raise DataError(
            f"Only {len(complete)} shared trading days are available, and at least"
            f" {min_rows} are needed."
        )
    return Aligned(complete, warnings, unknown)
