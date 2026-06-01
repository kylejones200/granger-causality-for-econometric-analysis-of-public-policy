"""Shell vs Brent proxy: correlation, OLS, stationarity, and Granger causality.

Consolidated from the 2025-03-01 notebook. Order of operations:
  configuration → data download → save → correlation / OLS → stationarity
  (levels and returns) → Granger on returns (both directions) → figures.

Uses yfinance only (no third-party API keys). Tickers: SHEL (Shell) and BNO
(Brent oil fund proxy), matching the original yfinance-based cells.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def check_stationarity(series: pd.Series, name: str) -> None:
    s = series.dropna()
    result = adfuller(s)
    logging.info("%s: ADF statistic = %.4f, p-value = %.4f", name, result[0], result[1])
    if result[1] > 0.05:
        logging.info("  -> do not reject unit root at 5%% (treat as non-stationary).")
    else:
        logging.info("  -> reject unit root at 5%% (treat as stationary).")


def download_prices(
    tickers: Tuple[str, str], start: datetime.datetime, end: datetime.datetime
) -> pd.DataFrame:
    """Fetch daily close prices and return a frame with levels and returns."""
    joined = " ".join(tickers)
    raw = None
    for attempt in range(MAX_RETRIES):
        try:
            raw = yf.download(joined, start=start, end=end, progress=False)
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            logging.warning(
                "Download attempt %s failed (%s); sleeping %ss.",
                attempt + 1,
                e,
                RETRY_SLEEP_S,
            )
            time.sleep(RETRY_SLEEP_S)
    if raw is None:
        raise RuntimeError("yfinance download failed")
    close = raw["Close"].copy()
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = ["Shell_Close", "Brent_Close"]
    else:
        if getattr(close, "ndim", 1) == 1 or close.shape[1] != 2:
            raise ValueError("Expected two tickers in Close data.")
        close = close.rename(
            columns={tickers[0]: "Shell_Close", tickers[1]: "Brent_Close"}
        )
    df = close.dropna()
    df["Shell_Returns"] = df["Shell_Close"].pct_change()
    df["Brent_Returns"] = df["Brent_Close"].pct_change()
    return df.dropna()


def prepare_end_date() -> None:
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=YEAR_LOOKBACK * 365)
    df = download_prices(TICKERS, start_date, end_date)
    df.to_csv("shell_brent_returns.csv")
    corr_levels = df[["Shell_Close", "Brent_Close"]].corr()
    logging.info("Correlation (levels):\n%s", corr_levels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_levels, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation heatmap: Shell vs Brent (levels)")
    plt.tight_layout()
    plt.show()


def prepare_x() -> None:
    X = sm.add_constant(df["Brent_Close"])
    ols = sm.OLS(df["Shell_Close"], X).fit()
    logging.info("OLS (Shell_Close ~ Brent_Close):\n%s", ols.summary())
    logging.info("\nStationarity — levels")
    check_stationarity(df["Shell_Close"], "Shell_Close")
    check_stationarity(df["Brent_Close"], "Brent_Close")
    logging.info("\nStationarity — returns")
    check_stationarity(df["Shell_Returns"], "Shell_Returns")
    check_stationarity(df["Brent_Returns"], "Brent_Returns")
    logging.info(
        "\nGranger causality (returns): does Brent Granger-cause Shell?\n  statsmodels tests whether column 1 helps predict column 0."
    )
    grangercausalitytests(df[["Shell_Returns", "Brent_Returns"]], maxlag=MAX_LAG)
    logging.info("\nGranger causality (returns): does Shell Granger-cause Brent?")
    grangercausalitytests(df[["Brent_Returns", "Shell_Returns"]], maxlag=MAX_LAG)
    logging.info("\nKey statistics")
    logging.info("R-squared (OLS levels): %.4f", ols.rsquared)
    logging.info("Correlation (levels): %.4f", corr_levels.iloc[0, 1])
    ret_corr = df[["Shell_Returns", "Brent_Returns"]].corr().iloc[0, 1]
    logging.info("Correlation (returns): %.4f", ret_corr)
    logging.info(
        "Daily vol (std of returns): Shell %.2f%%, Brent %.2f%%",
        df["Shell_Returns"].std() * 100,
        df["Brent_Returns"].std() * 100,
    )
    df[["Shell_Close", "Brent_Close"]].plot(figsize=(12, 6))
    plt.title("Shell vs Brent (close)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()


def figure() -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Brent_Close"], df["Shell_Close"], alpha=0.5)
    plt.plot(df["Brent_Close"], ols.predict(X), color="red", linewidth=2)
    plt.xlabel("Brent (proxy) close")
    plt.ylabel("Shell close")
    plt.title("Shell vs Brent with OLS line")
    plt.tight_layout()
    plt.show()


def figure_2() -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Shell_Returns"], label="Shell returns", alpha=0.7)
    plt.plot(df.index, df["Brent_Returns"], label="Brent returns", alpha=0.7)
    plt.title("Daily returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def figure_3() -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Brent_Returns"], df["Shell_Returns"], alpha=0.5)
    plt.xlabel("Brent returns")
    plt.ylabel("Shell returns")
    plt.title("Shell returns vs Brent returns")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    prepare_end_date()
    prepare_x()
    figure()
    figure_2()
    figure_3()


if __name__ == "__main__":
    main()
