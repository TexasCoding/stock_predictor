# src/stock_trader/yahoo_finance/yahoo_financials.py
# This module contains the YahooFinancials class, which is used to retrieve financial data for a given stock symbol.
from stock_predictor.global_settings import check_for_float_or_int
import pandas as pd

import yfinance as yf

from stock_predictor.stock_module.stock_base import StockBase


class StockFinancials(StockBase):
    def __init__(self):
        super().__init__()
        self.check_for_float_or_int = check_for_float_or_int

    ##############################
    # Compute margin
    ##############################
    def compute_margin(
        self, df: pd.DataFrame, numerator: str, denominator: str
    ) -> pd.Series:
        """
        Computes the margin for a given DataFrame using the specified numerator and denominator columns.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            numerator (str): The column name for the numerator.
            denominator (str): The column name for the denominator.

        Returns:
            pd.Series: The computed margin as a Series.

        """
        pd.set_option("future.no_silent_downcasting", True)

        if numerator in df and denominator in df:
            return (df[numerator] / df[denominator] * 100).replace(
                {float("inf"): 0.0, float("-inf"): 0.0, pd.NA: 0.0}
            )
        return pd.Series(0.0, index=df.index)

    ##############################
    # Get financials
    ##############################
    def get_financials(self, symbol: str, ticker: yf.Ticker) -> pd.DataFrame:
        """
        Retrieves financial data for a given stock symbol.

        Args:
            symbol (str): The stock symbol to retrieve financial data for.

        Returns:
            pd.DataFrame: A DataFrame containing the financial data for the stock symbol.

        Raises:
            Exception: If there is an error processing the financial data.

        """
        try:
            # ticker = self.get_ticker(symbol)

            financials = (
                ticker.financials.transpose().iloc[::-1].copy()
            )  # transpose and reverse the order
            df = pd.DataFrame(index=financials.index)
            df["market_cap"] = int(ticker.info.get("marketCap", 0))
            df["gross_margin_pct"] = self.compute_margin(
                financials, "Gross Profit", "Total Revenue"
            )
            df["net_margin_pct"] = self.compute_margin(
                financials, "Net Income", "Total Revenue"
            )

            info = ticker.info

            df["symbol"] = symbol
            df["name"] = info.get("shortName", "")
            df["trailing_pe"] = (
                info.get("trailingPE", 0)
                if self.check_for_float_or_int(info.get("trailingPE", 0))
                else 0
            )
            df["forward_pe"] = (
                info.get("forwardPE", 0)
                if self.check_for_float_or_int(info.get("forwardPE", 0))
                else 0
            )
            df["industry"] = info.get("industry", "")

            return df.tail(1).dropna()

        except Exception as e:
            self.logger.info(f"Error processing financials for {symbol}: {e}")
            return pd.DataFrame()
