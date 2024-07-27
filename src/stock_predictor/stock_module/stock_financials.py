# src/stock_predictor/stock_module/stock_financials.py
# This module contains the StockFinancials class, which is used to retrieve financial data for a given stock symbol.
import pandas as pd
import yfinance as yf

from stock_predictor.global_settings import check_for_float_or_int
from stock_predictor.stock_module.stock_base import StockBase


class StockFinancials(StockBase):
    def __init__(self):
        super().__init__()
        self.check_for_float_or_int = check_for_float_or_int

    ###############################################################
    # Compute Margin
    ###############################################################
    def compute_margin(
        self, df: pd.DataFrame, numerator: str, denominator: str
    ) -> pd.Series:
        """Compute the margin given a DataFrame, numerator, and denominator columns.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            numerator (str): The column name for the numerator.
            denominator (str): The column name for the denominator.

        Returns:
            pd.Series: Computed margin as a Series.
        """
        pd.set_option("future.no_silent_downcasting", True)
        try:
            if numerator in df and denominator in df:
                margin = (df[numerator] / df[denominator] * 100).replace(
                    {float("inf"): 0.0, float("-inf"): 0.0, pd.NA: 0.0}
                )
                return margin
            return pd.Series(0.0, index=df.index)
        except ZeroDivisionError:
            return pd.Series(0.0, index=df.index)

    ###############################################################
    # Get Financials
    ###############################################################
    def get_financials(self, symbol: str, ticker: yf.Ticker) -> pd.DataFrame:
        """Retrieve financial data for a given stock symbol.

        Args:
            symbol (str): The stock symbol to retrieve financial data for.
            ticker (yf.Ticker): The yfinance Ticker object.

        Returns:
            pd.DataFrame: DataFrame containing the financial data for the stock symbol.
        """
        try:
            financials = ticker.financials.transpose().iloc[::-1].copy()
            df = pd.DataFrame(index=financials.index)
            df["market_cap"] = int(ticker.info.get("marketCap", 0))
            df["gross_margin_pct"] = self.compute_margin(
                financials, "Gross Profit", "Total Revenue"
            )
            df["net_margin_pct"] = self.compute_margin(
                financials, "Net Income", "Total Revenue"
            )

            info = ticker.info

            # Assigning Info Values
            df["symbol"] = symbol
            df["name"] = info.get("shortName", "")
            df["trailing_pe"] = self._get_valid_float(info, "trailingPE")
            df["forward_pe"] = self._get_valid_float(info, "forwardPE")
            df["industry"] = info.get("industry", "")

            return df.tail(1).dropna()
        except Exception as e:
            self.logger.error(f"Error processing financials for {symbol}: {e}")
            return pd.DataFrame()

    ###############################################################
    # Get Valid Float
    ###############################################################
    def _get_valid_float(self, info: dict, key: str) -> float:
        """Helper function to get a valid float value from the ticker info.

        Args:
            info (dict): Ticker information dictionary.
            key (str): Key for the required value.

        Returns:
            float: Valid float value or 0 if invalid.
        """
        value = info.get(key, 0)
        return value if self.check_for_float_or_int(value) else 0.0
