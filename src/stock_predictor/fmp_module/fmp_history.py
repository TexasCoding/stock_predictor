import pandas as pd
from stock_predictor.fmp_module.fmp_base import FmpBase
from stock_predictor.global_settings import logger


class FmpHistory(FmpBase):
    def __init__(self) -> None:
        super().__init__()

    def daily_history(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Get the daily historical data for a stock symbol.

        Args:
            symbol (str): The stock symbol to get historical data for.
            from_date (str): The start date to get historical data for.
            to_date (str): The end date to get historical data for.

        Returns:
            pd.DataFrame: The historical data for the stock symbol.
        """
        url = f"{self.fmp_url}historical-price-full/{symbol}?from={from_date}&to={to_date}"

        try:
            history = self.request("GET", url)
            history_df = pd.DataFrame(history.get("historical", []))

            if history_df.empty:
                return history_df

            history_df["symbol"] = symbol
            history_df["date"] = pd.to_datetime(history_df["date"]).dt.date
            history_df.index = history_df["date"]
            history_df = history_df.drop(columns=["date"])
            history_df = history_df.rename(
                columns={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )
            history_df = history_df.round({"open": 2, "high": 2, "low": 2, "close": 2})
            history_df["volume"] = history_df["volume"].astype(int)

            return history_df
        except Exception as e:
            logger.error(f"Error getting daily history for {symbol}: {e}")
            return pd.DataFrame()
