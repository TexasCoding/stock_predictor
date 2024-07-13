import pandas as pd
import pendulum
from stock_predictor.global_settings import logger
from stock_predictor.polygon.polygon_base import PolygonBase

yesterday = pendulum.now().subtract(days=1).to_date_string()
past = pendulum.now().subtract(years=3).to_date_string()


class PolygonHistory(PolygonBase):
    def __init__(self) -> None:
        super().__init__()

    def daily_history(
        self,
        symbol: str,
        from_date: str = past,
        to_date: str = yesterday,
    ) -> pd.DataFrame:
        """
        Get the daily historical data for a stock symbol.

        Args:
            symbol (str): The stock symbol to get historical data for.
            date (str): The date to get historical data for.

        Returns:
            dict: The historical data for the stock symbol.
        """
        url = f"{self.polyv2}aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}"

        try:
            history = self.request("GET", url)
            history_df = pd.DataFrame(history.get("results", []))

            if history_df.empty:
                return history_df

            history_df["symbol"] = symbol
            history_df["date"] = pd.to_datetime(history_df["t"], unit="ms").dt.date
            history_df.index = history_df["date"]
            history_df = history_df.drop(columns=["t"])
            history_df = history_df.rename(
                columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "vw": "vwap",
                    "n": "transactions",
                }
            )
            history_df = history_df.round(
                {"open": 2, "high": 2, "low": 2, "close": 2, "vwap": 2}
            )
            history_df["volume"] = history_df["volume"].astype(int)

            return history_df
        except Exception as e:
            logger.warning(f"Error getting daily history for {symbol}: {e}")
            return pd.DataFrame()
