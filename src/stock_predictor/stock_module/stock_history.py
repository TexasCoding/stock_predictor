import pandas as pd
import requests
from typing import Optional
from stock_predictor.global_settings import BASE_URL, logger
from stock_predictor.stock_module.stock_base import StockBase


class StockHistory(StockBase):
    def __init__(self) -> None:
        super().__init__()

    def local_daily_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """Retrieve daily stock history from local database.

        Args:
            symbol (str): The stock symbol.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing stock history.
        """
        try:
            response = requests.get(f"{BASE_URL}/stock_data/{symbol}")
            response.raise_for_status()
            data_df = (
                pd.DataFrame(response.json())
                .sort_values("date")
                .reset_index(drop=True)
                .drop(columns=["id"])
            )
            data_df["date"] = pd.to_datetime(data_df["date"])
            data_df.set_index("date", inplace=True)
            return data_df
        except Exception as e:
            logger.error(f"Error fetching local daily history for {symbol}: {e}")
            return None

    def daily_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """Retrieve daily stock history using yfinance.

        Args:
            symbol (str): The stock symbol.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing stock history.
        """
        try:
            ticker = self.get_ticker(symbol)
            stock_data = ticker.history(
                start=self.calendar.past_trade_date,
                end=self.calendar.todays_date,
                interval="1d",
            )
            if stock_data.empty:
                return None
            stock_data = self._process_stock_data(stock_data, symbol)
            return stock_data
        except Exception as e:
            logger.error(f"Error fetching daily history for {symbol}: {e}")
            return None

    def _process_stock_data(
        self, stock_data: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Process raw stock data into a formatted DataFrame.

        Args:
            stock_data (pd.DataFrame): Raw stock data.
            symbol (str): The stock symbol.

        Returns:
            pd.DataFrame: Processed stock data.
        """
        bars_df = (
            stock_data.drop(columns=["Dividends", "Stock Splits"])
            .rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            .astype(
                {
                    "open": float,
                    "high": float,
                    "low": float,
                    "close": float,
                    "volume": int,
                }
            )
            .round({"open": 2, "high": 2, "low": 2, "close": 2})
        )

        for col in ["open", "high", "low", "close", "volume"]:
            mean_val = bars_df[col].mean(skipna=True)
            bars_df[col].replace({0.0: mean_val, 0: mean_val}, inplace=True)
            bars_df[col].fillna(mean_val, inplace=True)

        bars_df["date"] = bars_df.index.strftime("%Y-%m-%d")
        bars_df["symbol"] = symbol.upper()
        bars_df["vwap"] = self._calculate_vwap(bars_df)

        return bars_df

    @staticmethod
    def _calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate the Volume Weighted Average Price (VWAP).

        Args:
            df (pd.DataFrame): DataFrame containing stock data.

        Returns:
            pd.Series: Series containing VWAP values.
        """
        vwap = (
            ((df["high"] + df["low"] + df["close"]) / 3) * df["volume"]
        ).cumsum() / df["volume"].cumsum()
        return vwap.round(2)


# import pandas as pd
# import requests

# from stock_predictor.global_settings import BASE_URL
# from stock_predictor.stock_module.stock_base import StockBase


# class StockHistory(StockBase):
#     def __init__(self):
#         super().__init__()

#     def local_daily_history(self, symbol: str) -> pd.DataFrame:
#         response = requests.get(f"{BASE_URL}/stock_data/{symbol}")

#         data_df = (
#             pd.DataFrame(response.json())
#             .sort_values("date")
#             .reset_index(drop=True)
#             .drop(columns=["id"])
#         )

#         data_df["date"] = pd.to_datetime(data_df["date"])
#         data_df.index = data_df["date"]

#         return data_df

#     def daily_history(self, symbol: str) -> pd.DataFrame:
#         ticker = self.get_ticker(symbol)
#         stock_data = ticker.history(
#             start=self.calendar.past_trade_date,
#             end=self.calendar.todays_date,
#             interval="1d",
#         )

#         if not stock_data.empty:
#             # stock_data["Volume"] = stock_data["Volume"].astype("Int64")
#             # stock_data = stock_data.mask(stock_data=="na")

#             bars_df: pd.DataFrame = (
#                 stock_data.drop(columns={"Dividends", "Stock Splits"})
#                 .rename(
#                     columns={
#                         "Open": "open",
#                         "High": "high",
#                         "Low": "low",
#                         "Close": "close",
#                         "Volume": "volume",
#                     }
#                 )
#                 .astype(
#                     {
#                         "open": float,
#                         "high": float,
#                         "low": float,
#                         "close": float,
#                         "volume": int,
#                     }
#                 )
#                 .round(
#                     {
#                         "open": 2,
#                         "high": 2,
#                         "low": 2,
#                         "close": 2,
#                     }
#                 )
#             )

#             mean_volume = int(bars_df["volume"].mean(skipna=True))
#             mean_high = float(bars_df["high"].mean(skipna=True))
#             mean_low = float(bars_df["low"].mean(skipna=True))
#             mean_open = float(bars_df["open"].mean(skipna=True))
#             mean_close = float(bars_df["close"].mean(skipna=True))

#             bars_df["close"] = bars_df["close"].fillna(mean_close)
#             bars_df["high"] = bars_df["high"].fillna(mean_high)
#             bars_df["low"] = bars_df["low"].fillna(mean_low)
#             bars_df["open"] = bars_df["open"].fillna(mean_open)
#             bars_df["volume"] = bars_df["volume"].fillna(mean_volume)

#             bars_df["close"] = bars_df["close"].mask(
#                 bars_df["close"] == 0.0, mean_close
#             )
#             bars_df["high"] = bars_df["high"].mask(bars_df["high"] == 0.0, mean_high)
#             bars_df["low"] = bars_df["low"].mask(bars_df["low"] == 0.0, mean_low)
#             bars_df["open"] = bars_df["open"].mask(bars_df["open"] == 0.0, mean_open)
#             bars_df["volume"] = bars_df["volume"].mask(
#                 bars_df["volume"] == 0, mean_volume
#             )

#             def vwap(df):
#                 return (
#                     (
#                         ((df["high"] + df["low"] + df["close"]) / 3) * df["volume"]
#                     ).cumsum()
#                     / df["volume"].cumsum()
#                 ).round(2)

#             bars_df["date"] = bars_df.index.strftime("%Y-%m-%d")
#             bars_df["symbol"] = symbol.upper()
#             bars_df.index = bars_df.index.strftime("%Y-%m-%d")

#         bars_df["vwap"] = bars_df.groupby(bars_df.index, group_keys=False).apply(vwap)

#         return bars_df
