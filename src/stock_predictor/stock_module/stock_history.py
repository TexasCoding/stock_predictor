import pandas as pd
import requests

from stock_predictor.global_settings import BASE_URL
from stock_predictor.stock_module.stock_base import StockBase


class StockHistory(StockBase):
    def __init__(self):
        super().__init__()

    def local_daily_history(self, symbol: str) -> pd.DataFrame:
        response = requests.get(f"{BASE_URL}/stock_data/{symbol}")

        data_df = (
            pd.DataFrame(response.json())
            .sort_values("date")
            .reset_index(drop=True)
            .drop(columns=["id"])
        )

        data_df["date"] = pd.to_datetime(data_df["date"])
        data_df.index = data_df["date"]

        return data_df

    def daily_history(self, symbol: str) -> pd.DataFrame:
        ticker = self.get_ticker(symbol)
        stock_data = ticker.history(
            start=self.calendar.past_trade_date,
            end=self.calendar.todays_date,
            interval="1d",
        )

        if not stock_data.empty:
            # stock_data["Volume"] = stock_data["Volume"].astype("Int64")
            # stock_data = stock_data.mask(stock_data=="na")

            bars_df: pd.DataFrame = (
                stock_data.drop(columns={"Dividends", "Stock Splits"})
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
                .round(
                    {
                        "open": 2,
                        "high": 2,
                        "low": 2,
                        "close": 2,
                    }
                )
            )

            mean_volume = int(bars_df["volume"].mean(skipna=True))
            mean_high = float(bars_df["high"].mean(skipna=True))
            mean_low = float(bars_df["low"].mean(skipna=True))
            mean_open = float(bars_df["open"].mean(skipna=True))
            mean_close = float(bars_df["close"].mean(skipna=True))

            bars_df["close"] = bars_df["close"].fillna(mean_close)
            bars_df["high"] = bars_df["high"].fillna(mean_high)
            bars_df["low"] = bars_df["low"].fillna(mean_low)
            bars_df["open"] = bars_df["open"].fillna(mean_open)
            bars_df["volume"] = bars_df["volume"].fillna(mean_volume)

            bars_df["close"] = bars_df["close"].mask(
                bars_df["close"] == 0.0, mean_close
            )
            bars_df["high"] = bars_df["high"].mask(bars_df["high"] == 0.0, mean_high)
            bars_df["low"] = bars_df["low"].mask(bars_df["low"] == 0.0, mean_low)
            bars_df["open"] = bars_df["open"].mask(bars_df["open"] == 0.0, mean_open)
            bars_df["volume"] = bars_df["volume"].mask(
                bars_df["volume"] == 0, mean_volume
            )

            def vwap(df):
                return (
                    (
                        ((df["high"] + df["low"] + df["close"]) / 3) * df["volume"]
                    ).cumsum()
                    / df["volume"].cumsum()
                ).round(2)

            bars_df["date"] = bars_df.index.strftime("%Y-%m-%d")
            bars_df["symbol"] = symbol.upper()
            bars_df.index = bars_df.index.strftime("%Y-%m-%d")

        bars_df["vwap"] = bars_df.groupby(bars_df.index, group_keys=False).apply(vwap)

        return bars_df
