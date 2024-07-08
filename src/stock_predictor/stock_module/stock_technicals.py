from stock_predictor.global_settings import logger
import pandas as pd
from stock_predictor.stock_module.stock_base import StockBase

from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator


class StockTechnicals(StockBase):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def check_for_overbought(history: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Check for overbought stocks.

        Returns:
            pd.DataFrame: DataFrame containing the overbought stocks.
        """
        history_df = history.copy()
        rsi = RSIIndicator(close=history_df["close"], window=window).rsi()
        bb = BollingerBands(close=history_df["close"], window=window, window_dev=2)
        sma = SMAIndicator(close=history_df["close"], window=window)
        history_df[f"rsi{window}"] = rsi
        history_df[f"bbhi{window}"] = bb.bollinger_hband_indicator()
        history_df[f"bblo{window}"] = bb.bollinger_lband_indicator()
        history_df[f"sma{window}"] = sma.sma_indicator()

        return history_df

    def get_technicals_df(self, symbol: str, history: pd.DataFrame) -> bool:
        """
        Retrieves technical indicators for a given stock symbol based on historical data.

        Args:
            symbol (str): The stock symbol.
            history (pd.DataFrame): The historical data for the stock.

        Returns:
            bool: True if the technical indicators are successfully retrieved, False otherwise.
        """
        df_tech = []
        history_df = history.copy()
        try:
            # history = self.get_data(symbol=symbol)
            if history_df.empty:
                return False
            for window in [30, 50, 200]:
                history_df = self.check_for_overbought(history_df, window)

            df_tech_temp = history_df.tail(1)
            df_tech.append(df_tech_temp)
        except Exception as e:
            logger.error(f"Unhandled exception processing {symbol}. Error: {e}")
            return False

        if df_tech:
            df_tech = pd.concat([x for x in df_tech if not x.empty], axis=0)
            df_tech = df_tech.drop(
                columns={
                    "open",
                    "high",
                    "low",
                    "volume",
                }
            )
        else:
            df_tech = pd.DataFrame()

        return self.filter_criteria(df_tech)

    def filter_criteria(self, df: pd.DataFrame) -> bool:
        """
        Filter the DataFrame based on technical indicators.

        Args:
            df (pd.DataFrame): DataFrame containing the technical indicators.

        Returns:
            pd.DataFrame: DataFrame containing the filtered technical indicators.
        """
        RSI_COLUMNS = ["rsi30", "rsi50", "rsi200"]
        BBHI_COLUMNS = ["bbhi30", "bbhi50", "bbhi200"]
        SMA_COLUMNS = ["sma30", "sma50", "sma200"]

        criteria = (
            df[RSI_COLUMNS].gt(70).any(axis=1)
            | df[BBHI_COLUMNS].eq(1).any(axis=1)
            | df[SMA_COLUMNS].gt(df["close"]).any(axis=1)
        )
        return False if criteria.iloc[0] else True
