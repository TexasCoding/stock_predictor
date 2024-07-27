# src/stock_predictor/stock_module/stock_technicals.py
# Description: Class to retrieve technical indicators for a given stock symbol based on historical data.
from stock_predictor.global_settings import logger
import pandas as pd
from stock_predictor.stock_module.stock_base import StockBase

from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator, EMAIndicator

WINDOW = 40


class StockTechnicals(StockBase):
    def __init__(self) -> None:
        super().__init__()

    ###############################################################
    # Check for Tradeable
    ###############################################################
    def check_for_tradeable(self, df: pd.DataFrame) -> bool:
        """
        Check if the given DataFrame has tradeable conditions based on specific technical indicators.

        Args:
            df (pd.DataFrame): DataFrame containing historical data with added technical indicators columns.

        Returns:
            bool: True if the conditions for a tradeable asset are met, False otherwise.
        """
        data_df = self.add_technicals_to_df(df)

        if data_df.empty:
            return False

        # Get the last row of the DataFrame
        yesterday = data_df.iloc[-1]

        # Check the conditions in a single detailed conditional statement
        is_tradeable = (
            yesterday["wae_uptrend"] == 1
            and float(yesterday["rsi"]) < 70.0
            and yesterday["bbhi"] == 0
            and yesterday["bblo"] == 0
            and yesterday["sma"] < yesterday["vwap"]
            and yesterday["sma"] > yesterday["sma80"]
        )

        return is_tradeable

    ###############################################################
    # Check for Overbought
    ###############################################################
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

    ###############################################################
    # Add Technicals to DataFrame
    ###############################################################
    def add_technicals_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing historical data.

        Returns:
            pd.DataFrame: DataFrame containing the historical data with technical indicators.
        """
        # window = WINDOW
        df = df.copy()
        df = self.calculate_waddah_attar_explosion(
            df=df, n_fast=20, n_slow=40, channel_period=20, mul=2.0, sensitivity=150
        )

        rsi = RSIIndicator(close=df["close"], window=14, fillna=True).rsi()
        bb = BollingerBands(close=df["close"], window=20, window_dev=2, fillna=True)
        sma = SMAIndicator(close=df["close"], window=40, fillna=True)
        sma80 = SMAIndicator(close=df["close"], window=80, fillna=True)
        df["rsi"] = rsi.round(2)
        df["bbhi"] = bb.bollinger_hband_indicator().astype(int)
        df["bblo"] = bb.bollinger_lband_indicator().astype(int)
        df["sma"] = sma.sma_indicator().round(2)
        df["sma80"] = sma80.sma_indicator().round(2)

        df = df.drop(columns={"open", "high", "low", "volume", "close"})

        return df

    ###############################################################
    # Get Technicals DataFrame
    ###############################################################
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

    ###############################################################
    # Filter Criteria
    ###############################################################
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

    def calculate_waddah_attar_explosion(
        self,
        df: pd.DataFrame,
        n_fast: int = 20,
        n_slow: int = 40,
        channel_period: int = 20,
        mul: float = 2.0,
        sensitivity: int = 150,
    ) -> pd.DataFrame:
        """
        Calculate Waddah Attar Explosion indicator.
        Args:
            df (pd.DataFrame): DataFrame containing historical data with columns 'close', 'high', 'low'.
            n_fast (int): Period for the fast EMA.
            n_slow (int): Period for the slow EMA.
            channel_period (int): Period for calculating the Bollinger Bands.
            mul (float): Multiplier for the Bollinger Bands.
            sensitivity (int): Sensitivity factor for explosion value.
        Returns:
            pd.DataFrame: DataFrame containing the Waddah Attar Explosion values.
        """

        # Calculate the MACD
        def calc_macd(close: pd.Series, n_fast: int, n_slow: int) -> pd.Series:
            fast_ma = EMAIndicator(close=close, window=n_fast).ema_indicator()
            slow_ma = EMAIndicator(close=close, window=n_slow).ema_indicator()
            return fast_ma - slow_ma

        # Calculate the Bollinger Bands
        def calc_bb_bands(close: pd.Series, channel_period: int, mul: float):
            bb = BollingerBands(close=close, window=channel_period, window_dev=mul)
            return bb.bollinger_hband(), bb.bollinger_lband()

        df = df.copy()
        macd_diff = calc_macd(df["close"], n_fast, n_slow) - calc_macd(
            df["vwap"], n_fast, n_slow
        ).shift(1)
        explosion = macd_diff * sensitivity

        bb_upper, bb_lower = calc_bb_bands(df["close"], channel_period, mul)
        explosion_band = bb_upper - bb_lower

        df["wae_value"] = explosion
        df["wae_explosion"] = explosion_band

        # Fill NaNs and round the values
        df["wae_value"] = df["wae_value"].fillna(df["wae_value"].mean()).round(2)
        df["wae_explosion"] = (
            df["wae_explosion"].fillna(df["wae_explosion"].mean()).round(2)
        )

        # Calculate the uptrend
        df["wae_uptrend"] = (
            (df["wae_value"] > df["wae_value"].shift(1))
            & (df["wae_value"] > 0)
            & (df["wae_value"] > df["wae_explosion"])
        ).astype(int)

        return df
