# src/stock_predictor/stock_module/stock_recommendations.py
# Description: Class to retrieve stock recommendations and calculate sentiment.
from typing import Union
import pandas as pd
from stock_predictor.stock_module.stock_base import StockBase

DEFAULT_PREVIOUS_MONTHS = 2


class StockRecommendations(StockBase):
    def __init__(self) -> None:
        super().__init__()

    ##############################
    # Get recommendations
    ##############################
    def get_recommendations(
        self, symbol: str, previous_months: int = DEFAULT_PREVIOUS_MONTHS
    ) -> Union[pd.DataFrame, str]:
        """
        Retrieves recommendations for a given stock symbol.

        Args:
            symbol (str): The stock symbol to retrieve recommendations for.
            previous_months (int, optional): The number of previous months of recommendations to retrieve. Defaults to DEFAULT_PREVIOUS_MONTHS.

        Returns:
            Union[pd.DataFrame, str]: DataFrame containing recommendations or "No recommendations found."
        """
        try:
            ticker = self.get_ticker(symbol)
            recommendations = ticker.recommendations
            if recommendations.empty:
                return "No recommendations found."

            return recommendations.head(previous_months)
        except Exception as e:
            self.logger.error(f"Error retrieving recommendations for {symbol}: {e}")
            return "Error retrieving recommendations."

    ##############################
    # Get sentiment
    ##############################
    def get_sentiment(
        self, symbol: str, previous_months: int = DEFAULT_PREVIOUS_MONTHS
    ) -> str:
        """
        Calculates the sentiment of a stock based on its recommendations.

        Args:
            symbol (str): The stock symbol.
            previous_months (int, optional): The number of previous months to consider for recommendations. Defaults to DEFAULT_PREVIOUS_MONTHS.

        Returns:
            str: The sentiment of the stock. Possible values are "BULLISH", "BEARISH", or "NEUTRAL".
        """
        try:
            recommendations_frame = self.get_recommendations(symbol, previous_months)
            if isinstance(recommendations_frame, str):
                return "NEUTRAL"

            buy_signals = (
                recommendations_frame["strongBuy"].sum()
                + recommendations_frame["buy"].sum()
            )
            sell_signals = (
                recommendations_frame["strongSell"].sum()
                + recommendations_frame["sell"].sum()
                + recommendations_frame["hold"].sum()
            )

            if buy_signals == 0 and sell_signals == 0:
                return "NEUTRAL"

            sentiment_ratio = buy_signals / (buy_signals + sell_signals)
            return "BULLISH" if sentiment_ratio > 0.7 else "BEARISH"
        except Exception as e:
            self.logger.error(f"Error calculating sentiment for {symbol}: {e}")
            return "Error calculating sentiment."
