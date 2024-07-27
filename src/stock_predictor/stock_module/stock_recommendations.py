# src/stock_predictor/stock_module/stock_recommendations.py
# Description: Class to retrieve stock recommendations and calculate sentiment.
from typing import Union
import pandas as pd
from stock_predictor.stock_module.stock_base import StockBase
from fmp_py.fmp_company_information import FmpCompanyInformation
from fmp_py.fmp_valuation import FmpValuation
from fmp_py.fmp_statement_analysis import FmpStatementAnalysis
from fmp_py.fmp_price_targets import FmpPriceTargets
from fmp_py.fmp_quote import FmpQuote
from fmp_py.fmp_upgrades_downgrades import FMPUpgradesDowngrades

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

    def get_fmp_recommendations(self, symbol: str) -> str:
        try:
            recommendations = (
                FmpCompanyInformation().analyst_recommendations(symbol).iloc[0]
            )
            scores = FmpStatementAnalysis().financial_score(symbol=symbol)
            targets = FmpPriceTargets().price_target_consensus(symbol=symbol)
            quote = FmpQuote().simple_quote(symbol=symbol)
            rating = FmpValuation().company_rating(symbol).rating_score
            consensus = (
                FMPUpgradesDowngrades()
                .upgrades_downgrades_consensus(symbol)
                .consensus.lower()
            )
        except ValueError:
            return "NEUTRAL"

        buy_sum = (
            recommendations["analyst_ratings_buy"]
            + recommendations["analyst_ratings_strong_buy"]
        )
        sell_sum = (
            recommendations["analyst_ratings_sell"]
            + recommendations["analyst_ratings_strong_sell"]
            + recommendations["analyst_ratings_hold"]
        )

        if (buy_sum == 0 and sell_sum == 0) or rating < 4 or scores.piotroski_score < 5:
            return "NEUTRAL"
        if (
            (buy_sum > sell_sum)
            and rating >= 4
            and scores.piotroski_score >= 5
            and targets.target_consensus > quote.price
            and consensus in ["buy", "strong buy"]
        ):
            return "BULLISH"
        else:
            return "BEARISH"
