# src/stock_predictor/stock_module/stock_screener.py
# Description: Class to filter stocks based on technical indicators, recommendations, and predictions.
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pendulum
import requests
from stock_predictor.global_settings import (
    BASE_URL,
    FILTER_TECHNICALS,
    FILTER_RECOMMENDATIONS,
    FILTER_PREDICTIONS,
    FILTER_NEWS,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# from stock_predictor.stock_module.stock_charts import StockCharts
from stock_predictor.polygon.polygon_history import PolygonHistory

# from stock_predictor.stock_module.stock_history import StockHistory
from stock_predictor.stock_module.stock_industries import StockIndustries
from stock_predictor.stock_module.stock_news import StockNews
from stock_predictor.stock_module.stock_openai_chat import StockOpenaiChat
from stock_predictor.stock_module.stock_predictor import StockPredictor
from stock_predictor.stock_module.stock_recommendations import StockRecommendations
from stock_predictor.stock_module.stock_tickers import StockTickers
from stock_predictor.stock_module.stock_technicals import StockTechnicals

from rich.console import Console
from dotenv import load_dotenv

from rich.status import Status

load_dotenv()
console = Console()
DEFAULT_AVG_REDUCTION = float(os.getenv("DEFAULT_AVG_REDUCTION", 0.50))


class StockScreener:
    def __init__(self):
        self.industry_avg_df = StockIndustries(
            average_reduction=DEFAULT_AVG_REDUCTION
        ).industry_avg_df(sort_by="count", ascending=False)
        self.tickers = StockTickers().get_profitable_tickers(
            industry_avg_df=self.industry_avg_df
        )
        self.filtered_technicals_tickers: List[str] = []
        self.filtered_recommended_tickers: List[str] = []
        self.filtered_predicted_tickers: List[Dict[str, str]] = []

    ########################################################
    # This method is used to filter tickers based on
    # technical indicators, recommendations, and predictions.
    ########################################################
    def filter_tickers(self) -> Tuple[List[Dict[str, str]], pd.DataFrame]:
        """
        Filters the list of tickers based on technical indicators, recommendations, TensorFlow predictions, and news sentiment analysis.

        Returns:
            A list of dictionaries representing the filtered tickers.
        """
        all_predictions_df = pd.DataFrame()
        start_count = len(self.tickers)
        with console.status(FILTER_TECHNICALS) as status:
            # Filter tickers based on technical indicators.
            self._filter_tickers_technicals(status=status, start=start_count)
            start_count = len(self.filtered_technicals_tickers)

            # Filter tickers based on recommendations.
            self._filter_tickers_recommendations(status=status, start=start_count)
            filtered_count = len(self.filtered_recommended_tickers)

            # Filter tickers based on TensorFlow predictions.
            for ticker in self.filtered_recommended_tickers:
                filtered_count -= 1
                self.status_update(
                    status=status,
                    message=FILTER_PREDICTIONS,
                    count_len=filtered_count,
                    symbol=ticker,
                )
                # Filter ticker based on AI predictions.
                prediction, prediction_df = self._filter_ticker_prediction(
                    ticker=ticker
                )
                if prediction:
                    self.status_update(
                        status=status,
                        message=FILTER_NEWS,
                        symbol=ticker,
                    )
                    # Filter tickers based on news sentiment analysis using Openai.
                    if self._filter_news(symbol=ticker):
                        all_predictions_df = pd.concat(
                            [all_predictions_df, prediction_df]
                        )
                        # Add predicted ticker to database.
                        self._add_predicted_to_db(prediction)
                        # Append predicted ticker to list.
                        self.filtered_predicted_tickers.append(prediction)

        return self.filtered_predicted_tickers, all_predictions_df

    ########################################################
    # This method is used to update the status message.
    ########################################################
    @staticmethod
    def status_update(
        status: Status, message: str, count_len: int = 0, symbol: str = ""
    ) -> int:
        """
        Update the status with a formatted message.

        Args:
            status (Status): The status object to update.
            message (str): The message to be formatted and updated.
            count_len (int, optional): The length of the count. Defaults to 0.
            symbol (str, optional): The symbol to be replaced in the message. Defaults to "".

        Returns:
            int: The count length.

        """
        message_fmt = message.replace("{start_count}", str(count_len)).replace(
            "{symbol}", symbol
        )
        status.update(message_fmt)
        return count_len

    ########################################################
    # This method is used to filter news articles based on
    # sentiment analysis.
    ########################################################
    def _filter_news(self, symbol: str) -> bool:
        """
        Filters news articles based on sentiment analysis.

        Args:
            symbol (str): The stock symbol to filter news for.

        Returns:
            bool: True if the sentiment score is greater than 0, False otherwise.
        """
        stock_news = StockNews()
        openai_chat = StockOpenaiChat()
        news_articles = stock_news.get_news(symbol=symbol)

        sentiment_score = sum(
            1
            if openai_chat.get_sentiment_analysis(
                title=article["title"], symbol=symbol, article=article["content"]
            )
            == "BULLISH"
            else -1
            for article in news_articles
        )
        return sentiment_score > 0

    ########################################################
    # This method is used to filter tickers based on
    # technical indicators.
    ########################################################
    def _filter_tickers_technicals(self, status: Status, start) -> None:
        """
        Filters the tickers based on technical analysis.

        Args:
            status (Status): The status object for updating the status.
            start (int): The starting count length.

        Returns:
            None
        """

        ########################################################
        # This method is used to filter tickers based on
        # technical indicators.
        ########################################################
        def filter_technicals(ticker):
            """
            Filters the given ticker based on technical analysis.

            Args:
                ticker (str): The ticker symbol of the stock to filter.

            Returns:
                bool: True if the ticker passes the technical analysis filter, False otherwise.
            """
            # history = StockHistory().local_daily_history(symbol=ticker, limit=200)
            history = PolygonHistory().daily_history(
                symbol=ticker,
                from_date=pendulum.now().subtract(years=1).to_date_string(),
            )
            if history.empty:
                return False

            technicals = StockTechnicals()
            if technicals.check_for_tradeable(df=history):
                self.filtered_technicals_tickers.append(ticker)

        for ticker in self.tickers:
            start -= 1
            self.status_update(
                status=status,
                message=FILTER_TECHNICALS,
                count_len=start,
            )
            filter_technicals(ticker)

    ########################################################
    # This method is used to filter tickers based on
    # recommendations.
    ########################################################
    def _filter_tickers_recommendations(self, status: Status, start) -> None:
        """
        Filters the tickers based on recommendations and updates the status.

        Args:
            status (Status): The status object to update.
            start (int): The starting count length.

        Returns:
            None
        """
        recommendations = StockRecommendations()
        for ticker in self.filtered_technicals_tickers:
            start -= 1

            self.status_update(
                status=status,
                message=FILTER_RECOMMENDATIONS,
                count_len=start,
            )
            sentiment = recommendations.get_sentiment(symbol=ticker)
            if sentiment == "BULLISH":
                self.filtered_recommended_tickers.append(ticker)

    ########################################################
    # This method is used to filter tickers based on
    # predictions.
    ########################################################
    def _filter_ticker_prediction(
        self, ticker: str
    ) -> Tuple[Optional[Dict[str, str]], pd.DataFrame]:
        """
        Filters the ticker prediction based on certain criteria.

        Args:
            ticker (str): The ticker symbol of the stock.

        Returns:
            Optional[Dict[str, str]]: A dictionary containing the filtered ticker prediction information,
            including the symbol, open price, and take price. Returns None if the ticker prediction does not meet the criteria.
        """
        # history = StockHistory().local_daily_history(symbol=ticker)
        history = PolygonHistory().daily_history(symbol=ticker)
        if history.empty:
            return None

        stock_predictor = StockPredictor(symbol=ticker, data_df=history)
        prediction_df = stock_predictor.execute_model()

        pred_df = prediction_df.tail(4)
        prediction_max = round(pred_df.tail(3)["predicted_vwap"].max(), 2)
        current_close = round(pred_df["close"].iloc[0], 2)
        current_vwap = round(pred_df["vwap"].iloc[0], 2)
        future_goal = round(current_vwap * 1.03, 2)
        try:
            percentage_change = (
                round((prediction_max - current_vwap) / current_vwap, 2) * 100
            )
        except ZeroDivisionError:
            percentage_change = 0

        # print(f"Prediction Max: {prediction_max - current_vwap}")

        if prediction_max >= future_goal:
            # StockCharts().plot_prediction_chart(prediction_df.tail(100))
            # stock_predictor.plot_chart(prediction_df)
            return {
                "symbol": ticker,
                "open_price": current_close,
                "take_price": round(current_close + (prediction_max - current_vwap), 2),
                "percentage_change": percentage_change,
            }, prediction_df.tail(200)
        return None, pd.DataFrame()

    def _add_predicted_to_db(self, predicted_dict: Dict[str, str]) -> None:
        """
        Adds a ticker and its predicted target price to the database.

        Args:
            predicted_dict (Dict[str, str]): The dictionary containing ticker symbol, open price and target price.
        """
        response = requests.post(
            f"{BASE_URL}/predicted_trades",
            data=json.dumps(
                {
                    "symbol": predicted_dict["symbol"],
                    "open_price": predicted_dict["open_price"],
                    "take_price": predicted_dict["take_price"],
                }
            ),
        )
        response.raise_for_status()


# if __name__ == "__main__":
#     screener = StockScreener()
#     filtered_tickers, prediction_df = screener.filter_tickers()
#     print(filtered_tickers)
#     print(prediction_df)
#     print(len(filtered_tickers))
#     print(len(prediction_df))
