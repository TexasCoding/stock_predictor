import json
import os
import time
from typing import Dict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd
import requests
from stock_predictor.global_settings import BASE_URL
from stock_predictor.polygon.polygon_history import PolygonHistory
from stock_predictor.stock_module.stock_industries import StockIndustries
from stock_predictor.stock_module.stock_news import StockNews
from stock_predictor.stock_module.stock_openai_chat import StockOpenaiChat
from stock_predictor.stock_module.stock_predictor import StockPredictor
from stock_predictor.stock_module.stock_recommendations import StockRecommendations
from stock_predictor.stock_module.stock_technicals import StockTechnicals
from stock_predictor.stock_module.stock_tickers import StockTickers

from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
console = Console()
DEFAULT_AVG_REDUCTION = float(os.getenv("DEFAULT_AVG_REDUCTION", 0.50))


class StockScreener2:
    def __init__(self) -> None:
        industry_avg_df = StockIndustries(
            average_reduction=DEFAULT_AVG_REDUCTION
        ).industry_avg_df(sort_by="count", ascending=False)
        tickers = StockTickers().get_profitable_tickers(industry_avg_df=industry_avg_df)
        # Remove duplicates, if any
        self.tickers = list(set(tickers))

    ###############################################################
    # Stock Screener
    ###############################################################
    def screener(self) -> pd.DataFrame:
        """
        Perform stock screening based on various filters and return a DataFrame of predicted stock gains.

        Returns:
            pd.DataFrame: DataFrame containing the predicted stock gains.
        """
        tickers = self.tickers.copy()
        predictions_df = pd.DataFrame()
        start_count = len(tickers)

        filtered_tickers = []

        with console.status("[bold green]Screening started...") as status:
            for ticker in tickers:
                status.update(
                    f"[green1]Screening [bold]{ticker}[/bold], ([bright_magenta]{start_count}[/bright_magenta]) remaining stocks..."
                )
                start_count -= 1
                time.sleep(0.5)
                history_df = PolygonHistory().daily_history(ticker)

                if history_df.empty:
                    continue

                if not self._filter_technicals(ticker, history_df):
                    continue

                if not self._filter_recommendations(ticker):
                    continue

                if not self._filter_news(ticker):
                    continue

                status.update(
                    f"[bright_cyan]AI predicting [bold]{ticker}[/bold] for future gains..."
                )
                prediction_df = self._filter_prediction(ticker, history_df)
                if not prediction_df.empty:
                    status.update(
                        f"[magenta3]AI predicted [bold]{ticker}[/bold] will gain..."
                    )
                    predictions_df = pd.concat([predictions_df, prediction_df])
                    filtered_tickers.append(ticker)
                    time.sleep(2)
                else:
                    status.update(
                        f"[red3][bold]{ticker}[/bold] not predicted for gains..."
                    )
                    time.sleep(2)
                    continue

        if predictions_df.empty:
            console.print("[red3]No stocks found for trading.")
        else:
            console.print(f"[green1]Stocks found for trading: {len(filtered_tickers)}")
        return predictions_df

    ###############################################################
    # Add Predicted Trade to Database
    ###############################################################
    def _add_predicted_to_db(self, predicted_dict: Dict[str, str]) -> None:
        """
        Adds the predicted trade details to the database.

        Args:
            predicted_dict (Dict[str, str]): A dictionary containing the predicted trade details.
                The dictionary should have the following keys:
                - "symbol": The symbol of the stock.
                - "open_price": The predicted open price of the stock.
                - "take_price": The predicted take price of the stock.
                - "percentage_change": The predicted percentage change of the stock.

        Raises:
            requests.HTTPError: If there is an error while making the POST request to the API.

        Returns:
            None
        """
        response = requests.post(
            f"{BASE_URL}/predicted_trades",
            data=json.dumps(
                {
                    "symbol": predicted_dict["symbol"],
                    "open_price": predicted_dict["open_price"],
                    "take_price": predicted_dict["take_price"],
                    "percentage_change": predicted_dict["percentage_change"],
                }
            ),
        )
        response.raise_for_status()

    ###############################################################
    # Filter Prediction
    ###############################################################
    def _filter_prediction(self, ticker: str, history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the prediction for a given stock ticker based on certain conditions.

        Args:
            ticker (str): The stock ticker symbol.
            history_df (pd.DataFrame): The historical data for the stock.

        Returns:
            pd.DataFrame: The filtered prediction dataframe.

        """
        copy_df = history_df.copy()
        if copy_df.empty:
            return pd.DataFrame()

        stock_predictor = StockPredictor(symbol=ticker, data_df=copy_df)
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

        if prediction_max >= future_goal:
            self._add_predicted_to_db(
                {
                    "symbol": ticker,
                    "open_price": current_close,
                    "take_price": round(
                        current_close + (prediction_max - current_vwap), 2
                    ),
                    "percentage_change": percentage_change,
                }
            )
            return prediction_df.tail(50)
        return pd.DataFrame()

    ###############################################################
    # Filter News
    ###############################################################
    def _filter_news(self, ticker: str) -> bool:
        """
        Filters news articles based on sentiment analysis.

        Args:
            ticker (str): The ticker symbol of the stock.

        Returns:
            bool: True if the sentiment score is greater than or equal to 1, False otherwise.
        """
        stock_news = StockNews()
        openai_chat = StockOpenaiChat()
        news_articles = stock_news.get_news(symbol=ticker, limit=5)

        sentiment_score = sum(
            1
            if openai_chat.get_sentiment_analysis(
                title=article["title"], symbol=ticker, article=article["content"]
            )
            == "BULLISH"
            else -1
            for article in news_articles
        )
        if sentiment_score < 1:
            return False

        return True

    ###############################################################
    # Filter Recommendations
    ###############################################################
    def _filter_recommendations(self, ticker: str) -> bool:
        """
        Filters the recommendations based on the sentiment of the given ticker.

        Args:
            ticker (str): The ticker symbol of the stock.

        Returns:
            bool: True if the sentiment is bullish, False otherwise.
        """
        recommendations = StockRecommendations()

        sentiment = recommendations.get_sentiment(symbol=ticker)
        if sentiment != "BULLISH":
            return False

        return True

    ###############################################################
    # Filter Technicals
    ###############################################################
    def _filter_technicals(self, ticker: str, history_df: pd.DataFrame) -> bool:
        """
        Filters the given historical data for a specific ticker based on technical indicators.

        Args:
            ticker (str): The ticker symbol for the stock.
            history_df (pd.DataFrame): The historical data for the stock.

        Returns:
            bool: True if the stock passes the technical indicators filter, False otherwise.
        """
        copy_df = history_df.copy()

        if copy_df.empty:
            return False

        technicals = StockTechnicals()
        if not technicals.check_for_tradeable(df=copy_df):
            return False

        return True
