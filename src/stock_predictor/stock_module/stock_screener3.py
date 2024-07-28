import json
import os
import time
from typing import Dict
from copy import deepcopy
import pendulum

from stock_predictor.stock_module.stock_charts import StockCharts

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd
import requests
from stock_predictor.global_settings import BASE_URL

from stock_predictor.stock_module.stock_news import StockNews
from stock_predictor.stock_module.stock_openai_chat import StockOpenaiChat
from stock_predictor.stock_module.stock_predictor import StockPredictor
from stock_predictor.stock_module.stock_recommendations import StockRecommendations
# from stock_predictor.stock_module.stock_technicals import StockTechnicals

from stock_predictor.stock_module.slack import Slack


from fmp_py.fmp_chart_data import FmpChartData
from fmp_py.fmp_company_information import FmpCompanyInformation

from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
console = Console()
DEFAULT_AVG_REDUCTION = float(os.getenv("DEFAULT_AVG_REDUCTION", 0.20))


yesterday = pendulum.now().subtract(days=1).to_date_string()
past = pendulum.now().subtract(years=3).to_date_string()


class StockScreener3:
    def __init__(self) -> None:
        self.tickers = list(self._get_tickers())

    def _get_tickers(self) -> list:
        screened = FmpCompanyInformation().stock_screener(
            market_cap_more_than=100000000,
            price_lower_than=300,
            limit=10000,
            volume_more_than=200000,
            is_actively_trading=True,
        )

        return screened["symbol"].tolist()

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
                time.sleep(0.3)
                chart = FmpChartData(
                    symbol=ticker,
                    from_date=past,
                    to_date=yesterday,
                )

                status.update(
                    f"[green1]Screening [bold]{ticker}[/bold], ([bright_magenta]{start_count}[/bright_magenta]) remaining stocks..."
                )
                start_count -= 1

                if not self._filter_technicals(
                    chart=chart, fast_period=20, slow_period=40
                ):
                    continue

                if not self._filter_recommendations(ticker):
                    continue

                if not self._filter_news(ticker):
                    continue

                status.update(
                    f"[bright_cyan]AI predicting [bold]{ticker}[/bold] for future gains..."
                )
                prediction_df = self._filter_prediction(ticker, chart)
                if not prediction_df.empty:
                    status.update(
                        f"[magenta3]AI predicted [bold]{ticker}[/bold] will gain..."
                    )
                    StockCharts().plot_prediction_chart(prediction_df)
                    predictions_df = pd.concat([predictions_df, prediction_df])
                    filtered_tickers.append(ticker)
                else:
                    status.update(
                        f"[red3][bold]{ticker}[/bold] not predicted for gains..."
                    )
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
    def _filter_prediction(self, ticker: str, chart: FmpChartData) -> pd.DataFrame:
        """
        Filters the prediction for a given stock ticker based on certain conditions.

        Args:
            ticker (str): The stock ticker symbol.
            history_df (pd.DataFrame): The historical data for the stock.

        Returns:
            pd.DataFrame: The filtered prediction dataframe.

        """
        # history = FmpChartData(
        #             symbol=ticker,
        #             from_date=past,
        #             to_date=yesterday,
        #         )
        history = deepcopy(chart)

        copy_df = history.return_chart()

        if copy_df.empty:
            return pd.DataFrame()

        stock_predictor = StockPredictor(symbol=ticker, data_df=copy_df)
        prediction_df = stock_predictor.execute_model()

        pred_df = prediction_df.tail(4)
        prediction_max = round(pred_df.tail(3)["predicted_close"].max(), 2)
        current_close = round(pred_df["close"].iloc[0], 2)
        # current_vwap = round(pred_df["vwap"].iloc[0], 2)
        future_goal = round(current_close * 1.03, 2)
        try:
            percentage_change = (
                round((prediction_max - current_close) / current_close, 2) * 100
            )
        except ZeroDivisionError:
            percentage_change = 0

        if prediction_max >= future_goal:
            self._add_predicted_to_db(
                {
                    "symbol": ticker,
                    "open_price": current_close,
                    "take_price": round(
                        current_close + (prediction_max - current_close), 2
                    ),
                    "percentage_change": percentage_change,
                }
            )
            message = f"{ticker} predicted for gains:\n Open Price: {current_close}\n Take Price: {round(current_close + (prediction_max - current_close), 2)}\n Percentage Change: {percentage_change}"
            Slack().send_message(
                channel="#app-development",
                text=message,
                username=os.getenv("SLACK_USERNAME"),
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

        sentiment = recommendations.get_fmp_recommendations(symbol=ticker)
        if sentiment != "BULLISH":
            return False

        return True

    ###############################################################
    # Filter Technicals
    ###############################################################
    def _filter_technicals(
        self, chart: FmpChartData, fast_period: int = 20, slow_period: int = 40
    ) -> bool:
        """
        Filters the given historical data for a specific ticker based on technical indicators.

        Args:
            ticker (str): The ticker symbol for the stock.
            history_df (pd.DataFrame): The historical data for the stock.

        Returns:
            bool: True if the stock passes the technical indicators filter, False otherwise.
        """
        # history = FmpChartData(
        #             symbol=ticker,
        #             from_date=past,
        #             to_date=yesterday,
        #         )
        history = deepcopy(chart)
        # history.vwap()
        # history.sma(fast_period)
        # history.sma(slow_period)
        history.rsi(fast_period)
        history.bb(fast_period, 2)
        history.waddah_attar_explosion(fast_period, slow_period, 20, 2.0, 150)

        history_df = history.return_chart()

        if history_df.empty:
            return False

        # Get the last row of the DataFrame
        prev_day = history_df.iloc[-1]

        # Check the conditions in a single detailed conditional statement
        is_tradeable = (
            prev_day["wae_uptrend"] == 1
            and float(prev_day[f"rsi{fast_period}"]) < 70.0
            and prev_day[f"bb_h{fast_period}_ind"] == 0
            and prev_day[f"bb_l{fast_period}_ind"] == 0
            # and prev_day[f"sma{fast_period}"] < prev_day["vwap"]
            # and prev_day[f"sma{fast_period}"] > prev_day[f"sma{slow_period}"]
        )

        return is_tradeable
