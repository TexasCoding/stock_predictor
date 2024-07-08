import json
import os
from typing import Dict, List

import requests
from stock_predictor.global_settings import BASE_URL
from stock_predictor.stock_module.stock_history import StockHistory
from stock_predictor.stock_module.stock_industries import StockIndustries
from stock_predictor.stock_module.stock_news import StockNews
from stock_predictor.stock_module.stock_openai_chat import StockOpenaiChat
from stock_predictor.stock_module.stock_predictor import StockPredictor
from stock_predictor.stock_module.stock_recommendations import StockRecommendations
from stock_predictor.stock_module.stock_tickers import StockTickers
from stock_predictor.stock_module.stock_technicals import StockTechnicals

from rich.console import Console

from dotenv import load_dotenv

load_dotenv()

console = Console()

DEFAULT_AVG_REDUCTION = float(os.getenv("DEFAULT_AVG_REDUCTION", 0.2))


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

    def filter_tickers(self) -> List[Dict[str, str]]:
        """
        Filter tickers based on technical indicators, recommendations, and predictions.
        """
        with console.status(
            f"[bold green]Filtering {len(self.tickers)} stocks for technicals..."
        ) as status:
            self.filter_tickers_technicals()
            status.update(
                f"[bold green]Filtering {len(self.filtered_technicals_tickers)} stocks for recommendations..."
            )
            self.filter_tickers_recommendations()

            filtered_count = len(self.filtered_recommended_tickers)
            for ticker in self.filtered_recommended_tickers:
                filtered_count -= 1
                console.print(
                    f"[blue]Filtering stocks [bold]{filtered_count}[/bold] for AI predictions..."
                )
                status.update(
                    f"[yellow]Analizing [bold]{ticker}[/bold] for futures preditions..."
                )
                prediction = self.filter_tickers_predictions(ticker=ticker)
                if prediction:
                    status.update(
                        f"[pink]Analizing news for [bold]{ticker}[/bold] with Openai..."
                    )
                    if self.filter_news(symbol=ticker):
                        self.add_predicted_to_db(predicted_dict=prediction)
                        status.update(
                            f"[green][bold]{ticker}[/bold] predicted to reach {prediction["take_price"]} within 3 days..."
                        )
                        self.filtered_predicted_tickers.append(prediction)

        return self.filtered_predicted_tickers

    def filter_news(self, symbol: str) -> bool:
        """Filters news articles based on sentiment analysis."""
        yahoo_news = StockNews()
        openai_chat = StockOpenaiChat()
        news_articles = yahoo_news.get_news(symbol=symbol)

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

    def filter_tickers_technicals(self) -> None:
        """
        Filter tickers based on technical indicators.
        """
        technicals = StockTechnicals()
        for ticker in self.tickers:
            history = StockHistory().local_daily_history(symbol=ticker)
            if history.empty:
                continue
            stock_technicals = technicals.get_technicals_df(
                symbol=ticker, history=history
            )
            if not stock_technicals:
                continue
            else:
                self.filtered_technicals_tickers.append(ticker)

    def filter_tickers_recommendations(self) -> None:
        """
        Filter tickers based on recommendations.
        """
        recommendations = StockRecommendations()
        for ticker in self.filtered_technicals_tickers:
            sentiment = recommendations.get_sentiment(symbol=ticker)
            if sentiment != "BULLISH":
                continue
            else:
                self.filtered_recommended_tickers.append(ticker)

    def filter_tickers_predictions(self, ticker: str) -> Dict[str, None]:
        """
        Filter tickers based on predictions.
        """

        history = StockHistory().local_daily_history(symbol=ticker)
        if history.empty:
            return {}
        stock_predictor = StockPredictor(symbol=ticker, data_df=history)
        prediction_df = stock_predictor.ExecuteModel()

        pred_df = prediction_df.tail(4)

        prediction_max = round(pred_df.tail(3)["vwap"].max(), 2)
        current_close = round(pred_df["close"].iloc[0], 2)
        future_goal = round(current_close * 1.03, 2)

        print(prediction_df)
        print(f"Prediction Max: {prediction_max}")
        print(f"Futur Goal: {future_goal}")
        stock_predictor.plot_chart(prediction_df)

        return (
            {
                "symbol": ticker,
                "open_price": current_close,
                "take_price": prediction_max,
            }
            if prediction_max >= future_goal
            else {}
        )

        ##############################

    # Add ticker to database
    ##############################
    def add_predicted_to_db(self, predicted_dict: dict) -> None:
        """
        Adds a ticker to the database.

        Args:
            ticker_df (pd.DataFrame): DataFrame containing the ticker information.

        Raises:
            Exception: If the API request to add the ticker fails.
        """
        base_url = BASE_URL

        params = {
            "symbol": predicted_dict["symbol"],
            "open_price": predicted_dict["open_price"],
            "take_price": predicted_dict["take_price"],
        }

        # print(json.dumps(params))

        response = requests.post(
            f"{base_url}/predicted_trades", data=json.dumps(params)
        )
        if response.status_code != 201:
            raise Exception(response.json()["detail"])
