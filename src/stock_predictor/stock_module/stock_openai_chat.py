# src/stock_predictor/stock_module/stock_openai_chat.py
# Description: Class to interact with the OpenAI API for sentiment analysis.
import os
import re
from typing import Any, Dict, List
from openai import OpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv

load_dotenv()


class StockOpenaiChat:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o"

    ###############################################################
    # Completion with Exponential Backoff
    ###############################################################
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Makes a completion call to the OpenAI API with exponential backoff in case of failures.

        Args:
            **kwargs: Arbitrary keyword arguments for the OpenAI API call.

        Returns:
            response (Any): The response object from the OpenAI API.

        Raises:
            OpenAIError: If an error occurs with the OpenAI API.
        """
        try:
            openai = OpenAI(api_key=self.api_key)
            return openai.chat.completions.create(**kwargs)
        except OpenAIError as e:
            print(f"OpenAI API error: {e}")
            raise

    ###############################################################
    # Chat with OpenAI API
    ###############################################################
    def chat(self, msgs: List[Dict[str, str]]) -> Any:
        """
        Chat with the OpenAI API.

        Args:
            msgs (List[Dict[str, str]]): List of messages.

        Returns:
            response (Any): The response from OpenAI.
        """
        return self.completion_with_backoff(model=self.model, messages=msgs)

    ###############################################################
    # Get Sentiment Analysis
    ###############################################################
    def get_sentiment_analysis(self, title: str, symbol: str, article: str) -> str:
        """
        Get the sentiment analysis for financial news.

        Args:
            title (str): The title of the news.
            symbol (str): The stock symbol associated with the news.
            article (str): The content of the news article.

        Returns:
            sentiment (str): The sentiment analysis result - "BEARISH", "BULLISH", or "NEUTRAL".

        Raises:
            ValueError: If an unexpected sentiment response is received.
        """
        system_message = (
            "You will work as a Sentiment Analysis for Financial news. "
            "I will share news headline, stock symbol, and article. "
            "You will only answer as: BEARISH, BULLISH, or NEUTRAL. "
            "No further explanation. Got it?"
        )

        message_history = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{title}\n{symbol}\n{article}"},
        ]

        try:
            response = self.chat(message_history)
            sentiment = re.sub(
                "[^a-zA-Z]", "", response.choices[0].message.content.strip().upper()
            )
            if "BEARISH" in sentiment:
                sentiment = "BEARISH"
            elif "BULLISH" in sentiment:
                sentiment = "BULLISH"
            else:
                sentiment = sentiment
            if sentiment not in {"BEARISH", "BULLISH", "NEUTRAL"}:
                raise ValueError(f"Unexpected sentiment response: {sentiment}")
            return sentiment
        except (OpenAIError, ValueError) as e:
            print(f"Error in get_sentiment_analysis: {e}")
            return "NEUTRAL"
