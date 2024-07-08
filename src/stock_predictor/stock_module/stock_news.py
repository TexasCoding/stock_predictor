# src/stock_trader/yahoo_module/yahoo_news.py
# The YahooNews class is used to retrieve news articles related to a stock symbol.
import textwrap
import time
from typing import Dict, List

from bs4 import BeautifulSoup as bs
import pendulum
from stock_predictor.stock_module.stock_base import StockBase
from stock_predictor.requests_retry import RequestsRetry

DEFAULT_TRUNCATE_LENGTH = 8000
DEFAULT_ARTICLE_LIMIT = 6


class StockNews(StockBase):
    def __init__(self):
        super().__init__()

    ########################################
    # Strip HTML
    ########################################
    @staticmethod
    def strip_html(content: str):
        """
        Removes HTML tags and returns the stripped content.

        Args:
            content (str): The HTML content to be stripped.

        Returns:
            str: The stripped content without HTML tags.
        """
        soup = bs(content, "html.parser")
        for data in soup(["style", "script"]):
            data.decompose()
        return " ".join(soup.stripped_strings)

    ########################################
    # Scrape article
    ########################################
    @staticmethod
    def scrape_article(url: str) -> str:
        """
        Scrapes the article text from the given URL.

        Args:
            url (str): The URL of the article.

        Returns:
            str: The text content of the article, or None if the article body is not found.
        """
        time.sleep(1)  # Sleep for 1 second to avoid rate limiting
        headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "en-US,en;q=0.9",
            "referer": "https://www.google.com",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, \
                like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.44",
        }
        request = RequestsRetry().request(method="GET", url=url, headers=headers)
        soup = bs(request.text, "html.parser")
        return (
            soup.find(class_="caas-body").text
            if soup.find(class_="caas-body")
            else None
        )

    ########################################
    # Truncate text
    ########################################
    @staticmethod
    def truncate(text: str, length: int) -> str:
        """
        Truncates a given text to a specified length.

        Args:
            text (str): The text to be truncated.
            length (int): The maximum length of the truncated text.

        Returns:
            str: The truncated text.
        """
        return (
            textwrap.shorten(text, length, placeholder="")
            if len(text) > length
            else text
        )

    ########################################
    # Get news
    ########################################
    def get_news(
        self,
        symbol: str,
        limit: int = DEFAULT_ARTICLE_LIMIT,
        max_article_length: int = DEFAULT_TRUNCATE_LENGTH,
    ) -> List[Dict[str, str]]:
        """
        Retrieves news articles related to the given symbol.

        Args:
            symbol (str): The symbol for which to retrieve news articles.
            limit (int, optional): The maximum number of articles to retrieve. Defaults to DEFAULT_ARTICLE_LIMIT.
            max_article_length (int, optional): The maximum length of each article. Defaults to DEFAULT_TRUNCATE_LENGTH.

        Returns:
            List[Dict[str, str]]: A list of dictionaries representing the retrieved news articles.
        """

        yahoo_news = self._get_yahoo_news(
            symbol=symbol, limit=limit, max_article_length=max_article_length
        )

        sorted_news = sorted(
            yahoo_news, key=lambda x: pendulum.parse(x["publish_date"]), reverse=True
        )

        return sorted_news[:limit]

    ########################################
    # Get Yahoo news
    ########################################
    def _get_yahoo_news(
        self,
        symbol: str,
        limit: int = DEFAULT_ARTICLE_LIMIT,
        max_article_length: int = DEFAULT_TRUNCATE_LENGTH,
    ) -> List[Dict[str, str]]:
        """
        Retrieves Yahoo news articles for a given stock symbol.

        Args:
            symbol (str): The stock symbol for which to retrieve news articles.
            limit (int, optional): The maximum number of news articles to retrieve. Defaults to DEFAULT_ARTICLE_LIMIT.
            max_article_length (int, optional): The maximum length of each news article. Defaults to DEFAULT_TRUNCATE_LENGTH.

        Returns:
            List[Dict[str, str]]: A list of dictionaries representing the retrieved news articles. Each dictionary contains the following keys:
                - "title": The title of the news article.
                - "url": The URL of the news article.
                - "source": The source of the news article (in this case, "yahoo").
                - "content": The truncated content of the news article.
                - "publish_date": The publish date of the news article.
                - "symbol": The stock symbol associated with the news article.
        """
        ticker = self.get_ticker(symbol=symbol)
        news_response = ticker.news

        yahoo_news = []
        news_count = 0
        for news in news_response:
            if news_count == limit:
                news_count = 0
                break

            try:
                content = self.strip_html(content=self.scrape_article(url=news["link"]))
                if not content:
                    continue
                news_count += 1
                yahoo_news.append(
                    {
                        "title": news["title"],
                        "url": news["link"],
                        "source": "yahoo",
                        "content": self.truncate(
                            text=content, length=max_article_length
                        )
                        if content
                        else None,
                        "publish_date": pendulum.from_timestamp(
                            news["providerPublishTime"]
                        ).to_datetime_string(),
                        "symbol": symbol,
                    }
                )
            except Exception as e:
                self.logger.warning(f"Error scraping article: {e}")
                continue

        return yahoo_news
