from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
import yfinance as yf
from alpaca.trading.client import TradingClient
from stock_predictor.global_settings import (
    BASE_URL,
    industry_sorter_check,
    logger,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_PAPER,
    stock_sorter_check,
)
from stock_predictor.stock_module.stock_calendar import Dates, StockCalendar


class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    """
    A session class that combines caching and rate limiting functionality.

    This class inherits from the `CacheMixin` and `LimiterMixin` classes, as well as the `Session` class.
    It provides the ability to cache HTTP responses and limit the rate of requests made using the session.
    """

    pass


class StockBase:
    def __init__(self):
        self.trading_client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=True if ALPACA_PAPER == "True" else False,
        )

        self.yfs_session = CachedLimiterSession(
            limiter=Limiter(
                RequestRate(2, Duration.SECOND * 5)
            ),  # max 2 requests per 5 seconds
            bucket_class=MemoryQueueBucket,
            backend=SQLiteCache("yfinance.cache"),
        )
        self.logger = logger

        self.calendar: Dates = StockCalendar().calendar()
        self.stock_sorter_check = stock_sorter_check
        self.industry_sorter_check = industry_sorter_check
        self.base_url = BASE_URL

    def get_ticker(self, symbol: str):
        """
        Retrieves the ticker object for a given stock symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            yfinance.Ticker: The ticker object.
        """
        return yf.Ticker(symbol, session=self.yfs_session)
