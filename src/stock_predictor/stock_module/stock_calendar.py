# src/stock_predictor/stock_module/stock_calendar.py
# Description: Class to retrieve trading calendar dates from Alpaca API.
from stock_predictor.global_settings import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_PAPER,
)
from dataclasses import dataclass
import pendulum
import datetime as dt
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


@dataclass
class FutureDates:
    next_day1: str
    next_day2: str
    next_day3: str
    next_day4: str


@dataclass
class Dates:
    todays_date: str
    past_trade_date: Optional[str]
    last_trade_date: str
    future_dates: FutureDates


class StockCalendar:
    def __init__(self):
        """Initialize the TradingClient with API credentials."""
        self.trading_client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=(ALPACA_PAPER.lower() == "true"),
        )

    @staticmethod
    def _get_date(date_obj: pendulum.DateTime) -> dt.date:
        """Convert pendulum.DateTime to datetime.date."""
        return dt.date(date_obj.year, date_obj.month, date_obj.day)

    def _get_trade_dates(self, start_date: dt.date, end_date: dt.date) -> list[dt.date]:
        """Get trading dates from Alpaca API within the specified range."""
        calendar = self.trading_client.get_calendar(
            GetCalendarRequest(start=start_date, end=end_date)
        )
        return [x.date for x in calendar]

    def _get_last_trade_date(
        self, start_date: dt.date, end_date: dt.date
    ) -> Optional[str]:
        """Get the last trading date within the specified range."""
        trade_dates = self._get_trade_dates(start_date, end_date)
        return trade_dates[-1].strftime("%Y-%m-%d") if trade_dates else None

    def _get_future_trade_dates(
        self, start_date: str, end_date: dt.date, num_days: int
    ) -> FutureDates:
        """Get future trading dates from the last trading date within the specified range."""
        trade_dates = self._get_trade_dates(start_date, end_date)
        future_dates_dict = {
            f"next_day{i+1}": trade_dates[i].strftime("%Y-%m-%d")
            for i in range(min(num_days, len(trade_dates)))
        }
        return FutureDates(**future_dates_dict)

    def _get_past_trade_date(
        self, start_date: dt.date, end_date: str, limit: int = 1095
    ) -> Optional[str]:
        """Get the past trading date within the specified look-back period."""
        calendar = self.trading_client.get_calendar(
            GetCalendarRequest(start=start_date, end=end_date)
        )[-limit:]
        past_dates = [x.date for x in calendar]
        return past_dates[0].strftime("%Y-%m-%d") if past_dates else None

    def calendar(self) -> Dates:
        """Retrieve the trading calendar including today's date, last, and future trading dates."""
        today = pendulum.now(tz="America/New_York")
        today_date = self._get_date(today)

        last_trade_date_str = self._get_last_trade_date(
            today_date - dt.timedelta(days=7), today_date
        )
        future_dates = self._get_future_trade_dates(
            last_trade_date_str, today_date + dt.timedelta(days=6), 4
        )
        past_trade_date_str = self._get_past_trade_date(
            today_date - dt.timedelta(days=5 * 365), last_trade_date_str
        )

        return Dates(
            todays_date=today_date.strftime("%Y-%m-%d"),
            past_trade_date=past_trade_date_str,
            last_trade_date=last_trade_date_str,
            future_dates=future_dates,
        )
