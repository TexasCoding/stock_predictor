from stock_predictor.global_settings import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_PAPER,
)
from dataclasses import dataclass
import pendulum
import datetime as dt
from alpaca.trading.client import TradingClient

# import pandas as pd
from alpaca.trading.requests import GetCalendarRequest

from dotenv import load_dotenv

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
    past_trade_date: str
    last_trade_date: str
    future_dates: FutureDates


class StockCalendar:
    def __init__(self):
        self.trading_client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=True if ALPACA_PAPER == "True" else False,
        )

    def _get_date(self, date_obj: pendulum.DateTime):
        return dt.date(date_obj.year, date_obj.month, date_obj.day)

    def _get_trade_dates(self, start_date, end_date):
        calendar = self.trading_client.get_calendar(
            GetCalendarRequest(start=start_date, end=end_date)
        )
        return [x.date for x in calendar]

    def _get_last_trade_date(self, start_date, end_date):
        trade_dates = self._get_trade_dates(start_date, end_date)
        if trade_dates:
            return trade_dates[-1].strftime("%Y-%m-%d")
        return None

    def _get_future_trade_dates(self, start_date, end_date, num_days):
        trade_dates = self._get_trade_dates(start_date, end_date)
        return {
            f"next_day{i+1}": trade_dates[i].strftime("%Y-%m-%d")
            for i in range(num_days)
        }

    def _get_past_trade_date(self, start_date, end_date, limit=1095):
        calendar = self.trading_client.get_calendar(
            GetCalendarRequest(start=start_date, end=end_date)
        )[-limit:]
        past_dates = [x.date for x in calendar]
        if past_dates:
            return past_dates[0].strftime("%Y-%m-%d")
        return None

    def calender(self) -> Dates:
        today = pendulum.now(tz="America/New_York")
        week_ago = today.subtract(days=7)
        future_days = today.add(days=6)
        past_days = today.subtract(years=5)

        today_date = self._get_date(today)
        week_back_date = self._get_date(week_ago)
        future_date = self._get_date(future_days)
        past_date = self._get_date(past_days)

        last_trade_date_str = self._get_last_trade_date(week_back_date, today_date)
        future_dates = self._get_future_trade_dates(last_trade_date_str, future_date, 4)
        past_trade_date_str = self._get_past_trade_date(past_date, last_trade_date_str)

        dates = Dates(
            todays_date=today_date.strftime("%Y-%m-%d"),
            past_trade_date=past_trade_date_str,
            last_trade_date=last_trade_date_str,
            future_dates=FutureDates(**future_dates),
        )
        return dates


if __name__ == "__main__":
    calendar = StockCalendar().calender()
    print(calendar)
