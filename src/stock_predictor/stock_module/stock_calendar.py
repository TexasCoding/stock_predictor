# src/stock_predictor/stock_module/stock_calendar.py
# Description: Class to retrieve trading calendar dates from Alpaca API.
from dataclasses import dataclass
import pendulum
from dotenv import load_dotenv
from typing import Optional

from stock_predictor.global_settings import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_PAPER,
)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

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
        pass

    ##############################
    # Method to retrieve the trading calendar including today's date, last, and future trading dates.
    ##############################
    def calendar(self) -> Dates:
        """
        Retrieves the stock market calendar information.

        Returns:
            Dates: An object containing the following information:
                - todays_date: The current date in the format "YYYY-MM-DD".
                - past_trade_date: The date of the last trade that occurred in the past, in the format "YYYY-MM-DD".
                - last_trade_date: The date of the last trade that occurred, in the format "YYYY-MM-DD".
                - future_dates: An object containing the following future dates:
                    - next_day1: The date of the next trading day after the last_trade_date, in the format "YYYY-MM-DD".
                    - next_day2: The date of the second next trading day after the last_trade_date, in the format "YYYY-MM-DD".
                    - next_day3: The date of the third next trading day after the last_trade_date, in the format "YYYY-MM-DD".
                    - next_day4: The date of the fourth next trading day after the last_trade_date, in the format "YYYY-MM-DD".
        """
        today = pendulum.now(tz="America/New_York")

        if today.day_of_week in [pendulum.SATURDAY, pendulum.SUNDAY, pendulum.MONDAY]:
            last_trade_date = today.previous(pendulum.FRIDAY)
            next_day1 = last_trade_date.next(pendulum.MONDAY)
            next_day2 = next_day1.next(pendulum.TUESDAY)
            next_day3 = next_day2.next(pendulum.WEDNESDAY)
            next_day4 = next_day3.next(pendulum.THURSDAY)
        elif today.day_of_week == pendulum.TUESDAY:
            last_trade_date = today.previous(pendulum.MONDAY)
            next_day1 = last_trade_date.next(pendulum.TUESDAY)
            next_day2 = next_day1.next(pendulum.WEDNESDAY)
            next_day3 = next_day2.next(pendulum.THURSDAY)
            next_day4 = next_day3.next(pendulum.FRIDAY)
        elif today.day_of_week == pendulum.WEDNESDAY:
            last_trade_date = today.previous(pendulum.TUESDAY)
            next_day1 = last_trade_date.next(pendulum.WEDNESDAY)
            next_day2 = next_day1.next(pendulum.THURSDAY)
            next_day3 = next_day2.next(pendulum.FRIDAY)
            next_day4 = next_day3.next(pendulum.MONDAY)
        elif today.day_of_week == pendulum.THURSDAY:
            last_trade_date = today.previous(pendulum.WEDNESDAY)
            next_day1 = last_trade_date.next(pendulum.THURSDAY)
            next_day2 = next_day1.next(pendulum.FRIDAY)
            next_day3 = next_day2.next(pendulum.MONDAY)
            next_day4 = next_day3.next(pendulum.TUESDAY)
        elif today.day_of_week == pendulum.FRIDAY:
            last_trade_date = today.previous(pendulum.THURSDAY)
            next_day1 = last_trade_date.next(pendulum.FRIDAY)
            next_day2 = next_day1.next(pendulum.MONDAY)
            next_day3 = next_day2.next(pendulum.TUESDAY)
            next_day4 = next_day3.next(pendulum.WEDNESDAY)

        future_dates = {
            "next_day1": next_day1.strftime("%Y-%m-%d"),
            "next_day2": next_day2.strftime("%Y-%m-%d"),
            "next_day3": next_day3.strftime("%Y-%m-%d"),
            "next_day4": next_day4.strftime("%Y-%m-%d"),
        }

        past_trade_date = today.subtract(days=5 * 365)

        calendar = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=(ALPACA_PAPER.lower() == "true"),
        ).get_calendar(
            GetCalendarRequest(start=past_trade_date.date(), end=last_trade_date.date())
        )

        past_trade_date = calendar[-1095].date

        return Dates(
            todays_date=today.strftime("%Y-%m-%d"),
            past_trade_date=past_trade_date.strftime("%Y-%m-%d"),
            last_trade_date=last_trade_date.strftime("%Y-%m-%d"),
            future_dates=FutureDates(**future_dates),
        )
