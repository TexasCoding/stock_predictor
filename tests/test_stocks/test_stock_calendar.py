import pytest  # type: ignore
from unittest.mock import patch
import pendulum
import datetime as dt

from stock_predictor.stock_module.stock_calendar import (
    StockCalendar,
    Dates,
    FutureDates,
)


@pytest.fixture
def stock_calendar():
    return StockCalendar()


def test_initialization(stock_calendar):
    assert isinstance(stock_calendar, StockCalendar)


def test_get_date(stock_calendar):
    dt_obj = pendulum.datetime(2023, 10, 10)
    assert stock_calendar._get_date(dt_obj) == dt.date(2023, 10, 10)


@patch.object(StockCalendar, "_get_trade_dates")
def test_get_last_trade_date(mock_get_trade_dates, stock_calendar):
    mock_get_trade_dates.return_value = [dt.date(2023, 10, 10), dt.date(2023, 10, 11)]
    last_trade_date = stock_calendar._get_last_trade_date(
        dt.date(2023, 10, 1), dt.date(2023, 10, 15)
    )
    assert last_trade_date == "2023-10-11"


@patch.object(StockCalendar, "_get_trade_dates")
def test_get_future_trade_dates(mock_get_trade_dates, stock_calendar):
    mock_get_trade_dates.return_value = [
        dt.date(2023, 10, 12),
        dt.date(2023, 10, 13),
        dt.date(2023, 10, 16),
        dt.date(2023, 10, 17),
    ]
    future_dates = stock_calendar._get_future_trade_dates(
        "2023-10-11", dt.date(2023, 10, 18), 4
    )
    expected = FutureDates("2023-10-12", "2023-10-13", "2023-10-16", "2023-10-17")
    assert future_dates == expected


@patch.object(StockCalendar, "_get_last_trade_date")
@patch.object(StockCalendar, "_get_past_trade_date")
@patch.object(StockCalendar, "_get_future_trade_dates")
def test_calendar(
    mock_future_trade_dates, mock_past_trade_date, mock_last_trade_date, stock_calendar
):
    today = pendulum.datetime(2023, 10, 10)
    mock_last_trade_date.return_value = "2023-10-09"
    mock_future_trade_dates.return_value = FutureDates(
        "2023-10-11", "2023-10-12", "2023-10-13", "2023-10-16"
    )
    mock_past_trade_date.return_value = "2018-10-10"

    with patch("pendulum.now", return_value=today):
        result = stock_calendar.calendar()

    expected = Dates(
        todays_date="2023-10-10",
        past_trade_date="2018-10-10",
        last_trade_date="2023-10-09",
        future_dates=FutureDates(
            "2023-10-11", "2023-10-12", "2023-10-13", "2023-10-16"
        ),
    )
    assert result == expected
