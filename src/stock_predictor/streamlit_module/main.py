import json
import os

# import time
from typing import Dict
from copy import deepcopy
import pendulum

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd
import requests
from stock_predictor.global_settings import BASE_URL

from stock_predictor.stock_module.stock_news import StockNews
from stock_predictor.stock_module.stock_openai_chat import StockOpenaiChat
from stock_predictor.stock_module.stock_predictor import StockPredictor
from stock_predictor.stock_module.stock_recommendations import StockRecommendations

from stock_predictor.stock_module.slack import Slack


from fmp_py.fmp_chart_data import FmpChartData
from fmp_py.fmp_company_information import FmpCompanyInformation

from rich.console import Console
from dotenv import load_dotenv

import streamlit as st

import plotly.graph_objs as go

load_dotenv()
console = Console()

yesterday = pendulum.now().subtract(days=1).to_date_string()
past = pendulum.now().subtract(years=4).to_date_string()


##########################################################
# Get Tickers
##########################################################
def get_tickers() -> list:
    screened = FmpCompanyInformation().stock_screener(
        market_cap_more_than=100000000,
        price_lower_than=300,
        limit=10000,
        volume_more_than=200000,
        is_actively_trading=True,
    )

    return screened["symbol"].tolist()


###############################################################
# Add Predicted Trade to Database
###############################################################
def add_predicted_to_db(predicted_dict: Dict[str, str]) -> None:
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
def filter_prediction(ticker: str, chart: FmpChartData) -> pd.DataFrame:
    history = deepcopy(chart)

    copy_df = history.return_chart()

    if copy_df.empty:
        return pd.DataFrame()

    fig = go.Figure()
    st.write(fig)

    stock_predictor = StockPredictor(symbol=ticker, data_df=copy_df)
    prediction_df = stock_predictor.execute_model()

    pred_df = prediction_df.tail(4)
    prediction_max = round(pred_df.tail(3)["predicted_close"].max(), 2)
    current_close = round(pred_df["close"].iloc[0], 2)
    future_goal = round(current_close * 1.03, 2)
    try:
        percentage_change = (
            round((prediction_max - current_close) / current_close, 2) * 100
        )
    except ZeroDivisionError:
        percentage_change = 0

    if prediction_max >= future_goal:
        predicted_df = prediction_df.tail(50)
        take_profit = round(current_close + (prediction_max - current_close), 2)
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=predicted_df.index,
                    open=predicted_df["open"],
                    high=predicted_df["high"],
                    low=predicted_df["low"],
                    close=predicted_df["close"],
                    name="Candlesticks",
                ),
                go.Scatter(
                    x=predicted_df.index,
                    y=predicted_df["predicted_close"],
                    name="Predicted Close",
                    line=dict(color="yellow"),
                ),
            ]
        )

        fig.update_layout(
            title=f"{ticker} - Entry Price: {current_close}, Take Price: {take_profit}, Percentage Change: %{percentage_change}",
            xaxis_title="",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            plot_bgcolor="rgba(0,0,0,0)",
            width=600,
            height=400,
        )
        fig.add_hline(y=take_profit, line_color="red", line_width=2)
        fig.add_hline(y=current_close, line_color="green", line_width=2)

        add_predicted_to_db(
            {
                "symbol": ticker,
                "open_price": current_close,
                "take_price": take_profit,
                "percentage_change": percentage_change,
            }
        )
        message = f"{ticker} predicted for gains:\n Open Price: {current_close}\n Take Price: {take_profit}\n Percentage Change: {percentage_change}"
        st.write(fig)

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
def filter_news(ticker: str) -> bool:
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

    return sentiment_score > 0


###############################################################
# Filter Recommendations
###############################################################
def filter_recommendations(ticker: str) -> bool:
    recommendations = StockRecommendations()

    sentiment = recommendations.get_fmp_recommendations(symbol=ticker)

    return sentiment == "BULLISH"


###############################################################
# Filter Technicals
###############################################################
def filter_technicals(
    chart: FmpChartData, fast_period: int = 20, slow_period: int = 40
) -> bool:
    history = deepcopy(chart)
    history.rsi(fast_period)
    history.bb(fast_period, 2)
    history.waddah_attar_explosion(fast_period, slow_period, 20, 2.0, 150)

    history_df = history.return_chart()

    if history_df.empty:
        return False

    prev_day = history_df.iloc[-1]

    return (
        prev_day["wae_uptrend"] == 1
        # and float(prev_day[f"rsi{fast_period}"]) < 70.0
        # and prev_day[f"bb_h{fast_period}_ind"] == 0
        # and prev_day[f"bb_l{fast_period}_ind"] == 0
    )


t = st
t = t.empty()


def run_program():
    tickers = get_tickers()
    start_count = len(tickers)

    predictions_df = pd.DataFrame()
    filtered_tickers = []

    for ticker in tickers:
        # time.sleep(0.2)
        t.markdown(
            f"<span style='font-size: 25px'>:green[Screening] **:blue[{ticker}]** --- **:orange[({start_count})]** :green[remaining stocks...]</span>",
            unsafe_allow_html=True,
        )
        start_count -= 1

        chart = FmpChartData(
            symbol=ticker,
            from_date=past,
            to_date=yesterday,
        )

        if not (
            filter_technicals(chart=chart, fast_period=20, slow_period=40)
            and filter_recommendations(ticker)
            and filter_news(ticker)
        ):
            continue

        t.markdown(
            f"<span style='font-size: 25px'>:blue[AI predicting] :violet[( **{ticker}** )] :blue[for future gains...]</span>",
            unsafe_allow_html=True,
        )
        prediction_df = filter_prediction(ticker, chart)
        if not prediction_df.empty:
            predictions_df = pd.concat([predictions_df, prediction_df])
            filtered_tickers.append(ticker)

    if predictions_df.empty:
        st.markdown(":red[No predictions found]")
    else:
        st.markdown(
            f":green[AI predicted {len(predictions_df)} stocks for future gains]"
        )
        st.write(predictions_df)


st.button("Get Predictions", on_click=run_program)
