import streamlit as st

from contextlib import contextmanager, redirect_stdout
from io import StringIO
from stock_predictor.stock_module.stock_screener import StockScreener


@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield


st.title("Stock Predictor")

start_button = st.button("Start Stock Predictor")

output = st.empty()

if start_button:
    with st_capture(output.code):
        predictor = StockScreener()
        filtered_tickers, filtered_tickers_df = predictor.filter_tickers()

    st.write(filtered_tickers)
    st.write(filtered_tickers_df)
