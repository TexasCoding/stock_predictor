import json
import pandas as pd
import requests
from stock_predictor.global_settings import BASE_URL, logger
from stock_predictor.stock_module.stock_history import StockHistory
from stock_predictor.stock_module.stock_tickers import StockTickers
from stock_predictor.stock_module.stock_industries import StockIndustries

from rich.console import Console

console = Console()


class StockData:
    def __init__(self) -> None:
        self.stock_history = StockHistory()
        self.base_url = BASE_URL

    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        return self.stock_history.daily_history(symbol)

    def add_stored_tickers_to_db(self) -> None:
        try:
            stock_industries = StockIndustries(0.4)

            industries = stock_industries.industry_avg_df()

            stock_tickers = StockTickers()
            tickers = stock_tickers.get_profitable_tickers(industry_avg_df=industries)

            total_steps = len(tickers)
            with console.status(
                f"[bold green]Adding {total_steps} tickers to database..."
            ) as status:
                for ticker in tickers:
                    total_steps -= 1
                    status.update(
                        f"[bold green]{total_steps} tickers remaining to add to database..."
                    )
                    self.add_stock_data_to_db(ticker)
        except Exception as e:
            print(f"Error: {e}")
            return

    def add_stock_data_to_db(self, symbol: str) -> None:
        # tradable_stocks =
        stock_data = self.get_stock_data(symbol)
        # Add stock data to database
        post_list = []
        for index, row in stock_data.iterrows():
            post_list.append(
                {
                    "symbol": symbol,
                    "date": row["date"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "vwap": row["vwap"],
                }
            )

        try:
            response = requests.post(
                f"{self.base_url}/stock_data", data=json.dumps(post_list)
            )
            if response.status_code != 201:
                logger.info(f"Error: {response.text}")
        except Exception as e:
            logger.warning(f"Error: {e}")

        return
