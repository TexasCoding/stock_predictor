# src/stock_predictor/stock_module/stock_data.py
# This module will be used to retrieve stock data and add it to the database.
import json
import pandas as pd
import requests
from stock_predictor.global_settings import BASE_URL, logger
from stock_predictor.stock_module.stock_history import StockHistory
from stock_predictor.stock_module.stock_tickers import StockTickers
from stock_predictor.stock_module.stock_industries import StockIndustries
from rich.console import Console
from typing import List

console = Console()


class StockData:
    def __init__(self) -> None:
        self.stock_history = StockHistory()
        self.base_url = BASE_URL

    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        """Retrieve daily stock history for a given symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            pd.DataFrame: DataFrame containing stock history.
        """
        return self.stock_history.daily_history(symbol)

    def add_stored_tickers_to_db(self) -> None:
        """Add stored tickers to the database."""
        try:
            stock_industries = StockIndustries(0.4)
            industries = stock_industries.industry_avg_df()
            stock_tickers = StockTickers()
            tickers = stock_tickers.get_profitable_tickers(industry_avg_df=industries)
            self._add_tickers_to_db(tickers)
        except Exception as e:
            logger.error(f"Error: {e}")

    def _add_tickers_to_db(self, tickers: List[str]) -> None:
        """Helper function to add tickers to the database.

        Args:
            tickers (List[str]): List of ticker symbols to add.
        """
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

    def add_stock_data_to_db(self, symbol: str) -> None:
        """Add stock data for a given symbol to the database.

        Args:
            symbol (str): The stock symbol.
        """
        stock_data = self.get_stock_data(symbol)
        post_list = self._prepare_stock_data_for_db(symbol, stock_data)

        try:
            response = requests.post(
                f"{self.base_url}/stock_data", data=json.dumps(post_list)
            )
            if response.status_code != 201:
                logger.error(f"Error: {response.text}")
        except Exception as e:
            logger.warning(f"Error: {e}")

    def _prepare_stock_data_for_db(
        self, symbol: str, stock_data: pd.DataFrame
    ) -> List[dict]:
        """Prepare stock data for database insertion.

        Args:
            symbol (str): The stock symbol.
            stock_data (pd.DataFrame): DataFrame containing stock data.

        Returns:
            List[dict]: List of dictionaries containing stock data.
        """
        post_list = []
        for _, row in stock_data.iterrows():
            post_list.append(
                {
                    "symbol": symbol,
                    "date": row["date"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "vwap": row.get("vwap", None),  # Use `get` to handle missing vwap
                }
            )
        return post_list


if __name__ == "__main__":
    StockData().add_stored_tickers_to_db()
# import json
# import pandas as pd
# import requests
# from stock_predictor.global_settings import BASE_URL, logger
# from stock_predictor.stock_module.stock_history import StockHistory
# from stock_predictor.stock_module.stock_tickers import StockTickers
# from stock_predictor.stock_module.stock_industries import StockIndustries

# from rich.console import Console

# console = Console()


# class StockData:
#     def __init__(self) -> None:
#         self.stock_history = StockHistory()
#         self.base_url = BASE_URL

#     def get_stock_data(self, symbol: str) -> pd.DataFrame:
#         return self.stock_history.daily_history(symbol)

#     def add_stored_tickers_to_db(self) -> None:
#         try:
#             stock_industries = StockIndustries(0.4)

#             industries = stock_industries.industry_avg_df()

#             stock_tickers = StockTickers()
#             tickers = stock_tickers.get_profitable_tickers(industry_avg_df=industries)

#             total_steps = len(tickers)
#             with console.status(
#                 f"[bold green]Adding {total_steps} tickers to database..."
#             ) as status:
#                 for ticker in tickers:
#                     total_steps -= 1
#                     status.update(
#                         f"[bold green]{total_steps} tickers remaining to add to database..."
#                     )
#                     self.add_stock_data_to_db(ticker)
#         except Exception as e:
#             print(f"Error: {e}")
#             return

#     def add_stock_data_to_db(self, symbol: str) -> None:
#         # tradable_stocks =
#         stock_data = self.get_stock_data(symbol)
#         # Add stock data to database
#         post_list = []
#         for index, row in stock_data.iterrows():
#             post_list.append(
#                 {
#                     "symbol": symbol,
#                     "date": row["date"],
#                     "open": row["open"],
#                     "high": row["high"],
#                     "low": row["low"],
#                     "close": row["close"],
#                     "volume": row["volume"],
#                     "vwap": row["vwap"],
#                 }
#             )

#         try:
#             response = requests.post(
#                 f"{self.base_url}/stock_data", data=json.dumps(post_list)
#             )
#             if response.status_code != 201:
#                 logger.info(f"Error: {response.text}")
#         except Exception as e:
#             logger.warning(f"Error: {e}")

#         return
