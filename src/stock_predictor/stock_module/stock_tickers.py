# src/stock_predictor/stock_module/stock_tickers.py
# Description: Class to retrieve and manage stock tickers from Alpaca and Yahoo Finance.
import json
from typing import List
import requests
import pendulum

import yfinance as yf

from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
import pandas as pd

from stock_predictor.global_settings import chunk_list

from stock_predictor.stock_module.stock_base import StockBase
from stock_predictor.stock_module.stock_financials import StockFinancials


from rich.console import Console

console = Console()


class StockTickers(StockBase):
    def __init__(self):
        """
        Initializes a new instance of the StockTickers class.
        """
        super().__init__()
        self.stock_financials = StockFinancials()

    ##############################
    # Get profitable tickers
    ##############################
    def get_profitable_tickers(
        self,
        industry_avg_df: pd.DataFrame,
    ) -> List[str]:
        """
        Returns a list of profitable tickers based on the given industry average DataFrame.

        Parameters:
        - industry_avg_df (pd.DataFrame): DataFrame containing industry average values.

        Returns:
        - List[str]: List of profitable tickers.
        """

        tickers_df = self.get_stored_tickers_df()
        merged_df = pd.merge(tickers_df, industry_avg_df, on="industry")

        condition = (
            (merged_df["gross_margin_pct"] > merged_df["gross_margin_avg"])
            & (merged_df["net_margin_pct"] > merged_df["net_margin_avg"])
            & (merged_df["trailing_pe"] > merged_df["trailing_pe_avg"])
            & (merged_df["forward_pe"] > merged_df["forward_pe_avg"])
        )

        profitable_tickers = merged_df[condition]["symbol"].tolist()

        return profitable_tickers

    ##############################
    # Get tradeable tickers
    ##############################
    def get_tradeable_tickers(self) -> list[str]:
        """
        Retrieves a list of tradeable stock tickers from the Alpaca trading platform.

        Returns:
            list[str]: A list of stock tickers that meet the specified criteria.
        """
        search_params = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY, status="active"
        )
        assets = self.trading_client.get_all_assets(search_params)
        tickers_df = pd.DataFrame([dict(asset) for asset in assets])

        # Filter tradeable, marginable, shortable, and non-OTC tickers
        filtered_tickers = tickers_df[
            tickers_df["tradable"]
            & tickers_df["marginable"]
            & tickers_df["shortable"]
            & (tickers_df["exchange"] != "OTC")
        ]

        return filtered_tickers["symbol"].tolist()

    ##############################
    # Add tickers to database
    ##############################
    def new_add_tickers_to_db(self):
        """
        Adds tickers to the database.

        This method retrieves a list of tickers, chunks them into smaller groups, and adds them to the database.
        It also performs various checks and updates on the tickers before adding them to the database.

        Returns:
            chunked_tickers (list): A list of tickers chunked into smaller groups.
        """
        stored_tickers, stored_tickers_list = self._get_stored_tickers()
        tickers = self.get_tradeable_tickers()
        chunked_tickers = chunk_list(tickers, 20)

        total_steps = len(tickers)
        with console.status(
            f"[bold green]Adding {total_steps} tickers to database..."
        ) as status:
            for chunk in chunked_tickers:
                tickers = yf.Tickers(tickers=chunk, session=self.yfs_session)

                for ticker in chunk:
                    total_steps -= 1
                    status.update(
                        f"[bold green]{total_steps} tickers remaining to add to database..."
                    )
                    ticker_df = self.stock_financials.get_financials(
                        symbol=ticker, ticker=tickers.tickers[ticker]
                    )

                    if ticker_df.empty:
                        continue

                    if self._update_ticker(
                        symbol=ticker,
                        ticker_df=ticker_df,
                        stored_tickers=stored_tickers,
                        stored_tickers_list=stored_tickers_list,
                    ):
                        continue

                    if self._tradeable_criteria(ticker_df=ticker_df):
                        self._add_ticker_to_db(ticker_df)
                    elif ticker in stored_tickers_list:
                        self._delete_ticker(
                            symbol=ticker, stored_tickers=stored_tickers
                        )

        return chunked_tickers

    ##############################
    # Get stored tickers DataFrame
    ##############################
    def get_stored_tickers_df(
        self, sort_by: str = "symbol", ascending: bool = True
    ) -> pd.DataFrame:
        """
        Retrieves stored tickers as a pandas DataFrame.

        Args:
            sort_by (str, optional): The column to sort the tickers by. Defaults to "symbol".
            ascending (bool, optional): Whether to sort in ascending order. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the stored tickers, sorted accordingly.
        """
        self.stock_sorter_check(sort_by)
        stored_tickers, stored_tickers_list = self._get_stored_tickers()
        return (
            pd.DataFrame(stored_tickers)
            .sort_values(by=sort_by, ascending=ascending)
            .reset_index(drop=True)
        )

    ##############################
    # Get stored tickers list
    ##############################
    def _get_stored_tickers(self) -> tuple[list[dict], list[str]]:
        """
        Get the list of tickers stored in the database.

        Returns:
            tuple: (list of ticker dicts, list of ticker symbols)
        """
        stored_tickers = requests.get(f"{self.base_url}/tickers").json()
        stored_tickers_list = [ticker["symbol"] for ticker in stored_tickers]
        return stored_tickers, stored_tickers_list

    ##############################
    # Update ticker
    ##############################
    def _update_ticker(
        self,
        symbol: str,
        ticker_df: pd.DataFrame,
        stored_tickers: list[dict],
        stored_tickers_list: list[str],
    ) -> bool:
        """
        Update ticker information in the database.

        Args:
            symbol (str): The symbol of the ticker to update.
            ticker_df (pd.DataFrame): The DataFrame containing updated ticker information.
            stored_tickers (list[dict]): List of stored ticker information.
            stored_tickers_list (list[str]): List of stored ticker symbols.

        Returns:
            bool: True if the ticker was successfully updated, False otherwise.
        """
        if symbol not in stored_tickers_list:
            return False

        if self._tradeable_criteria(ticker_df=ticker_df):
            ticker_id = next(
                ticker["id"] for ticker in stored_tickers if ticker["symbol"] == symbol
            )
            last_updated = next(
                ticker["updated_at"]
                for ticker in stored_tickers
                if ticker["symbol"] == symbol
            )

            if (
                pendulum.now(tz="America/New_York") - pendulum.parse(last_updated)
            ).days < 5:
                return True

            response = requests.put(
                f"{self.base_url}/tickers/{ticker_id}",
                data=self._ticker_json(ticker_df=ticker_df),
            )
            if response.status_code != 204:
                raise Exception(response.json()["detail"])

        else:
            self._delete_ticker(symbol=symbol, stored_tickers=stored_tickers)

        return True

    ##############################
    # Delete ticker
    ##############################
    def _delete_ticker(self, symbol: str, stored_tickers: list[dict]) -> None:
        """
        Delete a ticker from the database.

        Args:
            symbol (str): The ticker symbol to delete.
            stored_tickers (list[dict]): List of stored tickers in the database.

        Raises:
            Exception: If the API request to delete the ticker fails.
        """
        ticker_id = next(
            ticker["id"] for ticker in stored_tickers if ticker["symbol"] == symbol
        )
        response = requests.delete(f"{self.base_url}/tickers/{ticker_id}")
        if response.status_code != 204:
            raise Exception(response.json()["detail"])

    ##############################
    # Tradeable criteria
    ##############################
    @staticmethod
    def _tradeable_criteria(ticker_df: pd.DataFrame) -> bool:
        """
        Check if the ticker meets the tradeable criteria.

        Args:
            ticker_df (pd.DataFrame): DataFrame containing ticker information.

        Returns:
            bool: True if the ticker meets the criteria, False otherwise.
        """
        criteria = [
            ticker_df["industry"].values[0] != "",
            ticker_df["market_cap"].values[0] != 0,
            ticker_df["gross_margin_pct"].values[0] != 0,
            ticker_df["net_margin_pct"].values[0] != 0,
            ticker_df["trailing_pe"].values[0] != 0,
            ticker_df["forward_pe"].values[0] != 0,
        ]
        return all(criteria)

    ##############################
    # Ticker JSON
    ##############################
    @staticmethod
    def _ticker_json(ticker_df: pd.DataFrame) -> str:
        """
        Convert a DataFrame containing ticker information to JSON.

        Args:
            ticker_df (pd.DataFrame): DataFrame containing ticker information.

        Returns:
            str: JSON string of the ticker information.
        """

        json_data = {
            "name": ticker_df["name"].values[0],
            "symbol": ticker_df["symbol"].values[0],
            "market_cap": ticker_df["market_cap"].item(),
            "gross_margin_pct": ticker_df["gross_margin_pct"].item(),
            "net_margin_pct": ticker_df["net_margin_pct"].item(),
            "trailing_pe": ticker_df["trailing_pe"].item(),
            "forward_pe": ticker_df["forward_pe"].item(),
            "industry": ticker_df["industry"].values[0],
        }
        return json.dumps(json_data, allow_nan=False)

    ##############################
    # Add ticker to database
    ##############################
    def _add_ticker_to_db(self, ticker_df: pd.DataFrame) -> None:
        """
        Adds a ticker to the database.

        Args:
            ticker_df (pd.DataFrame): DataFrame containing the ticker information.

        Raises:
            Exception: If the API request to add the ticker fails.
        """
        response = requests.post(
            f"{self.base_url}/tickers", data=self._ticker_json(ticker_df=ticker_df)
        )
        if response.status_code != 201:
            raise Exception(response.json()["detail"])
