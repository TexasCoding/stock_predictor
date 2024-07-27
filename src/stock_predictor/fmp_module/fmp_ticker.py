import json
import time
import pandas as pd
import requests
from stock_predictor.global_settings import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_PAPER,
    BASE_URL,
    logger,
)
from stock_predictor.fmp_module.fmp_base import FmpBase

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

from rich.console import Console

console = Console()


class FmpTicker(FmpBase):
    def __init__(self):
        super().__init__()

        self.trading_client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=True if ALPACA_PAPER == "True" else False,
        )

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

    def add_tickers(self) -> None:
        """
        Adds a stock ticker to the Alpaca trading platform.

        Args:
            symbol (str): The stock symbol to add.
        """
        tickers = self.get_tradeable_tickers()
        start_count = len(tickers)
        console.print(f"[bold green]Adding {start_count} tickers to database...")
        with console.status("") as status:
            for ticker in tickers:
                status.update(
                    f"[magenta]Adding [bold]{ticker}[/bold] financials to database, {start_count} remaining..."
                )
                start_count -= 1
                try:
                    if not self.get_ticker(ticker):
                        continue
                    self._add_ticker_to_db(self.get_ticker(ticker))
                except Exception as e:
                    logger.warning(f"[bold red]Error adding {ticker} to database: {e}")
                    continue

    ##############################
    # Add ticker to database
    ##############################
    def _add_ticker_to_db(self, ticker_dict: dict) -> None:
        """
        Adds a ticker to the database.

        Args:
            ticker_df (pd.DataFrame): DataFrame containing the ticker information.

        Raises:
            Exception: If the API request to add the ticker fails.
        """
        response = requests.post(
            f"{BASE_URL}/tickers", data=self._ticker_json(ticker_dict=ticker_dict)
        )
        if response.status_code != 201:
            raise Exception(response.json()["detail"])

    ##############################
    # Ticker JSON
    ##############################
    @staticmethod
    def _ticker_json(ticker_dict: dict) -> str:
        """
        Convert a DataFrame containing ticker information to JSON.

        Args:
            ticker_df (pd.DataFrame): DataFrame containing ticker information.

        Returns:
            str: JSON string of the ticker information.
        """

        json_data = {
            "symbol": ticker_dict["symbol"],
            "name": ticker_dict["name"],
            "image": ticker_dict["image"],
            "industry": ticker_dict["industry"],
            "sector": ticker_dict["sector"],
            "market_cap": ticker_dict["market_cap"],
            "gross_margin_pct": ticker_dict["gross_margin_pct"],
            "net_margin_pct": ticker_dict["net_margin_pct"],
            "trailing_pe": ticker_dict["trailing_pe"],
            "piotroski_score": ticker_dict["piotroski_score"],
        }
        return json.dumps(json_data, allow_nan=False)

    def get_ticker(self, symbol: str) -> dict:
        """Retrieve the ticker information for a given stock symbol.

        Args:
            symbol (str): The stock symbol to retrieve ticker information for.

        Returns:
            dict: Ticker information for the stock symbol.
        """
        try:
            profile_url = f"{self.v3_base_url}profile/{symbol}?apikey={self.api_key}"
            profile_response = self.get_request(profile_url)
            time.sleep(0.05)
            ratios_url = f"{self.v3_base_url}ratios/{symbol}?period=quarter&apikey={self.api_key}"
            ratios_response = self.get_request(ratios_url)
            time.sleep(0.05)
            scores_url = (
                f"{self.v4_base_url}score?symbol={symbol}&apikey={self.api_key}"
            )
            scores_response = self.get_request(scores_url)

            ticker = str(profile_response[0].get("symbol", None)).upper()
            name = str(profile_response[0].get("companyName", None)).upper()
            image = str(profile_response[0].get("image", None)).lower()
            industry = str(profile_response[0].get("industry", None)).upper()
            sector = str(profile_response[0].get("sector", None)).upper()
            market_cap = int(profile_response[0].get("mktCap", None))
            gross_margin_pct = float(ratios_response[0].get("grossProfitMargin", None))
            gross_margin_pct = gross_margin_pct * 100 if gross_margin_pct else None
            net_margin_pct = float(ratios_response[0].get("netProfitMargin", None))
            net_margin_pct = net_margin_pct * 100 if net_margin_pct else None
            trailing_pe = float(ratios_response[0].get("priceEarningsRatio", None))
            piotroski_score = int(scores_response[0].get("piotroskiScore", None))

            if (
                not ticker
                or not name
                or not image
                or not industry
                or not sector
                or not market_cap
                or not gross_margin_pct
                or not net_margin_pct
                or not trailing_pe
                or not piotroski_score
            ):
                return {}

            return {
                "symbol": ticker,
                "name": name,
                "image": image,
                "industry": industry,
                "sector": sector,
                "market_cap": market_cap,
                "gross_margin_pct": round(gross_margin_pct, 2),
                "net_margin_pct": round(net_margin_pct, 2),
                "trailing_pe": round(trailing_pe, 2),
                "piotroski_score": piotroski_score,
            }
        except Exception as e:
            logger.warning(f"[bold red]Error retrieving {symbol} data: {e}")
            return {}
