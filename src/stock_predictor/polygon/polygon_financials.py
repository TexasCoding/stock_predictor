# import pandas as pd
# from pprint import pprint
import time
import pendulum
from stock_predictor.polygon.polygon_base import PolygonBase
from stock_predictor.global_settings import logger

CURRENT_QUARTER = pendulum.now().quarter


class PolygonFinancials(PolygonBase):
    def __init__(self, ticker: str) -> None:
        super().__init__()
        self.ticker = ticker

    def get_ticker(self) -> str:
        time.sleep(0.5)
        url = f"{self.polyv3}reference/tickers/{self.ticker}"

        try:
            response = self.request(method="GET", url=url)
            return response.get("results", [])
        except Exception as e:
            logger.warning(f"Error getting ticker for {self.ticker}: {e}")
            return None

    def get_ticker_previous_close(self) -> float:
        url = f"{self.polyv2}aggs/ticker/{self.ticker}/prev"

        try:
            response = self.request(method="GET", url=url)
            return response.get("results", [])[0].get("c", 0)
        except Exception as e:
            logger.warning(f"Error getting previous close for {self.ticker}: {e}")
            return 0

    def get_financials(self, limit: int = 1, timeframe: str = "annual") -> dict:
        """
        Get the financials for a stock.

        Args:
            ticker (str): The stock ticker to get financials for.
            limit (int): The number of financials to return.

        Returns:
            dict: The financials for the stock.
        """
        url = f"{self.polyvx}reference/financials"
        params = {
            "ticker": self.ticker,
            "timeframe": timeframe,  # "annual" or "quarterl
            "limit": limit,
        }

        """
        * The response will be a dictionary with the following keys:
        Long-Term Debt over Earnings - long_term_debt_over_earnings (long-term debt / earnings)
        Capital Expenditure over Net Income - capital_expenditure_over_net_income (capital expenditure / net income)
        P/E Ratio - pe_ratio (stock price / earnings per share)
        Net Margin - net_margin (net income / total revenue)
        Gross Margin - gross_margin (gross profit / total revenue)
        """

        try:
            response = self.request(method="GET", url=url, params=params)

            if response.get("results", []) == []:
                logger.warning(f"No financials found for {self.ticker}")
                return None

            results = response.get("results", [])[0].get("financials", [])

            # pprint(results)

            balance_sheet_statement = results.get("balance_sheet", [])
            # pprint(balance_sheet_statement)
            cash_flow_statement = results.get("cash_flow_statement", [])

            income_statement = results.get("income_statement", [])

            gross_profit = income_statement.get("gross_profit", []).get("value", 0)
            total_revenue = income_statement.get("revenues", []).get("value", 0)
            long_term_debt = balance_sheet_statement.get("long_term_debt", []).get(
                "value", 0
            )

            earnings_per_share = income_statement.get(
                "basic_earnings_per_share", 0
            ).get("value", 0)

            net_cash_flow_from_investing_activities = cash_flow_statement.get(
                "net_cash_flow_from_investing_activities", 0
            ).get("value", 0)

            net_income_loss = income_statement.get("net_income_loss", 0).get("value", 0)

            total_assets = balance_sheet_statement.get("current_assets", 0).get(
                "value", 0
            )

            gross_margin = self._calc_gross_margin(gross_profit, total_revenue)
            net_margin = self._calc_net_margin(net_income_loss, total_revenue)
            long_term_debt_over_earnings = self._calc_long_term_debt_over_earnings(
                long_term_debt, total_assets
            )
            capital_expenditure_over_net_income = (
                self._calc_capital_expenditure_over_net_income(
                    net_cash_flow_from_investing_activities, net_income_loss
                )
            )
            pe_ratio = self._calc_pe_ratio(
                self.get_ticker_previous_close(), earnings_per_share
            )

            return {
                "ticker": self.ticker,
                "name": self.get_ticker().get("name", ""),
                "industry": self.get_ticker().get("sic_description", ""),
                "mc": int(
                    self.get_ticker().get("market_cap", 0)
                ),  # Market Capitalization
                "gm": round(gross_margin, 2),  # Gross Margin Percentage
                "nm": round(net_margin, 2),  # Net Margin Percentage
                "ltdoe": round(
                    long_term_debt_over_earnings, 2
                ),  # Long-Term Debt over Earnings Percentage
                "ceoni": round(
                    capital_expenditure_over_net_income, 2
                ),  # Capital Expenditure over Net Income Percentage
                "pe": round(pe_ratio, 2),  # P/E Ratio
            }

        except Exception as e:
            logger.warning(f"Error getting daily history for {self.ticker}: {e}")

    def _calc_gross_margin(self, gross_profit: float, total_revenue: float) -> float:
        return (gross_profit / total_revenue) * 100

    def _calc_net_margin(self, net_income: float, total_revenue: float) -> float:
        return (net_income / total_revenue) * 100

    def _calc_long_term_debt_over_earnings(
        self, long_term_debt: float, net_income: float
    ) -> float:
        return long_term_debt / net_income

    def _calc_capital_expenditure_over_net_income(
        self, capital_expenditure: float, net_income: float
    ) -> float:
        return capital_expenditure / net_income

    def _calc_pe_ratio(self, stock_price: float, earnings_per_share: float) -> float:
        return stock_price / earnings_per_share
