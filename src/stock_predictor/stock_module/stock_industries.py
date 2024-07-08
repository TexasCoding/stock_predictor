# # src/stock_trader/stock_module/stock_industries.py
# # This module contains the Industries class, which is used to retrieve and calculate various metrics for different industries.
import os
import pandas as pd
from stock_predictor.stock_module.stock_base import StockBase
from stock_predictor.stock_module.stock_tickers import StockTickers

from dotenv import load_dotenv

load_dotenv()

REDUCE_BY_PERCENTAGE = float(os.getenv("DEFAULT_AVG_REDUCTION", 0.2))


class StockIndustries(StockBase):
    def __init__(self, average_reduction: float = REDUCE_BY_PERCENTAGE):
        """
        Initialize the StockIndustries class.

        Parameters:
            average_reduction (float): The average reduction value to be used. Defaults to REDUCE_BY_PERCENTAGE.
        """
        super().__init__()
        self.tickers: pd.DataFrame = StockTickers().get_stored_tickers_df()
        self.average_reduction = average_reduction

    ############################
    # Get unique industries list
    ############################
    def get_unique_industries(self) -> list[str]:
        """
        Returns a list of unique industries from the tickers DataFrame.

        Returns:
            list[str]: A list of unique industries.
        """
        return self.tickers["industry"].unique().tolist()

    ############################
    # Get stocks by industry
    ############################
    def get_stocks_by_industry(
        self, industry: str, sort_by: str = "symbol", ascending: bool = True
    ) -> pd.DataFrame:
        """
        Retrieves stocks by industry.

        Args:
            industry (str): The industry to filter stocks by.
            sort_by (str, optional): The column to sort the stocks by. Defaults to "symbol".
                * Allowed values are "symbol", "name", "gross_margin_pct", "net_margin_pct", "trailing_pe", "forward_pe".
            ascending (bool, optional): Whether to sort the stocks in ascending order. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the stocks filtered by industry.
        """
        self.stock_sorter_check(sort_by)
        return self.tickers[self.tickers["industry"] == industry].sort_values(
            by=sort_by, ascending=ascending
        )

    ############################
    # Industry average DataFrame
    ############################
    def industry_avg_df(
        self, sort_by: str = "count", ascending: bool = True
    ) -> pd.DataFrame:
        """
        Calculate the average values of various metrics for each industry and return the results as a DataFrame.

        Args:
            sort_by (str, optional): The column to sort the DataFrame by. Defaults to "count".
                * Allowed values are "industry", "count", "gross_margin_avg", "net_margin_avg", "trailing_pe_avg", "forward_pe_avg".
            ascending (bool, optional): Whether to sort the DataFrame in ascending order. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the average values of metrics for each industry, sorted according to the specified column.

        Raises:
            ValueError: If the specified sort_by column is not valid.
        """
        self.industry_sorter_check(sort_by)

        industry_metrics = (
            self.tickers.groupby("industry")
            .agg(
                count=("industry", "size"),
                gross_margin_avg=(
                    "gross_margin_pct",
                    lambda x: round(x.mean() * (1 - self.average_reduction), 2),
                ),
                net_margin_avg=(
                    "net_margin_pct",
                    lambda x: round(x.mean() * (1 - self.average_reduction), 2),
                ),
                trailing_pe_avg=(
                    "trailing_pe",
                    lambda x: round(x.mean() * (1 - self.average_reduction), 2),
                ),
                forward_pe_avg=(
                    "forward_pe",
                    lambda x: round(x.mean() * (1 - self.average_reduction), 2),
                ),
            )
            .reset_index()
        )

        return industry_metrics.sort_values(
            by=sort_by, ascending=ascending
        ).reset_index(drop=True)

    ############################
    # Average metric by industry
    ############################
    def _average_metric_by_industry(self, industry: str, metric: str) -> float:
        """
        Calculate the average of a given metric for a specified industry.

        Args:
            industry (str): The industry for which to calculate the metric.
            metric (str): The metric to calculate the average for.

        Returns:
            float: The average value of the specified metric for the industry.
        """
        return round(
            self.tickers[self.tickers["industry"] == industry][metric].mean()
            * (1 - self.average_reduction),
            2,
        )

    ############################
    # Count tickers per industry
    ############################
    def count_tickers_per_industry(self, industry: str) -> int:
        """
        Count the number of tickers for a given industry.

        Args:
            industry (str): The industry to count tickers for.

        Returns:
            int: The number of tickers in the specified industry.
        """
        return len(self.get_stocks_by_industry(industry))
