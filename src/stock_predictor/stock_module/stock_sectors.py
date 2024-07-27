# src/stock_trader/stock_module/stock_sectors.py
# This module contains the sectors class, which is used to retrieve and calculate various metrics for different sectors.
import os
import pandas as pd
from stock_predictor.stock_module.stock_base import StockBase
from stock_predictor.stock_module.stock_tickers import StockTickers

from dotenv import load_dotenv

load_dotenv()

REDUCE_BY_PERCENTAGE = float(os.getenv("DEFAULT_AVG_REDUCTION", 0.20))


class StockSectors(StockBase):
    def __init__(self, average_reduction: float = REDUCE_BY_PERCENTAGE):
        """
        Initialize the Stocksectors class.

        Parameters:
            average_reduction (float): The average reduction value to be used. Defaults to REDUCE_BY_PERCENTAGE.
        """
        super().__init__()
        self.tickers: pd.DataFrame = StockTickers().get_stored_tickers_df()
        self.average_reduction = average_reduction

    ############################
    # Get unique sectors list
    ############################
    def get_unique_sectors(self) -> list[str]:
        """
        Returns a list of unique sectors from the tickers DataFrame.

        Returns:
            list[str]: A list of unique sectors.
        """
        return self.tickers["sector"].unique().tolist()

    ############################
    # Get stocks by sector
    ############################
    def get_stocks_by_sector(
        self, sector: str, sort_by: str = "symbol", ascending: bool = True
    ) -> pd.DataFrame:
        """
        Retrieves stocks by sector.

        Args:
            sector (str): The sector to filter stocks by.
            sort_by (str, optional): The column to sort the stocks by. Defaults to "symbol".
                * Allowed values are "symbol", "name", "gross_margin_pct", "net_margin_pct", "trailing_pe", "forward_pe".
            ascending (bool, optional): Whether to sort the stocks in ascending order. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the stocks filtered by sector.
        """
        self.stock_sorter_check(sort_by)
        return self.tickers[self.tickers["sector"] == sector].sort_values(
            by=sort_by, ascending=ascending
        )

    ############################
    # sector average DataFrame
    ############################
    def sector_avg_df(
        self, sort_by: str = "count", ascending: bool = True
    ) -> pd.DataFrame:
        """
        Calculate the average values of various metrics for each sector and return the results as a DataFrame.

        Args:
            sort_by (str, optional): The column to sort the DataFrame by. Defaults to "count".
                * Allowed values are "sector", "count", "gross_margin_avg", "net_margin_avg", "trailing_pe_avg", "forward_pe_avg".
            ascending (bool, optional): Whether to sort the DataFrame in ascending order. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the average values of metrics for each sector, sorted according to the specified column.

        Raises:
            ValueError: If the specified sort_by column is not valid.
        """
        self.sector_sorter_check(sort_by)

        sector_metrics = (
            self.tickers.groupby("sector")
            .agg(
                count=("sector", "size"),
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
                piotroski_score_avg=(
                    "piotroski_score",
                    lambda x: int(x.mean() * (1 - self.average_reduction)),
                ),
            )
            .reset_index()
        )

        return sector_metrics.sort_values(by=sort_by, ascending=ascending).reset_index(
            drop=True
        )

    ############################
    # Average metric by sector
    ############################
    def _average_metric_by_sector(self, sector: str, metric: str) -> float:
        """
        Calculate the average of a given metric for a specified sector.

        Args:
            sector (str): The sector for which to calculate the metric.
            metric (str): The metric to calculate the average for.

        Returns:
            float: The average value of the specified metric for the sector.
        """
        return round(
            self.tickers[self.tickers["sector"] == sector][metric].mean()
            * (1 - self.average_reduction),
            2,
        )

    ############################
    # Count tickers per sector
    ############################
    def count_tickers_per_sector(self, sector: str) -> int:
        """
        Count the number of tickers for a given sector.

        Args:
            sector (str): The sector to count tickers for.

        Returns:
            int: The number of tickers in the specified sector.
        """
        return len(self.get_stocks_by_sector(sector))
