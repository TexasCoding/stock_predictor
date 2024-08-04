from stock_predictor.stock_module.stock_screener import StockScreener
from stock_predictor.stock_module.stock_charts import StockCharts
from rich.console import Console

import warnings

warnings.filterwarnings("ignore")


def main():
    console = Console()

    screened_df = StockScreener().screener()

    if not screened_df.empty:
        chart = StockCharts()
        filtered_tickers = screened_df.symbol.unique().tolist()
        for ticker in filtered_tickers:
            df = screened_df[screened_df["symbol"] == ticker]
            console.print(f"[bold magenta]{ticker}")
            chart.plot_prediction_chart(df)


if __name__ == "__main__":
    main()
