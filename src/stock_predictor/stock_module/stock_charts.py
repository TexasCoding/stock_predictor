import mplfinance as mpf
import pandas as pd


class StockCharts:
    def __init__(self) -> None:
        pass

    def plot_prediction_chart(self, history: pd.DataFrame) -> None:
        """
        Plots the stock price history and the predictions.

        Args:
            history (pd.DataFrame): The stock price history.
            predictions (pd.DataFrame): The stock price predictions.
        """
        # Plot the stock price history and the predictions
        mpf.plot(
            history,
            type="candle",
            style="charles",
            volume=True,
            addplot=[
                mpf.make_addplot(
                    history["close"],
                    color="orange",
                    linestyle="dashed",
                    panel=0,
                    secondary_y=True,
                ),
                mpf.make_addplot(
                    history["predicted_close"],
                    color="blue",
                    panel=0,
                    secondary_y=True,
                ),
            ],
        )
