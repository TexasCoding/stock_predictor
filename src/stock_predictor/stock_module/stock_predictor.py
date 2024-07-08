# src/stock_predictor/stock_module/stock_predictor.py
# Description: Class to predict stock prices using LSTM neural network.
import os
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM, Dropout, Input
import matplotlib.pyplot as plt

from stock_predictor.global_settings import VERBOSE, logger
from stock_predictor.stock_module.stock_base import StockBase

# Window size or the sequence length, 7 (1 week)
N_STEPS = 7
# Lookup steps, 1 is the next day, 3 = after tomorrow
LOOKUP_STEPS = [1, 2, 3]

if VERBOSE > 0:
    tf.get_logger().setLevel("WARNING")
    logger.setLevel("WARNING")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
else:
    tf.get_logger().setLevel("ERROR")
    logger.setLevel("ERROR")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


class StockPredictor(StockBase):
    def __init__(self, symbol: str, data_df: pd.DataFrame):
        super().__init__()
        self.symbol = symbol
        self.data_df = data_df[["vwap", "date", "close"]].copy()

        self.scaler = MinMaxScaler()
        self.data_df["scaled_vwap"] = self.scaler.fit_transform(
            np.expand_dims(self.data_df["vwap"].values, axis=1)
        )

    def prepare_data(self, days: int):
        """Prepare data for model training and prediction.

        Args:
            days (int): Number of future days to predict.

        Returns:
            Tuple containing DataFrame, last sequence, x_train, and y_train.
        """
        df = self.data_df.copy()
        df["future"] = df["scaled_vwap"].shift(-days)
        last_sequence = np.array(df[["scaled_vwap"]].tail(days))
        df.dropna(inplace=True)
        sequence_data = []
        sequences = deque(maxlen=N_STEPS)

        for entry, target in zip(
            df[["scaled_vwap", "date"]].values, df["future"].values
        ):
            sequences.append(entry)
            if len(sequences) == N_STEPS:
                sequence_data.append([np.array(sequences), target])

        last_sequence = list([s[:1] for s in sequences]) + list(last_sequence)
        last_sequence = np.array(last_sequence).astype(np.float32)

        # construct the X's and Y's
        X, Y = zip(*sequence_data)
        X, Y = np.array(X), np.array(Y)

        return df, last_sequence, X, Y

    def get_trained_model(self, x_train: np.array, y_train: np.array) -> Sequential:
        """Train the LSTM model.

        Args:
            x_train (np.array): Training data features.
            y_train (np.array): Training data labels.

        Returns:
            Sequential: Trained model.
        """
        BATCH_SIZE = 8
        EPOCHS = 80

        model = Sequential(
            [
                Input(shape=(N_STEPS, 1)),
                LSTM(60, return_sequences=True),
                Dropout(0.3),
                LSTM(120, return_sequences=False),
                Dropout(0.3),
                Dense(20),
                Dense(1),
            ]
        )

        model.compile(loss="mean_squared_error", optimizer="adam")

        model.fit(
            x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE
        )

        if VERBOSE > 0:
            model.summary()

        return model

    def get_predictions(self) -> Optional[tuple]:
        """Generate predictions for the upcoming days.

        Returns:
            Optional[tuple]: Model, predictions, training data features, and labels.
        """
        predictions = []

        for step in LOOKUP_STEPS:
            df, last_sequence, x_train, y_train = self.prepare_data(step)
            x_train = x_train[:, :, :1].astype(np.float32)

            model = self.get_trained_model(x_train, y_train)

            last_sequence = last_sequence[-N_STEPS:]
            last_sequence = np.expand_dims(last_sequence, axis=0)
            prediction = model.predict(last_sequence, verbose=VERBOSE)
            predicted_price = self.scaler.inverse_transform(prediction)[0][0]

            predictions.append(round(float(predicted_price), 2))

        if predictions:
            predictions_str = ", ".join(f"{d}$" for d in predictions)
            message = (
                f"{self.symbol} prediction for upcoming 3 days: ({predictions_str})"
            )
            if VERBOSE > 0:
                print(message)

            return model, predictions, x_train, y_train

        return None

    def execute_model(self) -> pd.DataFrame:
        """Execute the prediction model and update the DataFrame with predicted values.

        Returns:
            pd.DataFrame: DataFrame with updated predictions.
        """
        model, predictions, x_train, y_train = self.get_predictions()
        copy_df = self.data_df.copy()
        y_predicted = model.predict(x_train, verbose=VERBOSE)
        y_predicted_transformed = np.squeeze(self.scaler.inverse_transform(y_predicted))
        copy_df["predicted_close"] = np.concatenate(
            ([0] * 6, y_predicted_transformed, y_train[-3:]), axis=0
        )

        date_now = self.calendar.future_dates.next_day2
        date_tomorrow = self.calendar.future_dates.next_day3
        date_after_tomorrow = self.calendar.future_dates.next_day4

        prediction_dict = {
            date_now: predictions[0],
            date_tomorrow: predictions[1],
            date_after_tomorrow: predictions[2],
        }

        for date, prediction in prediction_dict.items():
            copy_df.loc[date] = [prediction, date, 0, 0]

        return copy_df

    def plot_chart(self, data_df: pd.DataFrame) -> None:
        """Plot the resulting chart with actual, predicted prices, and VWAP.

        Args:
            data_df (pd.DataFrame): DataFrame containing stock data and predictions.
        """
        plt.style.use(style="ggplot")
        plt.figure(figsize=(16, 10))
        plt.plot(data_df["close"][-150:].head(147))
        plt.plot(
            data_df["vwap"][-150:].head(147),
            linewidth=1,
            linestyle="dashed",
            color="red",
        )
        plt.plot(
            data_df["predicted_close"][-150:].head(147), linewidth=1, linestyle="dashed"
        )
        plt.plot(data_df["vwap"][-150:].tail(4))
        plt.xlabel("days")
        plt.ylabel("price")
        plt.legend(
            [
                f"Actual price for {self.symbol}",
                f"VWAP price for {self.symbol}",
                f"Predicted price for {self.symbol}",
                "Predicted price for future 3 days",
            ]
        )
        plt.show()

    def plot_clean_chart(self, symbol: str) -> None:
        """Plot a clean chart with the actual close prices.

        Args:
            symbol (str): The stock symbol.
        """
        data_df = self.data_df.copy()
        plt.style.use(style="ggplot")
        plt.figure(figsize=(16, 10))
        plt.plot(data_df["close"][-200:])
        plt.xlabel("days")
        plt.ylabel("price")
        plt.legend([f"Actual price for {symbol}"])
        plt.show()


# from stock_predictor.global_settings import VERBOSE, logger

# import os
# import numpy as np
# import pandas as pd

# import tensorflow as tf

# # Data preparation
# from sklearn.preprocessing import MinMaxScaler
# from collections import deque

# # AI
# from keras.api.models import Sequential
# from keras.api.layers import Dense, LSTM, Dropout, Input
# from stock_predictor.stock_module.stock_base import StockBase

# # Graphics library
# import matplotlib.pyplot as plt

# # Window size or the sequence length, 7 (1 week)
# N_STEPS = 7
# # Lookup steps, 1 is the next day, 3 = after tomorrow
# LOOKUP_STEPS = [1, 2, 3]

# if VERBOSE > 0:
#     tf.get_logger().setLevel("WARNING")
#     logger.setLevel("WARNING")
#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# else:
#     tf.get_logger().setLevel("ERROR")
#     logger.setLevel("ERROR")
#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


# class StockPredictor(StockBase):
#     def __init__(self, symbol: str, data_df: pd.DataFrame):
#         super().__init__()
#         self.symbol = symbol
#         self.data_df = data_df[["vwap", "date", "close"]].copy()

#         self.scaler = MinMaxScaler()
#         self.data_df["scaled_vwap"] = self.scaler.fit_transform(
#             np.expand_dims(self.data_df["vwap"].values, axis=1)
#         )

#     def PrepareData(self, days):
#         df = self.data_df.copy()
#         df["future"] = df["scaled_vwap"].shift(-days)
#         last_sequence = np.array(df[["scaled_vwap"]].tail(days))
#         df.dropna(inplace=True)
#         sequence_data = []
#         sequences = deque(maxlen=N_STEPS)

#         for entry, target in zip(
#             df[["scaled_vwap"] + ["date"]].values, df["future"].values
#         ):
#             sequences.append(entry)
#             if len(sequences) == N_STEPS:
#                 sequence_data.append([np.array(sequences), target])

#         last_sequence = list([s[: len(["scaled_vwap"])] for s in sequences]) + list(
#             last_sequence
#         )
#         last_sequence = np.array(last_sequence).astype(np.float32)

#         # construct the X's and Y's
#         X, Y = [], []
#         for seq, target in sequence_data:
#             X.append(seq)
#             Y.append(target)

#         # convert to numpy arrays
#         X = np.array(X)
#         Y = np.array(Y)

#         return df, last_sequence, X, Y

#     @staticmethod
#     def GetTrainedModel(x_train, y_train):
#         BATCH_SIZE = 8
#         EPOCHS = 80

#         model = Sequential(
#             [
#                 Input(shape=(BATCH_SIZE, EPOCHS)),
#                 LSTM(60, return_sequences=True),
#                 Dropout(0.3),
#                 LSTM(120, return_sequences=False),
#                 Dropout(0.3),
#                 Dense(20),
#                 Dense(1),
#             ]
#         )

#         model.compile(loss="mean_squared_error", optimizer="adam")

#         model.fit(
#             x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE
#         )

#         if VERBOSE > 0:
#             model.summary()

#         return model

#     def GetPredictions(self) -> Sequential:
#         # GET PREDICTIONS
#         predictions = []

#         for step in LOOKUP_STEPS:
#             df, last_sequence, x_train, y_train = self.PrepareData(step)
#             x_train = x_train[:, :, : len(["scaled_vwap"])].astype(np.float32)

#             model: Sequential = self.GetTrainedModel(x_train, y_train)

#             last_sequence = last_sequence[-N_STEPS:]
#             last_sequence = np.expand_dims(last_sequence, axis=0)
#             prediction = model.predict(last_sequence, verbose=VERBOSE)
#             predicted_price = self.scaler.inverse_transform(prediction)[0][0]

#             predictions.append(round(float(predicted_price), 2))

#         if bool(predictions) is True and len(predictions) > 0:
#             predictions_list = [str(d) + "$" for d in predictions]
#             predictions_str = ", ".join(predictions_list)
#             message = (
#                 f"{self.symbol} prediction for upcoming 3 days ({predictions_str})"
#             )
#             if VERBOSE > 0:
#                 print(message)

#             return model, predictions, x_train, y_train

#         return None

#     def ExecuteModel(self):
#         model, predictions, x_train, y_train = self.GetPredictions()
#         copy_df = self.data_df.copy()
#         y_predicted = model.predict(x_train, verbose=VERBOSE)
#         y_predicted_transformed = np.squeeze(self.scaler.inverse_transform(y_predicted))
#         first_seq = self.scaler.inverse_transform(np.expand_dims(y_train[:6], axis=1))
#         last_seq = self.scaler.inverse_transform(np.expand_dims(y_train[-3:], axis=1))
#         y_predicted_transformed = np.append(first_seq, y_predicted_transformed)
#         y_predicted_transformed = np.append(y_predicted_transformed, last_seq)
#         copy_df["predicted_close"] = y_predicted_transformed

#         date_now = self.calendar.future_dates.next_day2
#         date_tomorrow = self.calendar.future_dates.next_day3
#         date_after_tomorrow = self.calendar.future_dates.next_day4

#         copy_df.loc[date_now] = [predictions[0], f"{date_now}", 0, 0]
#         copy_df.loc[date_tomorrow] = [predictions[1], f"{date_tomorrow}", 0, 0]
#         copy_df.loc[date_after_tomorrow] = [
#             predictions[2],
#             f"{date_after_tomorrow}",
#             0,
#             0,
#         ]

#         return copy_df

#     def plot_chart(self, data_df: pd.DataFrame) -> None:
#         # Result chart
#         plt.style.use(style="ggplot")
#         plt.figure(figsize=(16, 10))
#         plt.plot(data_df["close"][-150:].head(147))
#         plt.plot(
#             data_df["vwap"][-150:].head(147),
#             linewidth=1,
#             linestyle="dashed",
#             color="red",
#         )
#         plt.plot(
#             data_df["predicted_close"][-150:].head(147), linewidth=1, linestyle="dashed"
#         )
#         plt.plot(data_df["vwap"][-150:].tail(4))
#         plt.xlabel("days")
#         plt.ylabel("price")
#         plt.legend(
#             [
#                 f"Actual price for {self.symbol}",
#                 f"VWAP price for {self.symbol}",
#                 f"Predicted price for {self.symbol}",
#                 "Predicted price for future 3 days",
#             ]
#         )
#         plt.show()

#     def plot_clean_chart(self, symbol: str) -> None:
#         data_df = self.data_df.copy()
#         print(data_df)
#         # Let's preliminary see our data on the graphic
#         plt.style.use(style="ggplot")
#         plt.figure(figsize=(16, 10))
#         plt.plot(data_df["close"][-200:])
#         plt.xlabel("days")
#         plt.ylabel("price")
#         plt.legend([f"Actual price for {symbol}"])
#         plt.show()
