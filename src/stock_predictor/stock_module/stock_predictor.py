# src/stock_predictor/stock_module/stock_predictor.py
# Description: Class to predict stock prices using LSTM neural network.
from stock_predictor.global_settings import VERBOSE, logger

import os
import numpy as np
import pandas as pd

import tensorflow as tf

# Data preparation
from sklearn.preprocessing import MinMaxScaler
from collections import deque

# AI
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM, Dropout, Input
from stock_predictor.stock_module.stock_base import StockBase

# Window size or the sequence length, 7 (1 week)
N_STEPS = 7
# Lookup steps, 1 is the next day, 3 = after tomorrow
LOOKUP_STEPS = [1, 2, 3]

if VERBOSE > 0:
    tf.get_logger().setLevel("ERROR")
    logger.setLevel("ERROR")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
else:
    tf.get_logger().setLevel("ERROR")
    logger.setLevel("ERROR")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class StockPredictor(StockBase):
    def __init__(self, symbol: str, data_df: pd.DataFrame):
        super().__init__()
        self.symbol = symbol
        self.data_df = data_df[["close", "open", "low", "high", "volume"]].copy()

        self.data_df["date"] = self.data_df.index
        self.data_df["symbol"] = self.symbol

        self.scaler = MinMaxScaler()
        self.data_df["scaled_close"] = self.scaler.fit_transform(
            np.expand_dims(self.data_df["close"].values, axis=1)
        )

    ###############################################################
    # Prepare the data
    ###############################################################
    def prepare_data(self, days):
        """
        Prepare the data for training the stock predictor model.

        Args:
            days (int): The number of days to look ahead for prediction.

        Returns:
            tuple: A tuple containing the following elements:
                - df (pandas.DataFrame): The modified data DataFrame.
                - last_sequence (numpy.ndarray): The last sequence of data.
                - X (numpy.ndarray): The input data for training.
                - Y (numpy.ndarray): The target data for training.
        """

        df = self.data_df.copy()
        df["future"] = df["scaled_close"].shift(-days)
        last_sequence = np.array(df[["scaled_close"]].tail(days))
        df.dropna(inplace=True)
        sequence_data = []
        sequences = deque(maxlen=N_STEPS)

        for entry, target in zip(
            df[["scaled_close"] + ["date"]].values, df["future"].values
        ):
            sequences.append(entry)
            if len(sequences) == N_STEPS:
                sequence_data.append([np.array(sequences), target])

        last_sequence = list([s[: len(["scaled_close"])] for s in sequences]) + list(
            last_sequence
        )
        last_sequence = np.array(last_sequence).astype(np.float32)

        # construct the X's and Y's
        X, Y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            Y.append(target)

        # convert to numpy arrays
        X = np.array(X)
        Y = np.array(Y)

        return df, last_sequence, X, Y

    ###############################################################
    # Get the trained model
    ###############################################################
    @staticmethod
    def get_trained_model(x_train, y_train):
        """
        Trains and returns a stock prediction model.

        Args:
            x_train (numpy.ndarray): The input training data.
            y_train (numpy.ndarray): The target training data.

        Returns:
            keras.models.Sequential: The trained stock prediction model.
        """
        BATCH_SIZE = 8
        EPOCHS = 80

        model = Sequential(
            [
                Input(shape=(BATCH_SIZE, EPOCHS)),
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

    ###############################################################
    # Get the predictions
    ###############################################################
    def get_predictions(self) -> Sequential:
        """
        Get predictions for the upcoming 3 days.

        Returns:
            - model: The trained model used for predictions.
            - predictions: A list of predicted prices for the upcoming 3 days.
            - x_train: The input data used for training the model.
            - y_train: The target data used for training the model.
        """
        predictions = []

        for step in LOOKUP_STEPS:
            df, last_sequence, x_train, y_train = self.prepare_data(step)
            x_train = x_train[:, :, : len(["scaled_close"])].astype(np.float32)

            model: Sequential = self.get_trained_model(x_train, y_train)

            last_sequence = last_sequence[-N_STEPS:]
            last_sequence = np.expand_dims(last_sequence, axis=0)
            prediction = model.predict(last_sequence, verbose=VERBOSE)
            predicted_price = self.scaler.inverse_transform(prediction)[0][0]

            predictions.append(round(float(predicted_price), 2))

        if bool(predictions) is True and len(predictions) > 0:
            predictions_list = [str(d) + "$" for d in predictions]
            predictions_str = ", ".join(predictions_list)
            message = (
                f"{self.symbol} prediction for upcoming 3 days ({predictions_str})"
            )
            if VERBOSE > 0:
                print(message)

            return model, predictions, x_train, y_train

        return None

    ###############################################################
    # Execute the model
    ###############################################################
    def execute_model(self):
        """
        Executes the stock predictor model and returns a copy of the data frame with predicted values.

        Returns:
            pandas.DataFrame: A copy of the data frame with predicted values.
        """
        model, predictions, x_train, y_train = self.get_predictions()
        copy_df = self.data_df.copy()
        y_predicted = model.predict(x_train, verbose=VERBOSE)
        y_predicted_transformed = np.squeeze(self.scaler.inverse_transform(y_predicted))
        first_seq = self.scaler.inverse_transform(np.expand_dims(y_train[:6], axis=1))
        last_seq = self.scaler.inverse_transform(np.expand_dims(y_train[-3:], axis=1))
        y_predicted_transformed = np.append(first_seq, y_predicted_transformed)
        y_predicted_transformed = np.append(y_predicted_transformed, last_seq)
        copy_df["predicted_close"] = y_predicted_transformed

        date_now = self.calendar.future_dates.next_day1
        date_tomorrow = self.calendar.future_dates.next_day2
        date_after_tomorrow = self.calendar.future_dates.next_day3

        copy_df.loc[date_now] = [
            # copy_df["symbol"].iloc[-1],
            # copy_df["vwap"].iloc[-1],
            # f"{date_now}",
            copy_df["close"].iloc[-1],
            copy_df["close"].iloc[-1],
            copy_df["close"].iloc[-1],
            copy_df["close"].iloc[-1],
            0,
            f"{date_tomorrow}",
            copy_df["symbol"].iloc[-1],
            0,
            predictions[0],
        ]
        copy_df.loc[date_tomorrow] = [
            # copy_df["symbol"].iloc[-1],
            # copy_df["vwap"].iloc[-1],
            copy_df["close"].iloc[-1],
            copy_df["close"].iloc[-1],
            copy_df["close"].iloc[-1],
            copy_df["close"].iloc[-1],
            0,
            f"{date_tomorrow}",
            copy_df["symbol"].iloc[-1],
            0,
            predictions[1],
        ]
        copy_df.loc[date_after_tomorrow] = [
            # copy_df["vwap"].iloc[-1],
            # f"{date_after_tomorrow}",
            copy_df["close"].iloc[-1],
            copy_df["close"].iloc[-1],
            copy_df["close"].iloc[-1],
            copy_df["close"].iloc[-1],
            0,
            f"{date_tomorrow}",
            copy_df["symbol"].iloc[-1],
            0,
            predictions[2],
        ]

        copy_df["date"] = pd.to_datetime(copy_df["date"])
        copy_df.index = pd.to_datetime(copy_df["date"])

        return copy_df
