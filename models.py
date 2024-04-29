import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import exp, sqrt

def create_gru_model(input_shape):
    model = Sequential([
        GRU(50, activation='relu', return_sequences=True, input_shape=input_shape),
        GRU(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        LSTM(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def create_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(50, activation='relu', return_sequences=True, input_shape=input_shape),
        SimpleRNN(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def create_dnn_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')  # or another activation function depending on your output
    ])
    return model

# Define other models similarly...

def train_model(model, train_data, train_labels, epochs=10, batch_size=32):
    return model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=1)

def evaluate_model(model, test_data, actual_rul):
    predicted_rul = model.predict(test_data).flatten()
    rmse = sqrt(mean_squared_error(actual_rul, predicted_rul))
    score = np.sum([score_function(actual, pred) for actual, pred in zip(actual_rul, predicted_rul)])
    return rmse, score

def score_function(actual, predicted):
    h_j = predicted - actual
    if h_j < 0:
        return exp(-h_j / 13) - 1
    else:
        return exp(-h_j / 10) - 1