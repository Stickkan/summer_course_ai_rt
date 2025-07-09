import numpy as np
import pandas as pd
import glob
from collections import Counter
from pandas.core import window
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from dataclasses import dataclass
from src.feature_extraction import extract_features
from src.pre_processor import PreProcessor


@dataclass
class ModelConfig:
    window_size: int
    overlap: float
    fs: int
    lowcut: int
    highcut: int
    filter_order: int
    wamp_threshold: float
    features: list[str]

@dataclass
class DataConfig:
    step: int
    data_len: int
    windows_count: int


def get_csv_file_list(dir: str) -> list[str]:
    files = glob.glob(dir + "/*.csv")
    return files


def get_file_data(file_path: str):
    df = pd.read_csv(file_path, sep=' ', header=0)

    signals = []
    for i in df.columns[0:-1]:
        signals.append(df[i].to_numpy())

    labels = df['label'].to_numpy()

    return (signals, labels, len(labels))


def get_majority_label(window):
    return Counter(window).most_common(1)[0][0]


def get_data_config(signals, data_len, config: ModelConfig) -> DataConfig:
    step = int(config.window_size * config.overlap)
    windows_count = (data_len - config.window_size) // step + 1

    return DataConfig(step=step, data_len=data_len, windows_count=windows_count)


def to_windows(signals, step, window_size, windows_count):
    windows = []
    for i in range(windows_count):
        start = i * step
        end = start + step
        window = signals[start:end]
        windows.append(window)

    return windows


def process_data(signals, data_config: DataConfig, config: ModelConfig):
    X = []
    for val in signals:
        windows = to_windows(signals=val, step=data_config.step, window_size=config.window_size, windows_count=data_config.windows_count)

        x = []
        for window in windows:
            x.append(extract_features(window=window, features=config.features, wamp_threshold=config.wamp_threshold))
        X.append(x)
    return X


def process_labels(labels, data_config: DataConfig, config: ModelConfig) -> list[int]:
    windowed_labels = to_windows(signals=labels, step=data_config.step, window_size=config.window_size, windows_count=data_config.windows_count)

    Y = []
    for w in windowed_labels:
        Y.append(get_majority_label(w))
    return Y

# def process_labels(labels: list[list[int]]) -> list[int]:
#     window_labels = []
#     for w in labels:
#         window_labels.append(get_majority_label(w))
#     return window_labels

def build_lstm_model(input_shape):
    model = Sequential(
        [
            InputLayer(input_shape=input_shape, unroll=True),
            LSTM(15, unroll=True),
            Dense(6, activation='relu'),
            Dense(2, activation='softmax')
        ]
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



if __name__ == '__main__':

    #  Some values for testing
    model_config = ModelConfig(
        window_size=200,
        overlap=0.5,
        fs=0,
        lowcut=20,
        highcut=450,
        filter_order=4,
        wamp_threshold=0.02,
        features=['mav', 'wl', 'wamp', 'mavs']
    )

    input_dir = "data/DB4_prepared/"  # Folder to glob
    input_files = get_csv_file_list(input_dir)  # Get csv files globbed
    (signals, labels, data_len) = get_file_data(input_files[2])  # Get the contents of a csv-file

    data_config = get_data_config(labels, data_len, model_config)
    processed_label_windows = process_labels(labels, data_config, model_config)
    processed_signals = process_data(signals, data_config, model_config)

    print('no')

    # training_data, validation_data = get_dataframe("")

    # model = build_lstm_model((10, 5))
    # model.summary()
