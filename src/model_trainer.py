import numpy as np
import pandas as pd
import glob
import os
import joblib
from collections import Counter
from pandas.core import window
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from dataclasses import dataclass
from feature_extraction import extract_features
from pre_processor import PreProcessor


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

# @dataclass
# class DataConfig:
#     step: int
#     data_len: int
#     windows_count: int


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


def get_multiple_files_data(files: list[str], step: int, window_size: int):
    data = []
    for f in files:
        signals, labels, data_len = get_file_data(f)
        windows_count = (data_len - window_size) // step + 1
        data.append({'signals': signals, 'labels': labels, 'data_len': data_len, 'windows_count': windows_count})
    return data


def get_majority_label(window):
    return Counter(window).most_common(1)[0][0]


def to_windows(signals, step, windows_count):
    windows = []
    for i in range(windows_count):
        start = i * step
        end = start + step
        window = signals[start:end]
        windows.append(window)

    return windows


def process_multiple_files(data, step, config: ModelConfig):
    processed_data = []
    for file_data in data:
        X = process_data(
            signals=file_data['signals'],
            step=step,
            windows_count=file_data['windows_count'],
            config=config
        )
        Y = process_labels(
            labels=file_data['labels'],
            step=step,
            windows_count=file_data['windows_count'],
            config=config
        )
        processed_data.append({'X': X, 'Y': Y})
        print(f"Processed {len(processed_data)} files")
    return processed_data


def process_data(signals, step: int, windows_count, config: ModelConfig):
    X = []
    for val in signals:
        windows = to_windows(signals=val, step=step, windows_count=windows_count)

        x = []
        for window in windows:
            x.append(extract_features(window=window, features=config.features, wamp_threshold=config.wamp_threshold))
        X.append(x)
    return X


def process_labels(labels, step: int, windows_count, config: ModelConfig) -> list[int]:
    windowed_labels = to_windows(signals=labels, step=step, windows_count=windows_count)

    Y = []
    for w in windowed_labels:
        Y.append(get_majority_label(w))
    return Y


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


def pre_process(input_dir: str, output_dir: str, fname: str, config: ModelConfig) -> None:
    output_file = os.path.join(output_dir, fname + '.pkl')

    input_files = get_csv_file_list(input_dir)  # Get csv files globbed
    step = int(model_config.window_size * model_config.overlap)
    test_data = get_multiple_files_data(files=input_files, step=step, window_size=config.window_size)
    processed_test_data = process_multiple_files(data=test_data, step=step, config=config)
    joblib.dump(processed_test_data, output_file)
    print(f"Dumped {output_file}")

if __name__ == '__main__':
    print(os.getcwd())
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

    # input_dir = "data/DB4_prepared/"  # Folder to glob
    input_dir = os.path.join('data', 'DB4_prepared')
    output_dir = "model"
    output_file = 'ninapro_DB4_4_emg'

    pre_process(input_dir, output_dir, output_file, model_config)

    # input_files = get_csv_file_list(input_dir)  # Get csv files globbed
    # step = int(model_config.window_size * model_config.overlap)
    # # (signals, labels, data_len) = get_file_data(input_files[0])  # Get the contents of a csv-file
    # test_data = get_multiple_files_data(files=input_files, step=step, window_size=model_config.window_size)
    # processed_test_data = process_multiple_files(data=test_data, config=model_config)
    # joblib.dump(processed_test_data, output_dir + 'test_data.pkl')

    # data_config = get_data_config(labels, data_len, model_config)
    # processed_label_windows = process_labels(labels, data_config, model_config)
    # processed_signals = process_data(signals, data_config, model_config)


    # training_data, validation_data = get_dataframe("")

    # model = build_lstm_model((10, 5))
    # model.summary()
