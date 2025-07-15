import numpy as np
import pandas as pd
import glob
import os
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Normalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from dataclasses import dataclass
from feature_extraction import extract_features


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


def to_windows(signals, step, windows_size, windows_count):
    windows = []
    for i in range(windows_count):
        start = i * step
        end = start + windows_size
        window = signals[start:end]
        windows.append(window)

    return windows


# TODO: Flatten output one level; func returns nested list with every file as a sep. list
def process_multiple_files(data, step, config: ModelConfig):
    i = 0
    X, Y = [], []
    for file_data in data:
        x = process_data(
            signals=file_data['signals'],
            step=step,
            windows_count=file_data['windows_count'],
            config=config
        )
        y = process_labels(
            labels=file_data['labels'],
            step=step,
            windows_count=file_data['windows_count'],
            config=config
        )
        X.append(x)
        Y.append(y)

        print(f"Processed {i} files")
        i += 1
    # combined_X, combined_Y = combine_data(X, Y)
    return X, Y

# def combine_data(X, Y):
#     combined_X, combined_Y = [[]], []
#     for i in range(len(X)): # i: int
#         combined_X[i].extend(X[i])
#     return combined_X, combined_Y


# def process_multiple_files_old(data, step, config: ModelConfig):
#     processed_data = []
#     for file_data in data:
#         X = process_data(
#             signals=file_data['signals'],
#             step=step,
#             windows_count=file_data['windows_count'],
#             config=config
#         )
#         Y = process_labels(
#             labels=file_data['labels'],
#             step=step,
#             windows_count=file_data['windows_count'],
#             config=config
#         )
#         processed_data.append({'X': X, 'Y': Y})  # instead of append
#         print(f"Processed {len(processed_data)} files")
#     return processed_data


def process_data(signals, step: int, windows_count, config: ModelConfig):
    X = []
    for val in signals:
        windows = to_windows(signals=val, step=step, windows_size=config.window_size, windows_count=windows_count)

        x = []
        for window in windows:
            x.append(extract_features(window=window, features=config.features, wamp_threshold=config.wamp_threshold))  # POOOP!!
        X.append(x)
    return X


def process_labels(labels, step: int, windows_count, config: ModelConfig) -> list[int]:
    windowed_labels = to_windows(signals=labels, step=step, windows_size=config.window_size, windows_count=windows_count)

    Y = []
    for w in windowed_labels:
        Y.append(get_majority_label(w))
    return Y


def split_data(X, Y, train_ratio=0.7, val_ratio=0.2):
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=train_ratio, random_state=42)  # training and combined validation and test
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=val_ratio, random_state=42)  # split up validation and test

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)



def build_lstm_model(input_shape, num_classes):
    model = Sequential(
        [
            InputLayer(shape=input_shape, unroll=True),
            Normalization(),
            LSTM(32, unroll=True),
            Dense(6, activation='relu'),
            Dense(2, activation='softmax')
        ]
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def pre_process(input_dir: str, output_dir: str, output_file: str, config: ModelConfig):
    input_files = get_csv_file_list(input_dir)  # Get csv files globbed
    step = int(model_config.window_size * model_config.overlap)
    test_data = get_multiple_files_data(files=input_files, step=step, window_size=config.window_size)
    processed_test_data = process_multiple_files(data=test_data, step=step, config=config)

    joblib.dump(processed_test_data, output_file)
    print(f"Saved {output_file} for future use.")

    return processed_test_data


def reshape_data_vec(X, Y):
    """Vectorized version of the reshaping function"""

    all_windows = []
    all_labels = []

    for file_idx in range(len(X)):
        file_X = np.array(X[file_idx])  # Shape: (n_channels, n_windows, n_features)
        file_Y = np.array(Y[file_idx])  # Shape: (n_windows,)

        # Transpose to (n_windows, n_channels, n_features)
        file_X = file_X.transpose(1, 0, 2)

        # Reshape to (n_windows, n_channels * n_features)
        file_X_reshaped = file_X.reshape(file_X.shape[0], -1)

        all_windows.append(file_X_reshaped)
        all_labels.append(file_Y)

    # Concatenate all files
    X_final = np.vstack(all_windows)
    Y_final = np.hstack(all_labels)

    return X_final, Y_final


def train_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, num_classes):
    # Reshape each split
    x_train, y_train = reshape_data_vec(X_train, Y_train)
    x_val, y_val = reshape_data_vec(X_val, Y_val)
    x_test, y_test = reshape_data_vec(X_test, Y_test)

    print(f"Model input shape: {x_train.shape}")
    model = build_lstm_model((x_train.shape[0], x_train.shape[1],), num_classes)
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.001
    )
    
    history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
    return model, history


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

    input_dir = os.path.join('data', 'DB4_prepared')  # Folder to glob
    output_dir = "model"
    pkl_file = 'ninapro_DB4_4_emg.pkl'
    output_file = os.path.join(output_dir, pkl_file)

    if os.path.exists(output_file):
        print("Loading data from file")
        X, Y = joblib.load(output_file)
    else:
        print("Pre-processing from csv-files")
        X, Y = pre_process(input_dir, output_dir, output_file, model_config)

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(X, Y)

    train_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, len(np.unique_values(Y)))
