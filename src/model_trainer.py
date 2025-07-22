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
import toml



@dataclass
class ModelConfig:
#* Similar to a struct in C. Specifies type for each variable.
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
    #* df = data file and contains the csv file created earlier.
    df = pd.read_csv(file_path, sep=' ', header=0) 

    signals = []
    for i in df.columns[0:-1]: #* Iterates through every column up until (not including) the last column
        #* Appends each column to the signal list
        signals.append(df[i].to_numpy())

    labels = df['label'].to_numpy()

    return (signals, labels, len(labels))


def get_multiple_files_data(files: list[str], step: int, window_size: int):
    data = []
    for f in files:
        #* Returns values for each variable from get_file_data(f).
        signals, labels, data_len = get_file_data(f)
        windows_count = (data_len - window_size) // step + 1
        #* Append dictionary within the data list where the 'key' is saved together with a key i.e. 'signals'(key):signals(value)
        data.append({'signals': signals, 'labels': labels, 'data_len': data_len, 'windows_count': windows_count})

    return data


def get_majority_label(window):
    #* Returns the most common label. 
    #* Usually .most_common() returns a list with a tuple with the most common element as well as the count.
    #* But with the addition of [0][0] it only returns the first element without count. The type is set automatically (as usual for python).
    return Counter(window).most_common(1)[0][0]


def to_windows(signals, step, windows_size, windows_count):
    #* For each element in the windows list is an embedded list (a list within a list). 
    #* Each list is the window_size and the following element has 50% of the previous values. The overlap is set in the model_config class.
    windows = []
    for i in range(windows_count):
        #* step = int(model_config.window_size * model_config.overlap) where overlap is set to 0.5 i.e. 50%
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

def process_data(signals, step: int, windows_count, config: ModelConfig):
    #* The upper case 'X' is the list and the lower case 'x' (which is also a list) is the element of upper case 'X'
    X = []
    for val in signals:
        windows = to_windows(signals=val, step=step, windows_size=config.window_size, windows_count=windows_count)
        x = []
        for window in windows:
            x.append(extract_features(window=window, features=config.features, wamp_threshold=config.wamp_threshold)) 
        X.append(x)
    return X


def process_labels(labels, step: int, windows_count, config: ModelConfig) -> list[int]:
    windowed_labels = to_windows(signals=labels, step=step, windows_size=config.window_size, windows_count=windows_count)
    #? is the name of the variable 'windowed_labels' suitable. to_windows() returns a list within a list with the overlapped data.
    Y = []
    for w in windowed_labels:
        Y.append(get_majority_label(w))
    return Y


def split_data(X, Y, train_ratio=0.7, val_ratio=0.2): #? Why not use the entire dataset? 0.7 + 0.2 = 0.9
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=train_ratio, random_state=42)  #* Training and combined validation and test
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=val_ratio, random_state=42)  #* Split up validation and test

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def build_lstm_model(input_shape, num_classes): #? the argument num_classes is not used. Is it there as a placeholder for future use?
    model = Sequential(
        [
            InputLayer(shape=input_shape, unroll=True), #? Maybe set the input layer activation function to 'tanh' or 'sigmoid'?
            Normalization(),
            LSTM(32, unroll=True),
            Dense(6, activation='relu'),
            Dense(6, activation='softmax') #* Different activation functions for each layer 
        ]
    )
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def pre_process(input_dir: str, output_dir: str, output_file: str, config: ModelConfig): #? the argument output_dir is not used. Is it there as a placeholder for future use?
    input_files = get_csv_file_list(input_dir)  #* Get csv files globbed (globbed = popular library)
    step = int(model_config.window_size * model_config.overlap)
    test_data = get_multiple_files_data(files=input_files, step=step, window_size=config.window_size)
    processed_test_data = process_multiple_files(data=test_data, step=step, config=config)

    #* joblib is a popular library which is used to save python objects and is optimized for large data.
    joblib.dump(processed_test_data, output_file)
    print(f"Saved {output_file} for future use.")

    return processed_test_data


def reshape_data_vec(X, Y):
    #* Vectorized version of the reshaping function

    all_windows = []
    all_labels = []

    for file_idx in range(len(X)):
        file_X = np.array(X[file_idx])  #* Shape: (n_channels, n_windows, n_features)
        file_Y = np.array(Y[file_idx])  #* Shape: (n_windows)

        #* Transpose to (n_windows, n_channels, n_features)
        file_X = file_X.transpose(1, 0, 2)

        #* Reshape to (n_windows, n_channels * n_features)
        file_X_reshaped = file_X.reshape(file_X.shape[0], -1)

        all_windows.append(file_X_reshaped)
        all_labels.append(file_Y)

    #* Concatenate all files
    X_final = np.vstack(all_windows)
    Y_final = np.hstack(all_labels)

    return X_final, Y_final


def train_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, num_classes):
    #* Reshape each split
    x_train, y_train = reshape_data_vec(X_train, Y_train)
    x_val, y_val = reshape_data_vec(X_val, Y_val)
    x_test, y_test = reshape_data_vec(X_test, Y_test)

    print(f"Model input shape: {x_train.shape}")
    print(f"Model output shape: {y_train.shape}")
    model = build_lstm_model((x_train.shape[1],1,), num_classes)
    model.summary()

    #* Callbacks
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

    #* Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    return model, history

def save_config(model_config: ModelConfig, model_path: str, config_path: str, output_states: list[int]) -> None:
    config = toml.dumps(model_config.__dict__)
    config += f"model_path = '{model_path}'"
    
    with open(config_path, 'w') as f:
        f.write(config)


if __name__ == '__main__':
    #*  Some values for testing
    model_config = ModelConfig(
        window_size=400,
        overlap=0.5,
        fs=0,
        lowcut=20,
        highcut=450,
        filter_order=4,
        wamp_threshold=0.02,
        features=['mav', 'wl', 'wamp', 'mavs']
    )

    f_name = 'ninapro_DB4_4_emg'
    input_dir = os.path.join('data', 'DB4_prepared')  # Folder to glob
    output_dir = os.path.join("model", f_name)
    pkl_path = os.path.join(output_dir, f_name  + '.pkl')
    model_path = os.path.join(output_dir, f_name + '.keras')
    training_history_path = os.path.join(output_dir, f_name + '_training_history.pkl')
    config_path = os.path.join(output_dir, f_name  + '.toml')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(pkl_path):
        print("Loading data from file")
        X, Y = joblib.load(pkl_path)
    else:
        print("Pre-processing from csv-files")
        X, Y = pre_process(input_dir, output_dir, pkl_path, model_config)

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(X, Y)
    output_states = np.unique_values(Y).tolist()
    model, history = train_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, len(np.unique_values(Y)))
    model.save(model_path)
    joblib.dump(history.history, training_history_path)
    save_config(model_config, model_path, config_path, output_states)
