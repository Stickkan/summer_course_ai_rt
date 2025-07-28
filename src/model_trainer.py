import numpy as np
import glob, os, joblib, toml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Normalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt
from config import Config, save_config
import data_shaper



def build_lstm_model(input_shape, num_classes):
    model = Sequential(
        [
            InputLayer(shape=input_shape, unroll=True), #? Maybe set the input layer activation function to 'tanh' or 'sigmoid'?
            # Normalization(),
            LSTM(32, unroll=True),
            Dense(6, activation='relu'),
            Dense(num_classes, activation='softmax') #* Different activation functions for each layer
        ]
    )
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, num_classes):
    #* Reshape each split
    x_train, y_train = data_shaper.reshape_data_vec(X_train, Y_train)
    x_val, y_val = data_shaper.reshape_data_vec(X_val, Y_val)
    x_test, y_test = data_shaper.reshape_data_vec(X_test, Y_test)

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


def convert_keras_to_tflite(keras_model, output_path):
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    return tflite_model
       
        
def save_tflite_model(model, output_path):
    tflite_model = convert_keras_to_tflite(model, output_path)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)


def plot_training_history(history, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_path)
    plt.close()


if __name__ == '__main__':

    f_name = 'DB4_prepared_4_states'

    input_dir = os.path.join('data', f_name)  #* Folder to glob
    output_dir = os.path.join("model", f_name)

    pkl_path = os.path.join(output_dir, f_name  + '.pkl')
    model_path = os.path.join(output_dir, f_name + '.keras')
    tflite_model_path = os.path.join(output_dir, f_name + '.tflite')
    config_path = os.path.join(output_dir, f_name  + '.toml')
    log_path = os.path.join(output_dir, 'output', 'inference.csv')  # Path for log-file during inference
    training_history_path = os.path.join(output_dir, f_name + '_training_history.pkl')

    #*  Some values for testing
    model_config = Config(
        window_size=400, #* Correspond to 200 ms since the sampling frequency for Ninapro DB4 is 2kHz
        window_overlap=0.5, #* Common degree of overlap. Less overlap can result in decreased accuracy but shorter computation duration.
        sampling_freq=2000,  # Sampling freq of recorded data, for filtering
        normalization="None",  # Normalization to be applied, TODO: Might want to include a band-stop filter for the 50 hz (+- 2 hz).
        fs=0,
        lowcut=20, #? This is too high cutoff frequency for a highpass filter. Recommend to lower it to around 5.
        highcut=450, #? This is also to high I would say.
        filter_order=4,
        wamp_threshold=0.02, #* Mess around with this. A lower value makes the feature more susceptible to noise.
        features=['mav', 'wl', 'wamp', 'mavs'],
        model_path=tflite_model_path,
        log_path=log_path,
        model_states=[],
        input_file_path=f"{os.path.join('data', f_name, 'test_data.pkl')}"
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(pkl_path):
        print("Loading data from file")
        X, Y = joblib.load(pkl_path)
    else:
        print("Pre-processing from csv-files")
        X, Y = data_shaper.pre_process(input_dir, pkl_path, model_config)

    # Reshape data
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = data_shaper.split_data(X, Y)
    model_config.model_states = np.unique_values(Y).tolist()
    print(model_config.model_states)

    # Train model
    model, history = train_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, len(model_config.model_states))

    # Save everything
    model.save(model_path)
    joblib.dump(history.history, training_history_path)
    save_tflite_model(model, tflite_model_path)
    save_config(model_config, config_path)

    # save X_test and Y_test
    joblib.dump((X_test, Y_test), os.path.join('data/DB4_prepared_4_states/', "test_data.pkl"))

    # save plot of training history
    plot_training_history(history.history, os.path.join(output_dir, 'training_history.png'))
