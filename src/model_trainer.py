from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from pre_processor import PreProcessor


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
    model = build_lstm_model((10, 5))
    model.summary()