import numpy as np
from collections import deque
from numpy.typing import NDArray
from src.config import Config
from src.data_input import FileInput, SensorInput
from src.logger import Logger


class PreProcessor:
    """
    Preprocessor class for processing EMG data.
    This class provides methods for preprocessing EMG data, including windowing and normalization.
    """

    def __init__(self, data_source: FileInput | SensorInput, config: Config, log_fn) -> None:
        self.data_source = data_source
        self.config = config
        self.buffer = deque(maxlen=config.pre_proc_buffer_len) # Buffer's depth, 2 is fine
        self.finalized_data = deque(maxlen=config.windows_count)
        self.step = round(config.window_size * config.window_overlap)
        self.index = 0
        self.log_fn = log_fn


    def get_next(self) -> np.ndarray | None:
        while len(self.finalized_data) < self.config.windows_count:
            window = self._get_next_window()
            if window is None:
                return None
            self.finalized_data.append(window)

        f_data = np.array(self.finalized_data, dtype=np.float32)
        self.finalized_data.popleft()  # Reset for next batch
        
        return np.array([f_data.flatten()])


    def _get_next_window(self):
        # Fill buffer first
        while len(self.buffer) < self.config.pre_proc_buffer_len:
            if self.data_source.has_next():  # non-blocking, busy-wait
                next_window = self.data_source.next()

                if self.log_fn is not None:
                    self.log_fn(next_window)  # Save input data to log
                self.buffer.append(next_window)  # TODO: Decide where to filter, here or in process_window?

            if self.data_source.is_done(): # If no more input then stop
                return None

        window = np.array(self.buffer, dtype=np.float32).flatten()[self.index:self.index+self.config.window_size]

        self.index += self.step
        if self.index + self.config.window_size >= self.config.window_size * self.config.pre_proc_buffer_len:
            self.index -= self.config.window_size
            if len(self.buffer) == self.buffer.maxlen:
                self.buffer.popleft()

        processed_window = self._process_window(window)
        return processed_window


    def _process_window(self, window: np.ndarray) -> np.ndarray | None:
        if (len(self.config.features) == 0):  # No feature extraction
            return window

        window = np.array(window)
        # window = np.array(self._bandpass_filter(window))

        #normalize the window - TODO: THIS!
        if self.config.normalization == "MinMax":
            window = (window - np.min(window)) / (np.max(window) - np.min(window))
            exit(-1) # TODO FIX THIS
        # elif self.config.normalization == "MeanStd":
        #     window = (window - self.config.window_normalization['global_mean']) / self.config.window_normalization['global_std']

        features = extract_features(
            window=window,
            features=self.config.features,
            wamp_threshold=self.config.wamp_threshold
        )

        normalized_features = []

        for feature_name, feature in zip(self.config.features, features):
            [mean, std] = self.config.feature_stats[feature_name]
            normalized_feature = (feature - mean) / std
            normalized_features.append(normalized_feature)

        return np.array(normalized_features, dtype=np.float32)
