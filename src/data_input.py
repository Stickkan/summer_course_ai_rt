import threading
import queue, joblib
import numpy as np
from config import Config

# Classes to handle data input from either a file or an ADC, interface
# is identical so as to be interchangeable

class SensorInput:
    #* is_done(), has_next() and next() returns a bool. They have not been implemented yet.

    def __init__(self, config) -> None:
        pass


    def is_done(self) -> bool:
        return False


    def has_next(self) -> bool:
        return True


    def next(self) -> np.ndarray | None:
        pass


class FileInput:

    def __init__(self, config) -> None:
    #* In __init__ Loads the data from the Ninapro DB4 into the self.data using the numpy library
        if config is None:
            raise ValueError('No config provided')

        self.X, self.Y = joblib.load(config.input_file_path)
        self.data = np.array(self.X[0], dtype=np.float32)
        self.it = 0
        self.data_len = len(self.data[0])
        self.window_size = config.window_size


    def is_done(self) -> bool:
        return not self.has_next()


    def has_next(self) -> bool:
        return self.it < self.data_len


    def next(self) -> np.ndarray | None:
        window = []
        try:
            for ft in self.data:
                window.extend(ft[self.it])
            self.it += 1
            return np.array(window, dtype=np.float32).reshape((1, len(window), 1))
        except Exception:
            return None



def get_input_handle(type: str, config: Config | None) -> SensorInput | FileInput:
    #* The -> operator specifies which return type SHOULD be returned.
    #* Depending on the input ('sensor' or 'file') a SensorInput class or FileInput class is returned.
    if type == 'sensor':
        return SensorInput(config)
    elif type == 'file':
        return FileInput(config)
    else:
        raise ValueError(f'Invalid input type: {type}')
