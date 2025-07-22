import threading
import queue
import numpy as np
from config import Config

# Classes to handle data input from either a file or an ADC, interface
# is identical so as to be interchangeable

class SensorInput:

    def __init__(self) -> None:
        pass


    def is_done(self):
        return False


    def has_next(self) -> bool:
        return True


    def next(self) -> np.ndarray | None:
        pass


class FileInput:
    #* is_done(), has_next() and next() returns a bool. They have not been implemented yet.

    def __init__(self, config) -> None:
    #* In __init__ Loads the data from the Ninapro DB4 into the self.data using the numpy library
        if config is None:
            raise ValueError('No config provided')

        self.data = np.loadtxt(config.emg_file_path, delimiter=",", dtype=np.float32)[:,1]
        self.window_size = config.window_size


    def _pop_front(self, array, n=200):
    #* _pop_front() the first n elements from the array and return them as a new array.
    #* Borrowed from Ludwig Bogsveen
        if len(array) < n:
            raise ValueError("Not enough elements to pop.")

        front = array[:n]
        array = array[n:]  # new view
        return front, np.array(array, dtype=np.float32)


    def is_done(self):
        return self.has_next


    def has_next(self) -> bool:
        return self.data.shape[0] > 0


    def next(self) -> np.ndarray | None:
        try:
            window, self.data = self._pop_front(self.data, n=self.window_size)
            return window
        except Exception:
            return None



def get_input_handle(type: str, config: Config | None) -> SensorInput | FileInput:
    #* The -> operator specifies which return type SHOULD be returned.
    #* Depending on the input ('sensor' or 'file') a SensorInput class or FileInput class is returned.
    if type == 'sensor':
        return SensorInput()
    elif type == 'file':
        return FileInput(config)
    else:
        raise ValueError(f'Invalid input type: {type}')
