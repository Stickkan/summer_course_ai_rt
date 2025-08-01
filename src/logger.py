import numpy


class Logger:
    def __init__(self, path='output/output.csv', state_header: list[int] = [0, 1, 2, 3]) -> None:
    #* saves path, opens the file in write mode, creates an empty list as buffer and WHAT?!
        self.path = path
        self.file = open(self.path, 'w')
        self.buffer = []
        self.file.write(f"time, input_data, output_state, {', '.join(map(str, state_header))}\n")


    def __del__(self) -> None:
    #* Closes and sends a message to the user
        self.file.close()
        print(f"Logger closed at {self.path}")


    def log_input_data(self, input_data) -> None:
    #* Buffer input data because output from model is gathered later
        self.buffer.extend(input_data)


    #* Input data is a 1D numpy array of raw input data
    #* Output data is a 1D numpy array of percentage values
    def log_output_data(self, output_data, time) -> None:
        input_data = self.buffer
        self.buffer = []
        output_data = output_data.tolist()

        output_state = [output_data.index(max(output_data))] * len(input_data)
        output_data = [output_data] * len(input_data)
        _time = [time] * len(input_data)

        for t, i, s, o in zip(_time, input_data, output_state, output_data):
            self.file.write(f"{t}, {i}, {s}, {','.join(map(str, o))}\n")
        self.file.flush()
