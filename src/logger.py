import numpy, time


class Logger:
    def __init__(self, path='output/output.csv', state_header: list[int] = [0, 1, 2, 3]) -> None:
    #* saves path, opens the file in write mode, creates an empty list as buffer and WHAT?!
        self.path = path
        self.file = open(self.path, 'w')
        self.buffer = []
        self.file.write(f"Current time, Inference time, Input data, Chosen output state, {', '.join(map(str, state_header))}\n")
        self.start_time = time.time()


    def __del__(self) -> None:
    #* Closes and sends a message to the user
        self.file.close()
        print(f"Logger closed at {self.path}")



    def log(self, input_data: list[float], output_data: numpy.ndarray, inference_time) -> None:
        current_time = time.time() - self.start_time
        output_state = output_data.tolist().index(max(output_data))  # shadowing
        input_data_str = ', '.join(map(str, input_data)).replace('[', '').replace(']', '').replace('.,', ',')
        output_data_str = ', '.join(map(str, output_data))

        output_str = f"{current_time}, {inference_time}, {input_data_str}, {output_state}, {output_data_str}\n"

        self.file.write(output_str)
        self.file.flush()
