import os, time
from config import get_config, Config
from pre_processor import PreProcessor, NoPreProcessor, get_pre_processor
from data_input import FileInput, SensorInput
from logger import Logger
from model_loader import Model
from data_input import get_input_handle
from model_trainer import Config



def get_data_input(config: Config, type='file'):
    if type == 'file':
        return FileInput(config)
    elif type == 'sensor':
        return SensorInput(config)
    else:
        raise Exception("Wrong input type")


if __name__ == '__main__':

    model_name = 'LSTM_DB4_prepared_4_states'

    config = get_config(file_path=os.path.join('model', model_name, f"{model_name}.toml"))
    data_input = get_data_input(config=config, type='file')
    logger = Logger(path=config.log_path, state_header=config.model_states)
    pre_process = get_pre_processor(config=config, data_source=data_input, log_fn=None)  # logger.log_input_data
    model = Model(model_path=config.model_path, logger=None)  # logger.log_output_data

    # start_time = time.time()

    while True:
        window = pre_process.get_next()

        if window is None:
            break

        inference_start_time = time.perf_counter()
        output_state = model.get_output_state(window)
        inference_end_time = time.perf_counter() - inference_start_time

        logger.log(window[0], output_state, inference_end_time)  # Saves input data
