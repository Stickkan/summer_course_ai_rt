import os, time
from config import get_config, Config
from pre_processor import PreProcessor
from data_input import FileInput, SensorInput
from logger import Logger
from model_loader import Model
from data_input import get_input_handle
from src.model_trainer import ModelConfig



def get_data_input(config: Config, type='file'):
    if type == 'file':
        return FileInput(config)
    elif type == 'sensor':
        return SensorInput(config)
    else:
        raise Exception("Wrong input type")


if __name__ == '__main__':
    model_name = 'DB4_prepared_4_states'
    config = get_config(file_path=os.path.join('model', model_name))  # TODO: update so model_trainer outputs the same schema
    logger = Logger(path=config.log_path, state_header=config.model_states)
    data_input = get_data_input(config=config, type='file')
    pre_process = PreProcessor(config=config, data_source=data_input, log_fn=logger.log_input_data)
    model = Model(model_path=config.model_path, logger=logger.log_output_data)
    
    start_time = time.time()
    
    while True:
        window = pre_process.get_next()
        
        if window is None:
            break
            
        output_state = model.get_output_state(window)
        logger.log_input_data(output_state)  # Saves input data
