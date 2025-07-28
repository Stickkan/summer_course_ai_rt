# Architecture

## main
- Instantiates all objects
- Runs while-loop
  * Instructs pre-processor to get new window
  * Runs inference on window
  * Sends in/out to logger

## model_trainer:
- Loads in data from .csv-files
- Uses `data_shaper` for reshaping input data
- Trains model
- Saves model and .toml-file used for configuration
  * The toml-file is considered the single source of truth
- Uses `ModelConfig` for model/training parameters

 ## pre_processor
- Receives raw data input (e.g., from sensors or files)
- Applies necessary transformations (e.g., normalization, filtering)
- Extracts a window of data for inference
- Passes processed window to inference engine

## model_loader
- Loads trained model and configuration from .toml-file
- Accepts processed data window from pre_processor
- Runs inference using the model
- Returns predictions/results to main loop

## config
- Reads and writes configuration from/to .toml-file
- Provides configuration parameters to other modules
- Ensures consistency and single source of truth for settings

## model_config
- Contains the `ModelConfig` dataclass specifying model/training parameters
- Used by both training and inference modules for consistent configuration

## utils - logger, feature_extraction, data_shaper
- `logger`: Handles logging of input/output data and states
- `feature_extraction`: Contains feature computation functions for EMG data
- `data_shaper`: Dedicated module for shaping and preparing input data for training/inference
- Used across multiple modules to avoid code duplication
