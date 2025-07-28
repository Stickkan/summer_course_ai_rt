# Architecture

## main
- Instantiates all objects
- Runs while-loop
  * Instructs pre-processor to get new window
  * Runs inference on window
  * Sends in/out to logger

## model_trainer:
- Loads in data from .csv-files
- Reshapes data
- Trains model
- Saves model and .toml-file used for configuration
  * The toml-file is considered the single source of truth!

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

## utils - logger, feature_extraction
- Contains helper functions for data manipulation, logging, and error handling
- Used across multiple modules to avoid code duplication
