import toml
from dataclasses import dataclass

@dataclass
class Config:
#* Similar to a struct in C. Specifies the type for each variable
    window_size: int
    window_overlap: float
    sampling_freq: int
    normalization: str
    fs: int
    lowcut: int
    highcut: int
    filter_order: int
    wamp_threshold: float
    features: list[str]
    model_path: str
    log_path: str
    model_states: list[int]
    input_file_path: str

def get_config(file_path: str) -> Config:
#* Open the file in read mode and then unpacking contents using toml.
    with open(file_path, 'r') as f:
        config_data = toml.load(f)
        return Config(**config_data) # Unpack contents of toml

def save_config(model_config: Config, config_path: str) -> None:
    config = toml.dumps(model_config.__dict__)

    with open(config_path, 'w') as f:
        f.write(config)
        
        

if __name__ == "__main__":
#* Main function
    config = get_config("config.toml")
    print(config)
