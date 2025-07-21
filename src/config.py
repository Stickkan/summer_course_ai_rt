import toml
from enum import Enum
from dataclasses import dataclass

@dataclass
class Config:
#* Similar to a struct in C. Specifies the type for each variable
    emg_file_path: str
    window_size: int
    window_overlap: float
    sampling_freq: int
    normalization: str
    pre_proc_buffer_len: int
    features: list[str]

def get_config(file_path: str) -> Config:
#* Open the file in read mode and then unpacking contents using toml.
    with open(file_path, 'r') as f:
        config_data = toml.load(f)
        return Config(**config_data) # Unpack contents of toml



if __name__ == "__main__":
#* Main function
    config = get_config("config.toml")
    print(config)
