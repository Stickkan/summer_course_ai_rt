import toml
from enum import Enum
from dataclasses import dataclass


@dataclass
class Config:
    emg_file_path: str
    window_size: int
    window_overlap: float
    sampling_freq: int
    normalization: str
    pre_proc_buffer_len: int
    features: list[str]


def get_config(file_path: str) -> Config:
    with open(file_path, 'r') as f:
        config_data = toml.load(f)
        return Config(**config_data) # Unpack contents of toml



if __name__ == "__main__":
    config = get_config("config.toml")
    print(config)
