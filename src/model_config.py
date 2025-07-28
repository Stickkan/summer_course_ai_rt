from dataclasses import dataclass

@dataclass
class ModelConfig:
#* Similar to a struct in C. Specifies type for each variable.
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