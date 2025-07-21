import numpy as np

def compute_mav(window):
    #* mav = Mean Absolute Value
    return np.mean(np.abs(window))

def compute_wl(window):
    #* wl = waveform length measures the cumulative length over the window. Gives an indication of waveform amplitude and frequency changes
    return np.sum(np.abs(np.diff(window)))

def compute_wamp(window, threshold=0.02):
    #* wamp = Wilson Amplitude sums the number of times the absolute difference in two following samples crosses the threshold. 
    #* Greater wamp -> greater muscle activation
    return np.sum(np.abs(np.diff(window)) > threshold)

def compute_mavs(window):
    #* mavs = Mean Absolute Value Slope calculates the difference of mav in consecutive windows.
    half = len(window) // 2
    return np.abs(compute_mav(window[:half]) - compute_mav(window[half:]))
    

def extract_features(window, features = ['mav', 'wl', 'wamp', 'mavs'], wamp_threshold=0.02):
    extracted_features = []
    for feature in features:
        if feature == 'mav':
            extracted_features.append(compute_mav(window))
        elif feature == 'wl':
            extracted_features.append(compute_wl(window))
        elif feature == 'wamp':
            extracted_features.append(compute_wamp(window, threshold=wamp_threshold))
        elif feature == 'mavs':
            extracted_features.append(compute_mavs(window))            
    return extracted_features