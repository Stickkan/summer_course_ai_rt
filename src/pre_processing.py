import numpy as np

# Kopierade in några funktioner bara, inte säker på om kommer använda


#Create the windows
def sliding_window(data, window_size, overlap):
    """
    Applies a sliding window with overlap to the input data.

    Parameters:
    data (ndarray): input data
    window_size (int): size of sliding window
    overlap (int): overlap between consecutive windows

    Returns:
    ndarray: 2D array with sliding windows of shape (n_windows, window_size)
    """
    n_samples = data.shape[0]
    overlap = int(np.floor((window_size*overlap)/100))
    n_windows = int(np.floor((n_samples - window_size) / overlap) + 1)
    windows = np.zeros((n_windows, window_size))

    for i in range(n_windows):
        start = i * overlap
        end = start + window_size
        windows[i] = data[start:end]

    return windows


def feature_windows(window, features):
    '''
    a function that measures features from each slides
    and creates a 1D array of measurements from each slide.
    '''

    var    = lambda data : np.var(data,axis =1)
    rms    = lambda data : np.sqrt(np.mean(data ** 2,axis =1))
    mav    = lambda data : np.sum(np.absolute(data),axis =1) / len(data)
    wl     = lambda data : np.sum(abs(np.diff(data)),axis =1)
    mean   = lambda data : np.mean(data,axis =1 )
    std    = lambda data : np.std(data,axis =1)
    median = lambda data : np.median(data,axis =1)
    peak   = lambda data : np.max(data,axis =1)
    min    = lambda data : np.min(data,axis =1)
    iemg   = lambda data : np.sum(abs(data),axis =1)
    aac    = lambda data : np.sum(abs(np.diff(data)),axis =1) / len(data)
    kur    = lambda data : kurtosis(data,axis =1)
    skewe  = lambda data : skew(data,axis =1)

    # features = [var,rms,mav,wl,mean,std,median,peak,min,iemg,aac,kur,skewe]
    # win_matrix = sliding_window(data, window_size, overlap)

    feature_vector = []
    for feature_func in features:
        feature_vector.extend(feature_func(win_matrix))

    return np.array(feature_vector)
