# imports
import numpy as np
import pandas as pd


# Univariate Time Series Regression Utils
def create_univariate_rnn_data(series, window):
    n = series.shape[0]
    y = series[window: ]
    data = series.values.reshape(-1, 1)  # 2D format of series
    X = np.hstack(tuple([data[i: n-j, :] for i, j in enumerate(range(window, 0, -1))]))
    return pd.DataFrame(X, index=y.index), y


# blocking split time series
class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, gropus):
        return self.n_splits
    
    def split(self, X, y=None, groups=None, margin=0):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]
            