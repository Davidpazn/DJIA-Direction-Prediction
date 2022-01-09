# imports
import numpy as np
import pandas as pd

# sklearn  imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

# tensorflow imports
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Reshape, BatchNormalization

from tensorflow import keras



# Univariate Time Series Regression Utils
def create_univariate_rnn_data(series, window):
    n = series.shape[0]
    y = series[window: ]
    data = series.values.reshape(-1, 1)  # 2D format of series
    X = np.hstack(tuple([data[i: n-j, :] for i, j in enumerate(range(window, 0, -1))]))
    return pd.DataFrame(X, index=y.index), y


# blocking split time series
# https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
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


# sliding window split
# https://ntguardian.wordpress.com/2017/06/19/walk-forward-analysis-demonstration-backtrader/
class SlidingWindowSplit(TimeSeriesSplit):
 
    def split(self, X, y=None, groups=None, fixed_length=False,
              train_splits=1, test_splits=1):

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        train_splits, test_splits = int(train_splits), int(test_splits)
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        if (n_folds - train_splits - test_splits) <= 0 and test_splits > 0:
            raise ValueError(
                ("Both train_splits and test_splits must be positive"
                 " integers."))
            
        indices = np.arange(n_samples)
        split_size = (n_samples // n_folds)
        test_size = split_size * test_splits
        train_size = split_size * train_splits
        test_starts = range(train_size + n_samples % n_folds,
                            n_samples - (test_size - split_size),
                            split_size)
        if fixed_length:
            for i, test_start in zip(range(len(test_starts)),
                                     test_starts):
                rem = 0
                if i == 0:
                    rem = n_samples % n_folds
                yield (indices[(test_start - train_size - rem):test_start],
                       indices[test_start:test_start + test_size])
        else:
            for test_start in test_starts:
                yield (indices[:test_start],
                    indices[test_start:test_start + test_size])      

class stacked_LSTM():
    def __init__(self, df, inputs=[], window=60):
        self.data = df
        self.inputs = inputs # 
        self.window = window
    
    def build_lstm_dataset(self, train_size=0.8):
        train_indices = list(range(int(len(self.data)*0.8)))
        test_indices = list(range(train_indices[-1]+1, len(self.data)))
        sequence = list(range(1, self.window+1))

        train_data, test_data = self.data.iloc[train_indices], self.data.iloc[test_indices]
        # Reshape training for neural net
        X_train = [
                # get first window returns
                train_data.loc[:, sequence].values.reshape(-1, self.window, 1)  
        ]
        y_train = train_data.Label
        
        # Reshape testing for neural net
        X_test = [
                test_data.loc[:, sequence].values.reshape(-1, self.window, 1)
        ]
        y_test = test_data.Label

        for inp in self.inputs:  # topic, sent, ind
            X_train.append(train_data.filter(like=inp))
            X_test.append(test_data.filter(like=inp))
        
        return (X_train, X_test, y_train, y_test)


    def build_stacked_model(self, 
            num_lstm_layers, 
            lstm_units=[25, 10],
            lstm_dropouts=[0.2, 0.2],
            dense_units=10,
            n_features=1, 
            input_names=['topics', 'indicators', 'sentiment'],
            learning_rate=0.001,
            epsilon=1e-08):

        model = Sequential()
        model.add(Input(shape=(self.window, n_features)))
        
        for ix, inp_shape in enumerate(self.inputs):
            model.add(Input(shape=(inp_shape, ), name=input_names[ix]))
        
        # Add LSTM Units:
        return_sequences = False
        for ix, (units, dropout) in enumerate(zip(lstm_units, lstm_dropouts)):
            if ix < len(lstm_units -1):
                return_sequences = True
            model.add(
                LSTM(units=units,
                     input_shape=(self.window, n_features),
                     name='LSTM' + str(ix),
                     dropout=dropout,
                     return_sequences=return_sequences)
                     )
        
        model.add(BatchNormalization)
        model.add(Dense(units=dense_units, name='Dense'))

        model.add(Dense(1, name='Output', activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         epsilon=epsilon)
        model.compile(loss='binary_cross_entropy',
                      optimizer=optimizer,
                      metrics=['accuracy',
                               tf.keras.metrics.AUC(name='AUC')
                      ]
        )
        return model


