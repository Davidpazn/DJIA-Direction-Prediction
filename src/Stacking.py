# # imports
import numpy as np
from xgboost import XGBClassifier


class StackedEnsemble:
    def __init__(self, models, X_train, y_train):
        self.models = models
        self.X = X_train
        self.y = y_train

    def _stacked_dataset(self, X_test=None):
        if X_test is None:
            X = self.X
        else:
            X = X_test

        stackX = None
        for model in self.models:
            yhat = model.predict(X, verbose=0)
            if stackX is None:
                stackX = yhat
            else:
                stackX = np.dstack((stackX, yhat))
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
        return stackX


    def predict(self, X_test):
        stackedX = self._stacked_dataset()
        # fit model based on outputs of ensemble models
        model = XGBClassifier(eval_metric='logloss')
        model.fit(stackedX, self.y)
        
        stackedX = self._stacked_dataset(X_test)
        yhat = model.predict(stackedX)

        return yhat






