import numpy as np

class BaseRegression:
    def __init__(self,learning_rate=0.01,epoch=1000,tol=1e-6):
        self.learning_rate=learning_rate
        self.epoch=epoch
        self.tol=tol
        self.theta=None
        self.loss_history=[]

    def _add_intercept(self,X):
        intercept=np.ones((X.shape[0],1))
        return np.concatenate((intercept,X),axis=1)
    def _compute_loss(self, X_padded, y):
        residuals = (X_padded @ self.theta) - y
        return float(np.mean(residuals ** 2))
    def fit(self,X,y):
        raise NotImplementedError("Subclasses must implement fit().")
    def predict():
        if self.theta is None:
            raise ValueError("Model must be fitted before calling predict().")
        X_padded = self._add_intercept(X)
        return X_padded @ self.theta