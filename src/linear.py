import numpy as np
from src.base import BaseRegression
class LinearRegressionOLS(BaseRegression):
    def fit(self,X,y):
        X_padded = self._add_intercept(X)
        self.theta = np.linalg.pinv(X_padded) @ y
        self.loss_history = [self._compute_loss(X_padded, y)]
        return self
        