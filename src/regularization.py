import numpy as np
from .base import BaseRegression

class RidgeRegression(BaseRegression):
    def __init__(self,alpha=1.0,**kwargs):
        super().__init__(**kwargs)
        self.alpha=alpha
    def fit(self,X,y):
        X_padded=self._add_intercept(X)
        n_features_padded = X_padded.shape[1]

        reg_matrix=np.eye(n_features_padded)
        reg_matrix[0,0]=0

        A=X_padded.T @ X_padded + self.alpha * reg_matrix
        B=X_padded.T @ y

        self.theta=np.linalg.pinv(A)@B

        self.loss_history=self._compute_loss(X_padded,y)

        return self


class LassoRegression(BaseRegression):
    def __init__(self,alpha=1.0,max_iter=1000,tol=1e-4,**kwargs):
        super().__init__(**kwargs)
        self.alpha=alpha
        self.max_iter=max_iter
        self.tol=tol
    @staticmethod
    def _soft_threshold(rho,alpha):
        if rho <-alpha:
            return rho+alpha
        elif rho>alpha:
            return rho-alpha
        else:
            return 0.0
    def fit(self,X,y):
        X_padded=self._add_intercept(X)
        n_samples,n_features_padded=X_padded.shape

        self.theta=np.zeros(n_features_padded)
        self.loss_history=[]

        col_norms_eq=np.sum(X_padded**2,axis=0)

        for iterations in range(self.max_iter):
            theta_old=self.theta.copy()

            for j in range(n_features_padded):

                residuals=y-X_padded@self.theta+X_padded[:,j]*self.theta[j]
                rho_j=X_padded[:,j]@residuals

                if j==0:
                    self.theta[j]=rho_j/col_norms_eq[j]
                else:
                    self.theta[j]=self._soft_threshold(rho_j,self.alpha*n_samples)/col_norms_eq[j]
            loss=self._compute_loss(X_padded,y)
            self.loss_history.append(loss)

            if np.max(np.abs(self.theta - theta_old)) < self.tol:
                break
        return self
