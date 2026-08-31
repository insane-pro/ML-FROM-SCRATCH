import numpy as np
import pytest
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from src.linear import LinearRegressionOLS
from src.regularization import RidgeRegression,LassoRegression


@pytest.fixture
def mock_data():
    rng=np.random.default_rng(42)
    X=rng.normal(0,1,size=(100,3))
    true_weights=np.array([1.5,-2.0,3.0])
    y=X@true_weights+rng.normal(0,0.1,size=100)
    return X,y

def test_ols_closed_form(mock_data):
    X,y=mock_data
    sk_model = LinearRegression().fit(X, y)
    sk_weights = np.insert(sk_model.coef_, 0, sk_model.intercept_)
    custom_model= LinearRegressionOLS().fit(X, y)
    np.testing.assert_allclose(custom_model.theta, sk_weights, rtol=1e-4)

def test_ridge_regression(mock_data):
    X,y=mock_data
    alpha=1.0
    sk_model=Ridge(alpha=alpha).fit(X,y)
    sk_weights = np.insert(sk_model.coef_, 0, sk_model.intercept_)
    custom_model=RidgeRegression(alpha=alpha).fit(X, y)
    np.testing.assert_allclose(custom_model.theta, sk_weights, rtol=1e-4)

def test_lasso_regression(mock_data):
    X,y=mock_data
    alpha=1.0
    sk_model=Lasso(alpha=alpha,max_iter=10000,tol=1e-8).fit(X,y)
    sk_weights = np.insert(sk_model.coef_, 0, sk_model.intercept_)
    custom_model= LassoRegression(alpha=alpha,max_iter=10000,tol=1e-8).fit(X, y)
    np.testing.assert_allclose(custom_model.theta, sk_weights, rtol=1e-3,atol=1e-3)
