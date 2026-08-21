import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import PolynomialFeatures

def get_real_world_data():
    X,y=load_diabetes(return_X_y=True)
    poly=PolynomialFeatures(degree=2,include_bias=False)
    X_poly=poly.fit_transform(X)
    return X_poly,y

def simple_train_test_split(X,y,test_size=0.3,seed=42):
    rng=np.random.default_rng(seed)
    indicies=rng.permutation(X.shape[0])
    split_idx=int((X.shape[0])*(1-test_size))
    train_idx,test_idx=indicies[:split_idx],indicies[split_idx:]
    return X[train_idx],X[test_idx],y[train_idx],y[test_idx]
def standardize(X_train,X_Test):
    mean_=X_train.mean(axis=0)
    std_=X_train.std(axis=0)
    std_=np.where(std_==0.0,1.0,std_)
    return (X_train-mean_)/std_,(X_Test-mean_)/std_,mean_,std_
