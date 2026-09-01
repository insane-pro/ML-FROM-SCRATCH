import sys 
from  sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error
from src.linear import LinearRegressionOLS
from src.regularization import RidgeRegression,LassoRegression
from src.logger import logging
from src.exception import CustomException
from src.pipeline import get_real_world_data,simple_train_test_split,standardize 


def run_benchmark():
    try:
        logging.info(" starting MLOps benchmarking program")

        logging.info(" Loading and splitting data")
        X,y=get_real_world_data()
        X_train,X_test,y_train,y_test=simple_train_test_split(X,y)

        logging.info("Standardization of features")
        X_train_scaled,X_test_scaled,_,_=standardize(X_train,X_test)

        models={
            "OLS":(LinearRegressionOLS(),LinearRegression()),
            "Ridge":(RidgeRegression(alpha=1.0),Ridge(alpha=1.0)),
            "Lasso":(LassoRegression(alpha=0.1,max_iter=10000),Lasso(alpha=0.1,max_iter=10000)),
        }
        for name , (custom_model,sk_model ) in models.items():
            logging.info(f"----benchmarking:{name}----")

            custom_model.fit(X_train_scaled,y_train)
            custom_preds=custom_model.predict(X_test_scaled)
            custom_mse=mean_squared_error(y_test,custom_preds)

            sk_model.fit(X_train_scaled,y_train)
            sk_preds=sk_model.predict(X_test_scaled)
            sk_mse=mean_squared_error(y_test,sk_preds)

            logging.info(f"Custom {name} MSE:  {custom_mse:.4f}")
            logging.info(f"Sklearn {name} MSE: {sk_mse:.4f}")
            logging.info(f"Absolute Diff:      {abs(custom_mse - sk_mse):.6f}")
        
        logging.info("benchmarking complete")
    
    except  Exception as e:
        logging.info("an exception that occured during benchmarking")
        raise CustomException(e,sys)

if __name__=="__main__":
    run_benchmark()