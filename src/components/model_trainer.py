import os
import sys
from dataclasses import dataclass
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoost
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    model_trainer_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:

            X_train,y_train,X_test,y_test = (
            train_arr[:,:-1],
            train_arr[:,-1],
            test_arr[:,:-1],
            test_arr[:,-1]
            )
            logging.info("Assigned Input and Target varaiable")
            models = {
                'Linear Regression' : LinearRegression(),
                'Ridge Regression' : Ridge(),
                'Lasso Regression' : Lasso(),
                'K Neighbours Regressior' : KNeighborsRegressor(),
                'SVM' : SVR(),
                'Decision Tree Regressior' : DecisionTreeRegressor(),
                'Random Forest Regressor' : RandomForestRegressor(),
                'Gradient Boosting Regressor' : GradientBoostingRegressor(),
                'AdaBoost Regressor' : AdaBoostRegressor(),
                "XGB Regressor" : XGBRegressor(),
                'CatBoost' : CatBoost(),
            }

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model found")
            
            logging.info('Best Mode and Score are found on training and testinf data')

            save_object(file_path=self.model_trainer_config.model_trainer_path, obj = best_model)

            predicted = best_model.predict(X_test)
            return r2_score(y_test,predicted)


        except Exception as e:
            raise CustomException(e,sys)
