import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

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
                'Decision Tree Regressior' : DecisionTreeRegressor(),
                'Random Forest Regressor' : RandomForestRegressor(),
                'Gradient Boosting Regressor' : GradientBoostingRegressor(),
                'AdaBoost Regressor' : AdaBoostRegressor(),
                "XGB Regressor" : XGBRegressor(),
                'CatBoost' : CatBoostRegressor(verbose=False),
            }

            params = {
                'Linear Regression' : {},
                'Decision Tree' : {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'splitter':['best','random'],
                    #'max_features':['sqrt','log2'],
                },
                'Random Forest' : {
                    #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Gradient Boosting' : {
                    #'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    #'criterion':['squared_error', 'friedman_mse'],
                    #'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'AdaBoost' : {
                    'learning_rate':[.1,.01,0.5,.001],
                    #'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'XGB Regressor' : {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'CatBoost' : {
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }
            }


            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,parameters=params)

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model found")
            
            logging.info('Best Mode and Score are found on training and testinf data')

            save_object(file_path=self.model_trainer_config.model_trainer_path, obj = best_model)

            predicted = best_model.predict(X_test)
            return best_model_name,r2_score(y_test,predicted)


        except Exception as e:
            raise CustomException(e,sys)
