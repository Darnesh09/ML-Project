import os
import sys
from src.exception import CustomException
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models:dict,parameters:dict):
    try:
        result = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = parameters[list(parameters)[i]]
            gs = GridSearchCV(model,param,cv=5)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            result[list(models)[i]] = r2_score(y_test,pred)
        return result

    except Exception as e:
        raise CustomException(e,sys)

def load_file(file_path):
    try:
        with open(file_path,'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e,sys)