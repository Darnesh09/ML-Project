import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing score','reading score']
            categorical_features = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']
            num_pipeline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info('numerical feature imputation and scaling is done')
            cat_pipeline  = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info('categorical features imputation and scaling done')

            preprocessor = ColumnTransformer(
                transformers = [
                    ('num_pipeline',num_pipeline,numerical_features),
                    ('cat_pipeline',cat_pipeline,categorical_features)
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data')
            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformer_object()

            target_feature = 'math score'
            input_feature_train_df = train_df.drop(target_feature,axis=1)
            target_feature_train_df = train_df[target_feature]
            input_feature_test_df = test_df.drop(target_feature,axis=1)
            target_feature_test_df = test_df[target_feature]
            
            logging.info('Applying preprocessing object on traing and testing data')
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)

            logging.info('Saved preprocessing object')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)