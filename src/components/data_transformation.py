import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
import os
import pickle
from src.util import save_object
# Removed unused import to fix circular import
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''Method to initiate data transformation pipeline'''
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical and categorical feature transformation pipelines created")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Data Transformation initiated")   
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print(f"Train columns: {train_df.columns.tolist()}")
            print(f"Test columns: {test_df.columns.tolist()}")
            logging.info(f"Train columns: {train_df.columns.tolist()}")
            logging.info(f"Test columns: {test_df.columns.tolist()}")
            print("Train and test data loaded successfully")
            logging.info("Train and test data loaded successfully")
            print("Obtaining preprocessor object")
            logging.info("Obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformer_object()   
            target_column_name = 'math_score'
            numerical_features = ['writing_score', 'reading_score']
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            print(f"Input feature train columns: {input_feature_train_df.columns.tolist()}")
            print(f"Input feature test columns: {input_feature_test_df.columns.tolist()}")
            logging.info(f"Input feature train columns: {input_feature_train_df.columns.tolist()}")
            logging.info(f"Input feature test columns: {input_feature_test_df.columns.tolist()}")
            print("Applying preprocessor on training and testing dataframes")
            logging.info("Applying preprocessor on training and testing dataframes")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            print(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            print(f"Length of target_feature_train_df: {len(target_feature_train_df)}")
            print(f"Shape of input_feature_test_arr: {input_feature_test_arr.shape}")
            print(f"Length of target_feature_test_df: {len(target_feature_test_df)}")
            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            print("Data transformation completed successfully")
            logging.info("Data transformation completed successfully")
            print("Saving preprocessor object")
            logging.info("Saving preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return(
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_transformation = DataTransformation()
    train_path = os.path.join("artifacts", "train.csv")
    test_path = os.path.join("artifacts", "test.csv")
    data_transformation.initiate_data_transformation(train_path, test_path)
