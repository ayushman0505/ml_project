##read data from db,load model to db...for mongodb
import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    """Saves the object to the specified file path using pickle."""
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(f"Error saving object: {e}", sys)
def evaluate_model(X_train, y_train, X_test, y_test, models,param):
    try:
        model_scores = {}
        for model_name, model in models.items():
            if model_name in param:
                model.set_params(**param[model_name])
            logging.info(f"Training model: {model_name}")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            model_scores[model_name] = score
            logging.info(f"{model_name} r2_score: {score}")
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            model_scores[model_name] = score
            logging.info(f"{model_name} r2_score: {score}")
        return model_scores
    except Exception as e:
        raise CustomException(f"Error evaluating models: {e}", sys)
def load_object(file_path):
       try:
        with open(file_path, 'rb') as file:
            obj = dill.load(file)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
       except Exception as e:
        raise CustomException(f"Error loading object: {e}", sys)    