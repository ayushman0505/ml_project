##read data from db,load model to db...for mongodb
import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill

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
