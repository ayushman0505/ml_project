
import os
import sys
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from src.exception import CustomException
from src.logger import logging
from src.util import save_object, evaluate_model
from sklearn.metrics import r2_score
@dataclass
class ModelTrainerConfig:
    train_data_path: str = os.path.join('artifacts', 'model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self, train_array,test_array,preprocessor_path):
        try:
            logging.info("Loading training data")
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            models={
                "RandomForestClassifier":RandomForestClassifier(),
                "GradientBoostingClassifier":GradientBoostingClassifier(),
                "AdaBoostClassifier":AdaBoostClassifier(),
                "LogisticRegression":LogisticRegression(),
                "SVC":SVC(),
                "DecisionTreeClassifier":DecisionTreeClassifier(),
                "KNeighborsClassifier":KNeighborsClassifier()
            }
            params={
                "RandomForestClassifier":{'n_estimators':100, 'max_depth':10},
                "GradientBoostingClassifier":{'n_estimators':100, 'learning_rate':0.1},
                "AdaBoostClassifier":{'n_estimators':50, 'learning_rate':1.0},
                "LogisticRegression":{'C':1.0, 'solver':'lbfgs'},
                "SVC":{'C':1.0, 'kernel':'rbf'},
                "DecisionTreeClassifier":{'max_depth':5},
                "KNeighborsClassifier":{'n_neighbors':5}
            }
            models_scores = dict=evaluate_model(X_train, y_train, X_test, y_test, models,param=params)
            best_model_score= max(sorted(models_scores.values()))
            best_model_name = list(models_scores.keys())[list(models_scores.values()).index(best_model_score)]
            best_model_= models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No suitable model found with accuracy above 0.5", sys)
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            save_object(self.model_trainer_config.train_data_path, best_model_)
            logging.info(f"Model saved at {self.model_trainer_config.train_data_path}")
            predictions = best_model_.predict(X_test)
            accuracy = r2_score(y_test, predictions)
            logging.info(f"Model accuracy on test data: {accuracy}")
            return accuracy
            
        except Exception as e:
            raise CustomException(e, sys)
