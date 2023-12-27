import os
import pickle
import sys

import numpy as np
from sklearn.metrics import mean_squared_error

from src.exception import CustomException
from src.logger import logging


def evaluate_model(x_test, y_test, model) -> float:
    """
    Evaluates the model
    :param x_test: it should be scaled
    :param y_test: the ground scores
    :param model: trained model
    :return: the root mean squared error in float format
    """
    try:
        logging.info("Evaluating the model")
        preds = model.predict(x_test)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        logging.info("Evaluation completed")
        return rmse
    except Exception as e:
        logging.error("Error occurred while evaluating the model", exc_info=e)
        raise CustomException(e, sys)


def save_object(path: str, obj):
    """
    Saves the object in the given path  in pickle format
    :param path: the path where the object should be saved
    :param obj: the object that needs to be saved
    """
    try:
        logging.info(f"Dumping starts of {str(obj)}")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Dumping Completed")

    except Exception as e:
        logging.info("Error Occurred while dumping")
        raise CustomException(e, sys)
