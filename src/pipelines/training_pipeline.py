import sys

from src.components.data_injestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model import ModelTrainer
from src.exception import CustomException
from src.logger import logging

if __name__ == '__main__':
    try:
        logging.info("Training Pipeline Started")
        datainject = DataIngestion()
        data_frame = datainject.initiate_data_ingestion()

        # Object of data transformation class
        dd = DataTransformation()
        data = dd.initiate_data_transformation(data_frame)

        model = ModelTrainer()
        model.initiate_model_training(data)

    except Exception as e:
        logging.error("Error occurred In training pipeline", exc_info=True)
        raise CustomException(e, sys)
