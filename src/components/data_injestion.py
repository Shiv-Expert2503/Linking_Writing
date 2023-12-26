import os
from dataclasses import dataclass
from src.exception import CustomException
import sys
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data: str = os.path.abspath(os.path.join(os.getcwd(), 'artifacts', 'train_logs.csv'))
    train_score: str = os.path.abspath(os.path.join(os.getcwd(), 'artifacts', 'train_scores.csv'))


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        :return: the data path of train and score
        """
        try:
            logging.info('Fetching the data paths')
            print(self.ingestion_config.train_data, self.ingestion_config.train_score)
            logging.info('Data paths fetched successfully')
            return self.ingestion_config.train_data, self.ingestion_config.train_score
        except Exception as e:
            logging.error("Error while fetching the data paths", exc_info=True)
            raise CustomException(e, sys)


