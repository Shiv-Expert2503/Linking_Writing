import os
from dataclasses import dataclass

import polars as pl
from src.exception import CustomException
import sys
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data: str = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'artifacts', 'train_logs.csv'))
    train_score: str = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'artifacts', 'train_scores.csv'))


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        :return: the LazyFrame object
        """
        try:
            logging.info('Fetching the data paths and reading the data')
            print(self.ingestion_config.train_data, self.ingestion_config.train_score)

            train_logs = pl.scan_csv(self.ingestion_config.train_data)
            train_scores = pl.scan_csv(self.ingestion_config.train_score)

            logging.info('Data paths fetched successfully Now merging them')

            train_logs = train_logs.join(train_scores, on='id', how='left')

            logging.info('Data merged successfully')
            print(type(train_logs))
            return train_logs

        except Exception as e:
            logging.error("Error while fetching the data paths", exc_info=True)
            raise CustomException(e, sys)
