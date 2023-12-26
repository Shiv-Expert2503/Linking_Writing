import os
import sys
from src.exception import CustomException
from src.logger import logging
# from src.utils import save_object, evaluate_model
from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_path = os.path.abspath(os.path.join(os.getcwd(), 'artifacts', 'model.pkl'))

class ModelTrainer:

    def __init__(self):
        self.model_path = ModelConfig()
        self.models = {

        }
