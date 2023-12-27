import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object


@dataclass
class ModelConfig:
    model_path = os.path.abspath(os.path.join(os.getcwd(), 'artifacts', 'model.pkl'))


class ModelTrainer:

    def __init__(self):
        self.modelconfig = ModelConfig()
        self.base_regressors = [
            ('LGBM', LGBMRegressor(learning_rate=0.1, max_bin=511, max_depth=5, min_child_samples=30, n_estimators=50,
                                   num_leaves=31)),
            ('CatBoost', CatBoostRegressor(depth=5, l2_leaf_reg=5, learning_rate=0.1, n_estimators=100)),
            ('RandomForest',
             RandomForestRegressor(max_depth=7, max_features='sqrt', max_leaf_nodes=50, n_estimators=200)),
            ('SVR', SVR(C=1, gamma='auto', kernel='rbf')),
            ('XGB', XGBRegressor())
        ]
        self.model = StackingRegressor(estimators=self.base_regressors, final_estimator=LinearRegression())

    def initiate_model_training(self, df):
        try:
            logging.info('Training started')
            x_train, x_test, y_train, y_test = train_test_split(df.drop(['id', 'score'], axis=1), df['score'],
                                                                test_size=0.2, random_state=42)
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            logging.info("Fitting Started")
            self.model.fit(x_train_scaled, y_train)

            logging.info(f" The stacking model has a rmse of {evaluate_model(x_test_scaled, y_test, self.model)}")
            logging.info("Fitting completed")

            logging.info("Now saving the model")
            save_object(
                path=self.modelconfig.model_path,
                obj=self.model
            )
            logging.info("Model saved successfully")
        except Exception as e:
            logging.error('Error occurred while training the model', exc_info=True)
            raise CustomException(e, sys)
