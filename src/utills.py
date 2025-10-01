import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import optuna
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessingStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            # fillna with median for numerical product features
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)

            # replace missing reviews with text
            data["review_comment_message"].fillna("No review", inplace=True)

            # keep only numerical columns
            data = data.select_dtypes(include=[np.number])

            # drop identifiers not useful for modeling
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error(e)
            raise e


class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e


class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        return self.strategy.handle_data(self.df)


class Model(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        pass


class LinearRegresionModel(Model):  # Typo: should be LinearRegressionModel
    def train(self, x_train, y_train, **kwargs):
        reg = LinearRegression(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        reg = self.train(x_train, y_train)
        return reg.score(x_test, y_test)


class RandomForestModel(Model):  # Typo: should be RandomForestModel
    """
    Random Forest model
    """

    def train(self, x_train, y_train, **kwargs):
        reg = RandomForestRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        reg = self.train(
            x_train,
            y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        return reg.score(x_test, y_test)


class XGBoostModel(Model):
    """
    XGBoost model
    """

    def train(self, x_train, y_train, **kwargs):
        reg = xgb.XGBRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        #  min_samples_split is not valid for XGBRegressor
        reg = self.train(
            x_train,
            y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        return reg.score(x_test, y_test)


class LightGBMModel(Model):
    """
    LightGBM model
    """

    def train(self, x_train, y_train, **kwargs):
        reg = LGBMRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        
        reg = self.train(
            x_train,
            y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        return reg.score(x_test, y_test)


class HyperParameter:
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.model.optimize(
                trial, self.x_train, self.y_train, self.x_test, self.y_test
            ),
            n_trials=n_trials,
        )
        return study.best_trial.params


class  Evaluation(ABC):
    @abstractmethod
    def calculate_score(self, y_pred:np.ndarray, y_true:np.ndarray)->float:
        pass

class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error (MSE)
    """
    def calculate_score(self, y_pred:np.ndarray, y_true:np.ndarray)->float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        try:
            logging.info('calculate_score method of the MSE class')
            mse= mean_squared_error(y_true,y_pred)
            logging.info(f'mean squared error is {mse}')
            return mse
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the MSE class. Exception message:  "
                + str(e)
            )
            raise e
        
class R2score(Evaluation):
    """
    Evaluation strategy that uses R2 Score
    """
    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray)->float:
        
        try:
            logging.info('calculate_score method of R2_score')
            r2score= r2_score(y_true,y_pred)
            logging.info(f'r2 score is {r2score}')
            return r2score
        
        except Exception as e:
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Squared Error (RMSE)
    """
    def calculate_score(self, y_pred, y_true):
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            rmse: float
        """
        try:
            logging.info('calculating the rmse score')
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f'rmse is {rmse}')
            return rmse 
            
        except Exception as e:
            raise e  