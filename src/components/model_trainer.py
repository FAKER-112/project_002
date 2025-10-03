import logging 
import mlflow
import pandas as pd
from src.utills import HyperParameter, LinearRegresionModel, LightGBMModel, XGBoostModel,RandomForestModel
from .config import ModelNameConfig
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger


experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def train(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
)-> RegressorMixin:
    try:
        model= None
        tuner= None
        if config.model_name == "LinearRegression":
            model= LinearRegresionModel()
            mlflow.sklearn.autolog()
        elif config.model_name == "lightgbm":
            model= LightGBMModel()  
            mlflow.lightgbm.autolog()
        elif config.model_name == "XGBoost":
            model= XGBoostModel()
            mlflow.xgboost.autolog()
        elif config.model_name == "RandomForest":
            model= RandomForestModel()
            mlflow.sklearn.autolog()
        else:
            raise ValueError("Model name not supported")

        tuner= HyperParameter(model,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)

        if config.fine_tuning:
            best_param= tuner.optimize()
            trained_model= model.train(x_train, y_train, **best_param)
        else:
            trained_model = model.train(x_train, y_train)
        return trained_model
    except Exception as e:
        logging.error(e)
        raise e