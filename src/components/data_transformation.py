import pandas as pd
import numpy as np
import logging
from typing import Tuple
from zenml import step
from src.utills import DataCleaning, DataDivideStrategy, DataPreprocessingStrategy
from typing_extensions import Annotated

@step
def clean_data(data: pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    try:
        preprocess_strategy = DataPreprocessingStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocesed_data = data_cleaning.handle_data()
        divide_strategy = DataDivideStrategy()
        data_cleaning= DataCleaning(preprocesed_data, divide_strategy)
        x_train, x_test, y_train, y_test= data_cleaning.handle_data()
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e